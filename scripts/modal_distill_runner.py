"""Modal deployment script for running Experiment 1.2 on Nvidia L4 or A10G GPUs.

Usage:
  # 1. Generate & verify teacher traces only (Stage 1):
  modal run scripts/modal_distill_runner.py --generate-traces-only

  # 2. Single full run on L4 (Seed 42):
  modal run scripts/modal_distill_runner.py --seed 42

  # 3. Parallel run across all 3 seeds concurrently on L4:
  modal run scripts/modal_distill_runner.py --run-all-seeds

  # 4. Quick test run with reduced FLOPs on L4:
  modal run scripts/modal_distill_runner.py --target-flops 1e14
"""

from __future__ import annotations

import os
import subprocess
import sys
import modal

app = modal.App("multigen-exp1-2-distill")


def _get_target_gpu() -> str:
    """Parse GPU choice from CLI flags (--gpu or --gpu-type) or env var, defaulting to L4."""
    for i, arg in enumerate(sys.argv):
        if arg in ("--gpu", "--gpu-type") and i + 1 < len(sys.argv):
            return sys.argv[i + 1].upper()
        if arg.startswith("--gpu="):
            return arg.split("=", 1)[1].upper()
        if arg.startswith("--gpu-type="):
            return arg.split("=", 1)[1].upper()
    return os.environ.get("MODAL_GPU", "L4").upper()


TARGET_GPU = _get_target_gpu()
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "trl>=0.8.0,<0.14.0",
        "peft>=0.10.0",
        "vllm>=0.4.0",
        "math-verify>=0.1.0",
        "datasets>=2.18.0",
        "networkx>=3.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
    )
    .env({
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V1": "0",
    })
    .add_local_dir("multigen", remote_path="/root/multigen-rl/multigen")
    .add_local_dir("scripts", remote_path="/root/multigen-rl/scripts")
    .add_local_dir("configs", remote_path="/root/multigen-rl/configs")
    .add_local_file("pyproject.toml", remote_path="/root/multigen-rl/pyproject.toml")
)

volume = modal.Volume.from_name("multigen-storage", create_if_missing=True)


def _execute_distill(
    seed: int,
    target_flops: float,
    gpu_tag: str,
    generate_traces_only: bool = False,
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
    skip_eval: bool = False,
) -> str:
    """Internal helper to execute Experiment 1.2 within container."""
    os.chdir("/root/multigen-rl")
    os.environ["PYTHONPATH"] = "/root/multigen-rl"
    os.environ["HF_HOME"] = "/root/storage/hf_cache"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_V1"] = "0"

    out_dir = f"/root/storage/checkpoints/exp1_2_distill_seed{seed}"
    dag_path = "/root/storage/population_dag.json"
    eval_dir = "/root/storage/evals"

    if max_train_samples > 0:
        traces_path = f"/root/storage/data/teacher_traces_gsm8k_test_{max_train_samples}.jsonl"
    else:
        traces_path = "/root/storage/data/teacher_traces_gsm8k.jsonl"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs("/root/storage/data", exist_ok=True)

    if generate_traces_only:
        trace_cmd = [
            "python",
            "scripts/run_distill_experiment.py",
            "--config",
            "configs/experiments/exp1_2_distill_baseline.yaml",
            "--generate-traces-only",
            "--traces-path",
            traces_path,
        ]
        if max_train_samples > 0:
            trace_cmd.extend(["--max-train-samples", str(max_train_samples)])
        subprocess.run(trace_cmd, check=True)
        volume.commit()
        return f"Completed Stage 1: Verified teacher traces saved and committed to {traces_path}."

    # Stage 1: Ensure verified teacher traces exist on the persistent volume in a dedicated process
    # Running Stage 1 in its own subprocess guarantees the OS reclaims 100% of the 18 GB teacher GPU memory
    if not os.path.exists(traces_path) or os.path.getsize(traces_path) == 0:
        print(f"[{gpu_tag}] Generating teacher traces in isolated Stage 1 process...")
        trace_cmd = [
            "python",
            "scripts/run_distill_experiment.py",
            "--config",
            "configs/experiments/exp1_2_distill_baseline.yaml",
            "--generate-traces-only",
            "--traces-path",
            traces_path,
        ]
        if max_train_samples > 0:
            trace_cmd.extend(["--max-train-samples", str(max_train_samples)])
        subprocess.run(trace_cmd, check=True)
        volume.commit()
        print(f"[{gpu_tag}] Stage 1 complete: Teacher traces committed to volume at {traces_path}.")
    else:
        print(f"[{gpu_tag}] Found cached verified teacher traces at {traces_path}. Skipping Stage 1 generation.")

    # Stage 2: Student SFT training and evaluation in a fresh process with 100% free GPU memory
    cmd = [
        "python",
        "scripts/run_distill_experiment.py",
        "--config",
        "configs/experiments/exp1_2_distill_baseline.yaml",
        "--seed",
        str(seed),
        "--target-flops",
        str(target_flops),
        "--output-dir",
        out_dir,
        "--dag-path",
        dag_path,
        "--eval-output-dir",
        eval_dir,
        "--traces-path",
        traces_path,
    ]
    if max_train_samples > 0:
        cmd.extend(["--max-train-samples", str(max_train_samples)])
    if max_eval_samples > 0:
        cmd.extend(["--max-eval-samples", str(max_eval_samples)])
    if skip_eval:
        cmd.append("--skip-eval")

    subprocess.run(cmd, check=True)
    volume.commit()
    return f"Completed EXP-1.2 seed {seed} on {gpu_tag}. Artifacts committed to volume at {out_dir}"


@app.function(
    gpu=TARGET_GPU,
    image=image,
    volumes={"/root/storage": volume},
    timeout=86400,
)
def run_distill(
    seed: int = 42,
    target_flops: float = 1.0e16,
    generate_traces_only: bool = False,
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
    skip_eval: bool = False,
    gpu_name: str = "L4",
) -> str:
    """Run single seed of Experiment 1.2."""
    import torch

    physical_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Target GPU configured: {gpu_name} (Physical device: {physical_gpu})")
    print(
        f"Launching EXP-1.2 on Modal (GPU: {gpu_name}, Physical: {physical_gpu}, Seed: {seed}, Budget: {target_flops:.1e} FLOPs)..."
    )
    return _execute_distill(
        seed=seed,
        target_flops=target_flops,
        gpu_tag=f"{gpu_name} ({physical_gpu})",
        generate_traces_only=generate_traces_only,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        skip_eval=skip_eval,
    )


@app.local_entrypoint()
def main(
    seed: int = 42,
    target_flops: float = 1.0e16,
    run_all_seeds: bool = False,
    generate_traces_only: bool = False,
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
    skip_eval: bool = False,
    gpu: str = "L4",
) -> None:
    """Local CLI entrypoint for triggering Modal runs."""
    selected_gpu = gpu.upper() if gpu else TARGET_GPU

    if generate_traces_only:
        print(f"Dispatching Stage 1: Generate & verify teacher traces on {selected_gpu}...")
        res = run_distill.remote(
            seed=seed,
            target_flops=target_flops,
            generate_traces_only=True,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            skip_eval=skip_eval,
            gpu_name=selected_gpu,
        )
        print(f"✓ {res}")
        return

    if run_all_seeds:
        seeds = [42, 43, 44]
        print(
            f"Dispatching parallel EXP-1.2 runs across seeds {seeds} on {selected_gpu}..."
        )
        results = list(
            run_distill.starmap(
                [
                    (
                        s,
                        target_flops,
                        False,
                        max_train_samples,
                        max_eval_samples,
                        skip_eval,
                        selected_gpu,
                    )
                    for s in seeds
                ]
            )
        )
        for r in results:
            print(f"✓ {r}")
    else:
        print(f"Dispatching single EXP-1.2 run for seed {seed} on {selected_gpu}...")
        res = run_distill.remote(
            seed=seed,
            target_flops=target_flops,
            generate_traces_only=False,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            skip_eval=skip_eval,
            gpu_name=selected_gpu,
        )
        print(f"✓ {res}")
