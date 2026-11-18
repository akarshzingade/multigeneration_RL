"""Modal deployment script for running Experiment 1.1 on Nvidia L4 or A10G GPUs.

Usage:
  # 1. Quick test run with reduced FLOPs on L4:
  modal run scripts/modal_grpo_runner.py --target-flops 1e14

  # 2. Single full run on L4 (Seed 42):
  modal run scripts/modal_grpo_runner.py --seed 42

  # 3. Parallel run across all 3 seeds concurrently on L4:
  modal run scripts/modal_grpo_runner.py --run-all-seeds

  # 4. To use A10G instead of L4 (requires credit card added on modal.com/settings):
  MODAL_GPU=A10G modal run scripts/modal_grpo_runner.py --seed 42
"""

from __future__ import annotations

import os
import subprocess
import sys
import modal

app = modal.App("multigen-exp1-1-grpo")


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
        "trl>=0.14.0",
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


def _execute_grpo(
    seed: int,
    target_flops: float,
    gpu_tag: str,
    no_flop_constraint: bool = False,
    config: str = "",
    termination_criterion: str = "",
    epochs: float = 0.0,
    max_steps: int = 0,
    patience: int = 0,
    run_tag: str = "",
) -> str:
    """Internal helper to execute Experiment 1.1 within container."""
    os.chdir("/root/multigen-rl")
    os.environ["PYTHONPATH"] = "/root/multigen-rl"
    # Persist Hugging Face dataset & model weights cache across runs
    os.environ["HF_HOME"] = "/root/storage/hf_cache"

    cfg = config
    if not cfg:
        cfg = (
            "configs/experiments/exp1_1_grpo_unconstrained.yaml"
            if (no_flop_constraint or target_flops <= 0)
            else "configs/experiments/exp1_1_grpo_baseline.yaml"
        )

    out_tag = run_tag.strip().lstrip("_") if run_tag else (
        "unconstrained" if (no_flop_constraint or target_flops <= 0) else "baseline"
    )
    out_dir = f"/root/storage/checkpoints/exp1_1_grpo_{out_tag}_seed{seed}"
    dag_path = "/root/storage/population_dag.json"
    eval_dir = "/root/storage/evals"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    cmd = [
        "python",
        "scripts/run_grpo_experiment.py",
        "--config",
        cfg,
        "--seed",
        str(seed),
        "--run-tag",
        out_tag,
        "--output-dir",
        out_dir,
        "--dag-path",
        dag_path,
        "--eval-output-dir",
        eval_dir,
    ]

    if no_flop_constraint or target_flops <= 0:
        cmd.append("--no-flop-constraint")
    else:
        cmd.extend(["--target-flops", str(target_flops)])

    if termination_criterion:
        cmd.extend(["--termination-criterion", termination_criterion])
    if epochs > 0:
        cmd.extend(["--epochs", str(epochs)])
    if max_steps > 0:
        cmd.extend(["--max-steps", str(max_steps)])
    if patience > 0:
        cmd.extend(["--patience", str(patience)])

    subprocess.run(cmd, check=True)
    volume.commit()
    return f"Completed EXP-1.1 seed {seed} on {gpu_tag}. Artifacts committed to volume at {out_dir}"


@app.function(
    gpu=TARGET_GPU,
    image=image,
    volumes={"/root/storage": volume},
    timeout=7200,  # 2 hours maximum
)
def run_grpo(
    seed: int = 42,
    target_flops: float = 1.0e16,
    gpu_tag: str = "",
    no_flop_constraint: bool = False,
    config: str = "",
    termination_criterion: str = "",
    epochs: float = 0.0,
    max_steps: int = 0,
    patience: int = 0,
    run_tag: str = "",
) -> str:
    """Execute EXP-1.1 on the configured GPU."""
    active_gpu = gpu_tag or TARGET_GPU
    return _execute_grpo(
        seed=seed,
        target_flops=target_flops,
        gpu_tag=active_gpu,
        no_flop_constraint=no_flop_constraint,
        config=config,
        termination_criterion=termination_criterion,
        epochs=epochs,
        max_steps=max_steps,
        patience=patience,
        run_tag=run_tag,
    )


@app.local_entrypoint()
def main(
    seed: int = 42,
    target_flops: float = 1.0e16,
    gpu: str = "",
    gpu_type: str = "",
    run_all_seeds: bool = False,
    no_flop_constraint: bool = False,
    config: str = "",
    termination_criterion: str = "",
    epochs: float = 0.0,
    max_steps: int = 0,
    patience: int = 0,
    run_tag: str = "",
):
    """Local CLI entrypoint for launching Modal training."""
    active_gpu = (gpu or gpu_type or TARGET_GPU).upper()
    print(f"Target GPU configured: {active_gpu}")
    budget_label = "Unconstrained (No FLOP limit)" if (no_flop_constraint or target_flops <= 0) else f"{target_flops:.1e} FLOPs"
    resolved_tag = run_tag.strip().lstrip("_") if run_tag else (
        "unconstrained" if (no_flop_constraint or target_flops <= 0) else "baseline"
    )

    if run_all_seeds:
        seeds = [42, 43, 44]
        print(f"Launching EXP-1.1 concurrently across 3 seeds {seeds} on {active_gpu} (Tag: {resolved_tag}, Budget: {budget_label})...")
        args = [
            (s, target_flops, active_gpu, no_flop_constraint, config, termination_criterion, epochs, max_steps, patience, resolved_tag)
            for s in seeds
        ]
        results = list(run_grpo.starmap(args))
        for res in results:
            print("  ->", res)
        print("\nAll seeds completed! Download results with:")
        print("  modal volume get multigen-storage evals/ ./evals/")
        print("  modal volume get multigen-storage population_dag.json ./population_dag.json")
    else:
        print(f"Launching EXP-1.1 on Modal (GPU: {active_gpu}, Seed: {seed}, Tag: {resolved_tag}, Budget: {budget_label})...")
        result = run_grpo.remote(
            seed=seed,
            target_flops=target_flops,
            gpu_tag=active_gpu,
            no_flop_constraint=no_flop_constraint,
            config=config,
            termination_criterion=termination_criterion,
            epochs=epochs,
            max_steps=max_steps,
            patience=patience,
            run_tag=resolved_tag,
        )
        print("Result:", result)
        print("\nDownload results with:")
        print(f"  modal volume get multigen-storage evals/exp1_1_grpo_{resolved_tag}_seed{seed}_test.json ./evals/")
        print("  modal volume get multigen-storage population_dag.json ./population_dag.json")
