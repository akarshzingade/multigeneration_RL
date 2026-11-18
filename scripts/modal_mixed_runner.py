"""Modal deployment script for running Experiment 1.3 on Nvidia L4 GPUs.

Usage:
  # 1. Single full run on Nvidia L4 (Seed 42 with auto-resumption):
  modal run scripts/modal_mixed_runner.py --seed 42

  # 2. Parallel run across all 3 seeds (42, 43, 44) concurrently:
  modal run scripts/modal_mixed_runner.py --run-all-seeds

  # 3. Quick test run with reduced FLOPs:
  modal run scripts/modal_mixed_runner.py --target-flops 1e14

  # 4. Use A10G instead of L4:
  MODAL_GPU=A10G modal run scripts/modal_mixed_runner.py --seed 42
"""

from __future__ import annotations

import os
import subprocess
import sys
import modal

app = modal.App("multigen-exp1-3-mixed")


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


def _execute_mixed(
    seed: int,
    target_flops: float,
    gpu_tag: str,
    mode: str = "interleaved_sequential",
    resume: bool = True,
    skip_eval: bool = False,
    num_eval_samples: int = 0,
    no_flop_constraint: bool = False,
    config: str = "",
    stage_transition_criterion: str = "",
    termination_criterion: str = "",
    sft_epochs: float = 0.0,
    sft_steps: int = 0,
    sft_patience: int = 0,
    grpo_epochs: float = 0.0,
    grpo_steps: int = 0,
    grpo_patience: int = 0,
    patience: int = 0,
    run_tag: str = "",
) -> str:
    """Internal helper to execute Experiment 1.3 within Modal container."""
    os.chdir("/root/multigen-rl")
    os.environ["PYTHONPATH"] = "/root/multigen-rl"
    os.environ["HF_HOME"] = "/root/storage/hf_cache"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_V1"] = "0"

    cfg = config
    if not cfg:
        cfg = (
            "configs/experiments/exp1_3_mixed_unconstrained.yaml"
            if (no_flop_constraint or target_flops <= 0)
            else "configs/experiments/exp1_3_mixed_learning.yaml"
        )

    out_tag = run_tag.strip().lstrip("_") if run_tag else (
        "unconstrained" if (no_flop_constraint or target_flops <= 0) else "baseline"
    )
    out_dir = f"/root/storage/checkpoints/exp1_3_mixed_{out_tag}_seed{seed}"
    dag_path = "/root/storage/population_dag.json"
    eval_dir = "/root/storage/evals"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs("/root/storage/data", exist_ok=True)

    traces_path = "/root/storage/data/teacher_traces_gsm8k.jsonl"

    cmd = [
        "python",
        "scripts/run_mixed_experiment.py",
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
        "--mode",
        mode,
        "--traces-path",
        traces_path,
    ]

    if no_flop_constraint or target_flops <= 0:
        cmd.append("--no-flop-constraint")
    elif target_flops > 0:
        cmd.extend(["--target-flops", str(target_flops)])

    if stage_transition_criterion:
        cmd.extend(["--stage-transition-criterion", stage_transition_criterion])
    if termination_criterion:
        cmd.extend(["--termination-criterion", termination_criterion])
    if sft_epochs > 0:
        cmd.extend(["--sft-epochs", str(sft_epochs)])
    if sft_steps > 0:
        cmd.extend(["--sft-steps", str(sft_steps)])
    if sft_patience > 0:
        cmd.extend(["--sft-patience", str(sft_patience)])
    if grpo_epochs > 0:
        cmd.extend(["--grpo-epochs", str(grpo_epochs)])
    if grpo_steps > 0:
        cmd.extend(["--grpo-steps", str(grpo_steps)])
    if grpo_patience > 0:
        cmd.extend(["--grpo-patience", str(grpo_patience)])
    if patience > 0:
        cmd.extend(["--patience", str(patience)])

    if resume:
        cmd.append("--resume")
    else:
        cmd.append("--no-resume")
    if skip_eval:
        cmd.append("--skip-eval")
    if num_eval_samples > 0:
        cmd.extend(["--num-eval-samples", str(num_eval_samples)])

    try:
        subprocess.run(cmd, check=True)
    finally:
        # Guarantee volume commit even upon interruption or error
        volume.commit()

    return f"Completed EXP-1.3 seed {seed} on {gpu_tag}. Artifacts synced to volume at {out_dir}"


@app.function(
    gpu=TARGET_GPU,
    timeout=86400,
    volumes={"/root/storage": volume},
    image=image,
)
def run_single_seed(
    seed: int = 42,
    target_flops: float = 1.0e16,
    mode: str = "interleaved_sequential",
    resume: bool = True,
    skip_eval: bool = False,
    num_eval_samples: int = 0,
    no_flop_constraint: bool = False,
    config: str = "",
    stage_transition_criterion: str = "",
    termination_criterion: str = "",
    sft_epochs: float = 0.0,
    sft_steps: int = 0,
    sft_patience: int = 0,
    grpo_epochs: float = 0.0,
    grpo_steps: int = 0,
    grpo_patience: int = 0,
    patience: int = 0,
    run_tag: str = "",
) -> str:
    """Execute Experiment 1.3 for a single seed."""
    return _execute_mixed(
        seed=seed,
        target_flops=target_flops,
        gpu_tag=TARGET_GPU,
        mode=mode,
        resume=resume,
        skip_eval=skip_eval,
        num_eval_samples=num_eval_samples,
        no_flop_constraint=no_flop_constraint,
        config=config,
        stage_transition_criterion=stage_transition_criterion,
        termination_criterion=termination_criterion,
        sft_epochs=sft_epochs,
        sft_steps=sft_steps,
        sft_patience=sft_patience,
        grpo_epochs=grpo_epochs,
        grpo_steps=grpo_steps,
        grpo_patience=grpo_patience,
        patience=patience,
        run_tag=run_tag,
    )


@app.function(
    gpu=TARGET_GPU,
    timeout=86400,
    volumes={"/root/storage": volume},
    image=image,
)
def run_parallel_seed(
    seed: int,
    target_flops: float = 1.0e16,
    mode: str = "interleaved_sequential",
    resume: bool = True,
    skip_eval: bool = False,
    num_eval_samples: int = 0,
    no_flop_constraint: bool = False,
    config: str = "",
    stage_transition_criterion: str = "",
    termination_criterion: str = "",
    sft_epochs: float = 0.0,
    sft_steps: int = 0,
    sft_patience: int = 0,
    grpo_epochs: float = 0.0,
    grpo_steps: int = 0,
    grpo_patience: int = 0,
    patience: int = 0,
    run_tag: str = "",
) -> str:
    """Function spawned concurrently across 3 seeds on separate container workers."""
    return _execute_mixed(
        seed=seed,
        target_flops=target_flops,
        gpu_tag=TARGET_GPU,
        mode=mode,
        resume=resume,
        skip_eval=skip_eval,
        num_eval_samples=num_eval_samples,
        no_flop_constraint=no_flop_constraint,
        config=config,
        stage_transition_criterion=stage_transition_criterion,
        termination_criterion=termination_criterion,
        sft_epochs=sft_epochs,
        sft_steps=sft_steps,
        sft_patience=sft_patience,
        grpo_epochs=grpo_epochs,
        grpo_steps=grpo_steps,
        grpo_patience=grpo_patience,
        patience=patience,
        run_tag=run_tag,
    )


@app.local_entrypoint()
def main(
    seed: int = 42,
    gpu: str = "L4",
    run_all_seeds: bool = False,
    target_flops: float = 1.0e16,
    mode: str = "interleaved_sequential",
    no_resume: bool = False,
    skip_eval: bool = False,
    num_eval_samples: int = 0,
    no_flop_constraint: bool = False,
    config: str = "",
    stage_transition_criterion: str = "",
    termination_criterion: str = "",
    sft_epochs: float = 0.0,
    sft_steps: int = 0,
    sft_patience: int = 0,
    grpo_epochs: float = 0.0,
    grpo_steps: int = 0,
    grpo_patience: int = 0,
    patience: int = 0,
    run_tag: str = "",
) -> None:
    """CLI entrypoint for Modal deployment."""
    active_gpu = (gpu or TARGET_GPU).upper()
    resume = not no_resume
    budget_label = "Unconstrained (No FLOP limit)" if (no_flop_constraint or target_flops <= 0) else f"{target_flops:.2e}"
    resolved_tag = run_tag.strip().lstrip("_") if run_tag else (
        "unconstrained" if (no_flop_constraint or target_flops <= 0) else "baseline"
    )
    print(f"==================================================================")
    print(f"MULTIGEN-RL: EXP-1.3 Modal Remote Deployment")
    print(f"GPU Target:   {active_gpu}")
    print(f"Hybrid Mode:  {mode}")
    print(f"Run Tag:      {resolved_tag}")
    print(f"Target FLOPs: {budget_label}")
    print(f"Resume:       {resume}")
    print(f"==================================================================")

    if run_all_seeds:
        seeds = [42, 43, 44]
        print(f"Dispatching 3 parallel container jobs for seeds: {seeds} (Tag: {resolved_tag}, Budget: {budget_label})...")
        results = list(
            run_parallel_seed.starmap(
                [
                    (
                        s,
                        target_flops,
                        mode,
                        resume,
                        skip_eval,
                        num_eval_samples,
                        no_flop_constraint,
                        config,
                        stage_transition_criterion,
                        termination_criterion,
                        sft_epochs,
                        sft_steps,
                        sft_patience,
                        grpo_epochs,
                        grpo_steps,
                        grpo_patience,
                        patience,
                        resolved_tag,
                    )
                    for s in seeds
                ]
            )
        )
        for r in results:
            print(f"[Result] {r}")
        print("All 3 seeds completed successfully!")
    else:
        print(f"Executing single run for seed {seed} (Tag: {resolved_tag}, Budget: {budget_label})...")
        result = run_single_seed.remote(
            seed=seed,
            target_flops=target_flops,
            mode=mode,
            resume=resume,
            skip_eval=skip_eval,
            num_eval_samples=num_eval_samples,
            no_flop_constraint=no_flop_constraint,
            config=config,
            stage_transition_criterion=stage_transition_criterion,
            termination_criterion=termination_criterion,
            sft_epochs=sft_epochs,
            sft_steps=sft_steps,
            sft_patience=sft_patience,
            grpo_epochs=grpo_epochs,
            grpo_steps=grpo_steps,
            grpo_patience=grpo_patience,
            patience=patience,
            run_tag=resolved_tag,
        )
        print(f"[Result] {result}")
