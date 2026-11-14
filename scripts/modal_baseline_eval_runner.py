"""Modal deployment script for running baseline model evaluation (EXP-0.1) on Nvidia GPUs.

Evaluates any frozen base model (e.g. Qwen/Qwen3-0.6B, Qwen/Qwen2.5-Math-1.5B, etc.)
zero-shot and/or few-shot on GSM8K using high-throughput vLLM inference and mathematical verification.
Results and error sets are committed to the persistent Modal volume 'multigen-storage'.

Folder & Artifact Naming Structure:
  Primary Destination:
    /root/storage/experiment_results/exp0_1_baseline_{model}/evals/exp0_1_baseline_{model}_{split}_{num_shots}shot.json
  Flat Evals Mirror (for easy bulk-sync):
    /root/storage/evals/exp0_1_baseline_{model}_{split}_{num_shots}shot.json

Usage:
  # 1. Default model (Qwen/Qwen3-0.6B) quick test on 10 examples:
  modal run scripts/modal_baseline_eval_runner.py --limit 10

  # 2. Run a specific model (e.g. Qwen/Qwen2.5-Math-1.5B) zero-shot:
  modal run scripts/modal_baseline_eval_runner.py --model Qwen/Qwen2.5-Math-1.5B --num-shots 0

  # 3. Run a specific model with --model-name alias:
  modal run scripts/modal_baseline_eval_runner.py --model-name Qwen/Qwen2.5-Math-1.5B --num-shots 4

  # 4. Run both 0-shot and 4-shot in parallel for any model:
  modal run scripts/modal_baseline_eval_runner.py --model Qwen/Qwen2.5-Math-1.5B --run-both-shots

  # 5. Download results to your local repository:
  modal volume get multigen-storage experiment_results/ ./experiment_results/
  # or flat evals:
  modal volume get multigen-storage evals/ ./evals/
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional
import modal

app = modal.App("multigen-exp0-1-baseline-eval")


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
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "trl>=0.8.0",
        "peft>=0.10.0",
        "vllm>=0.4.0",
        "math-verify>=0.1.0",
        "datasets>=2.18.0",
        "networkx>=3.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
    )
    .add_local_dir("multigen", remote_path="/root/multigen-rl/multigen")
    .add_local_dir("scripts", remote_path="/root/multigen-rl/scripts")
    .add_local_dir("configs", remote_path="/root/multigen-rl/configs")
    .add_local_file("pyproject.toml", remote_path="/root/multigen-rl/pyproject.toml")
)

volume = modal.Volume.from_name("multigen-storage", create_if_missing=True)


def _execute_baseline_eval(
    model: str,
    split: str,
    num_shots: int,
    backend: str,
    limit: Optional[int],
    gpu_tag: str,
) -> Dict[str, Any]:
    """Execute evaluation in the remote container and persist results to volume."""
    os.chdir("/root/multigen-rl")
    os.environ["PYTHONPATH"] = "/root/multigen-rl"
    os.environ["HF_HOME"] = "/root/storage/hf_cache"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Sanitize model name for filesystem safety (e.g. 'Qwen/Qwen3-0.6B' -> 'Qwen_Qwen3-0.6B')
    model_clean = model.replace("/", "_")
    filename = f"exp0_1_baseline_{model_clean}_{split}_{num_shots}shot.json"

    # Structured folder: /root/storage/experiment_results/exp0_1_baseline_<model>/evals/
    exp_folder = f"exp0_1_baseline_{model_clean}"
    structured_dir = f"/root/storage/experiment_results/{exp_folder}/evals"
    os.makedirs(structured_dir, exist_ok=True)
    out_path = os.path.join(structured_dir, filename)

    # Flat evals mirror: /root/storage/evals/
    flat_dir = "/root/storage/evals"
    os.makedirs(flat_dir, exist_ok=True)
    flat_out_path = os.path.join(flat_dir, filename)

    cmd = [
        "python",
        "scripts/evaluate_baseline.py",
        "--model",
        model,
        "--split",
        split,
        "--num-shots",
        str(num_shots),
        "--backend",
        backend,
        "--output",
        out_path,
    ]
    if limit is not None and limit > 0:
        cmd.extend(["--limit", str(limit)])

    print(f"[{gpu_tag}] Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Mirror JSON to flat evals directory as well for easy access
    if os.path.exists(out_path):
        shutil.copy2(out_path, flat_out_path)

    volume.commit()

    # Load summary metrics to return back to caller
    summary: Dict[str, Any] = {
        "output_file": filename,
        "experiment_folder": exp_folder,
        "remote_path": f"experiment_results/{exp_folder}/evals/{filename}",
        "gpu": gpu_tag,
    }
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            summary.update({
                "model_name": data.get("model_name"),
                "split": data.get("split"),
                "num_shots": data.get("num_shots"),
                "total_examples": data.get("total_examples"),
                "accuracy": data.get("accuracy"),
                "permissive_accuracy": data.get("permissive_accuracy"),
                "format_compliance_rate": data.get("format_compliance_rate"),
                "mean_completion_tokens": data.get("mean_completion_tokens"),
                "wall_time_seconds": data.get("wall_time_seconds"),
                "error_count": len(data.get("error_indices", [])),
            })
        except Exception as e:
            summary["read_error"] = str(e)

    return summary


@app.function(
    gpu=TARGET_GPU,
    image=image,
    volumes={"/root/storage": volume},
    timeout=1800,  # 30 minutes
)
def run_eval(
    model: str = "Qwen/Qwen3-0.6B",
    split: str = "test",
    num_shots: int = 0,
    backend: str = "vllm",
    limit: int = 0,
    gpu_tag: str = "",
) -> Dict[str, Any]:
    """Modal remote function evaluating the model on a GPU."""
    active_gpu = gpu_tag or TARGET_GPU
    effective_limit = limit if limit > 0 else None

    return _execute_baseline_eval(
        model=model,
        split=split,
        num_shots=num_shots,
        backend=backend,
        limit=effective_limit,
        gpu_tag=active_gpu,
    )


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-0.6B",
    model_name: str = "",
    split: str = "test",
    num_shots: int = 0,
    run_both_shots: bool = False,
    limit: int = 0,
    backend: str = "vllm",
    gpu: str = "",
    gpu_type: str = "",
):
    """Local CLI entrypoint for running EXP-0.1 baseline evaluation on Modal."""
    target_model = model_name or model
    active_gpu = (gpu or gpu_type or TARGET_GPU).upper()
    print(f"Target Model: {target_model}")
    print(f"Target GPU configured: {active_gpu}")

    if run_both_shots:
        print(f"Launching EXP-0.1 in parallel on Modal (0-shot and 4-shot on {active_gpu})...")
        configs = [
            (target_model, split, 0, backend, limit, active_gpu),
            (target_model, split, 4, backend, limit, active_gpu),
        ]
        results = list(run_eval.starmap(configs))
        print("\n" + "=" * 70)
        print(f" EXP-0.1 Baseline Evaluations Finished for: {target_model}")
        print("=" * 70)
        for res in results:
            print(f"\nConfiguration: {res.get('num_shots')}-shot on split '{res.get('split')}'")
            print(f"  Strict Accuracy:        {res.get('accuracy', 0.0) * 100:.2f}%")
            print(f"  Permissive Accuracy:    {res.get('permissive_accuracy', 0.0) * 100:.2f}%")
            print(f"  Format Compliance:      {res.get('format_compliance_rate', 0.0) * 100:.2f}%")
            print(f"  Mean Tokens:            {res.get('mean_completion_tokens', 0.0):.1f}")
            print(f"  Total Errors:           {res.get('error_count', 0)}")
            print(f"  Artifact Path:          {res.get('remote_path')}")
        print("=" * 70)
        print("\nDownload results to local repository:")
        print(f"  modal volume get multigen-storage experiment_results/{results[0].get('experiment_folder')} ./experiment_results/")
        print("  # or flat sync:")
        print("  modal volume get multigen-storage evals/ ./evals/")
    else:
        print(
            f"Launching EXP-0.1 on Modal (Model: {target_model}, Shots: {num_shots}, Split: {split}, GPU: {active_gpu})..."
        )
        res = run_eval.remote(
            model=target_model,
            split=split,
            num_shots=num_shots,
            backend=backend,
            limit=limit,
            gpu_tag=active_gpu,
        )
        print("\n" + "=" * 70)
        print(f" EXP-0.1 Baseline Evaluation Finished: {res.get('model_name')}")
        print("=" * 70)
        print(f" Split:                   {res.get('split')} (N={res.get('total_examples')})")
        print(f" Few-shot exemplars:      {res.get('num_shots')}-shot")
        print(f" Strict Accuracy:         {res.get('accuracy', 0.0) * 100:.2f}%")
        print(f" Permissive Accuracy:     {res.get('permissive_accuracy', 0.0) * 100:.2f}%")
        print(f" Format Compliance:       {res.get('format_compliance_rate', 0.0) * 100:.2f}%")
        print(f" Mean Completion Tokens:  {res.get('mean_completion_tokens', 0.0):.1f}")
        print(f" Wall Clock Time:         {res.get('wall_time_seconds', 0.0):.2f}s")
        print(f" Error Set Size:          {res.get('error_count', 0)}")
        print(f" Saved Artifact:          {res.get('remote_path')}")
        print("=" * 70)
        print("\nDownload result to local repository:")
        print(f"  modal volume get multigen-storage {res.get('remote_path')} ./{res.get('remote_path')}")
        print("  # or sync flat:")
        print(f"  modal volume get multigen-storage evals/{res.get('output_file')} ./evals/{res.get('output_file')}")
