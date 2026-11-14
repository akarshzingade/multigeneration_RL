#!/usr/bin/env python3
"""CLI utility to download and organize experimental artifacts from Modal volume storage.

Downloads evaluation benchmarks, population DAG lineage metadata, trained LoRA checkpoints,
and execution logs into a structured local directory:
    experiment_results/<experiment_name>_<target_flops>/
        ├── evals/
        ├── checkpoints/
        │   ├── final_adapter/
        │   └── intermediate_checkpoints/ (optional)
        ├── logs/
        └── population_dag.json

Usage:
    # Download 1.0e16 production run for seed 42
    python scripts/download_experiment_results.py --experiment-id exp1_1_grpo_baseline --target-flops 1e16 --seed 42

    # Download including intermediate milestone checkpoints
    python scripts/download_experiment_results.py --seed 42 --include-intermediates
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and organize experiment artifacts from Modal storage volume."
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="exp1_1_grpo_baseline",
        help="Experiment identifier (default: exp1_1_grpo_baseline)",
    )
    parser.add_argument(
        "--target-flops",
        type=str,
        default="1e16",
        help="Target FLOP budget string for folder naming (default: 1e16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed evaluated (default: 42)",
    )
    parser.add_argument(
        "--volume-name",
        type=str,
        default="multigen-storage",
        help="Modal volume name (default: multigen-storage)",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="experiment_results",
        help="Base directory for experiment results (default: experiment_results)",
    )
    parser.add_argument(
        "--include-intermediates",
        action="store_true",
        help="Also download intermediate milestone checkpoints (can be large ~200MB)",
    )
    parser.add_argument(
        "--skip-adapter",
        action="store_true",
        help="Skip downloading the LoRA adapter weights (downloads only metrics & DAG)",
    )
    return parser.parse_args()


def find_modal_bin() -> str:
    """Find the path to the 'modal' executable."""
    # 1. Check current Python env bin dir
    env_bin = os.path.join(os.path.dirname(sys.executable), "modal")
    if os.path.exists(env_bin):
        return env_bin
    # 2. Check PATH
    which_modal = shutil.which("modal")
    if which_modal:
        return which_modal
    return "modal"


def run_modal_get(volume_name: str, remote_path: str, local_path: str) -> bool:
    """Execute 'modal volume get' via subprocess."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    modal_bin = find_modal_bin()
    cmd = [modal_bin, "volume", "get", volume_name, remote_path, local_path]
    logger.info("Executing: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.warning("Failed to download %s: %s", remote_path, e.stderr.strip() or e.stdout.strip())
        return False
    except FileNotFoundError:
        logger.error("'modal' CLI binary not found in PATH. Please activate your conda environment.")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    # 1. Resolve output paths
    folder_name = f"{args.experiment_id}_{args.target_flops}"
    dest_dir = Path(args.output_base_dir) / folder_name
    evals_dir = dest_dir / "evals"
    checkpoints_dir = dest_dir / "checkpoints"
    logs_dir = dest_dir / "logs"

    evals_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DOWNLOADING EXPERIMENT RESULTS: %s (Seed %d)", args.experiment_id, args.seed)
    logger.info("Destination directory: %s", dest_dir.resolve())
    logger.info("=" * 70)

    # 2. Download population_dag.json
    dag_dest = str(dest_dir / "population_dag.json")
    logger.info("Downloading population lineage DAG...")
    run_modal_get(args.volume_name, "population_dag.json", dag_dest)

    # 3. Download evals JSON artifact
    eval_remote = f"evals/exp1_1_grpo_seed{args.seed}_test.json"
    eval_local = str(evals_dir / f"exp1_1_grpo_seed{args.seed}_test.json")
    logger.info("Downloading evaluation report for seed %d...", args.seed)
    run_modal_get(args.volume_name, eval_remote, eval_local)

    # 4. Download final LoRA adapter
    if not args.skip_adapter:
        adapter_remote = f"checkpoints/exp1_1_grpo_seed{args.seed}/final_adapter"
        adapter_dest_parent = str(checkpoints_dir) + "/"
        logger.info("Downloading final LoRA adapter weights...")
        run_modal_get(args.volume_name, adapter_remote, adapter_dest_parent)

    # 5. Optionally download intermediate checkpoints
    if args.include_intermediates:
        for idx in range(1, 6):
            cp_remote = f"checkpoints/exp1_1_grpo_seed{args.seed}/intermediate_checkpoint_{idx}"
            cp_dest_parent = str(checkpoints_dir) + "/"
            logger.info("Downloading milestone checkpoint %d...", idx)
            run_modal_get(args.volume_name, cp_remote, cp_dest_parent)

    # 6. Copy local run log if present
    possible_logs = [
        Path(f"cmd_logs/expt1.1_prod_seed{args.seed}.txt"),
        Path(f"cmd_logs/expt1.1_testModal_{args.seed}.txt"),
    ]
    for log_path in possible_logs:
        if log_path.exists():
            shutil.copy2(log_path, logs_dir / log_path.name)
            logger.info("Copied execution log: %s -> %s", log_path, logs_dir / log_path.name)

    # 7. Print Quick Summary Analysis of Downloaded Artifacts
    logger.info("=" * 70)
    logger.info("DOWNLOAD COMPLETE. SUMMARY ANALYSIS:")
    logger.info("=" * 70)

    # Inspect evaluation report if downloaded
    if Path(eval_local).exists():
        try:
            with open(eval_local, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            total = eval_data.get("total_examples", 0)
            strict_acc = eval_data.get("accuracy", 0.0) * 100
            perm_acc = eval_data.get("permissive_accuracy", 0.0) * 100
            fmt_rate = eval_data.get("format_compliance_rate", 0.0) * 100
            mean_tokens = eval_data.get("mean_completion_tokens", 0.0)
            wall_time = eval_data.get("wall_time_seconds", 0.0)

            print(f"\nBenchmark Evaluation (GSM8K Test Split, {total} problems):")
            print(f"  * Strict Accuracy:           {strict_acc:.2f}%")
            print(f"  * Permissive Accuracy:       {perm_acc:.2f}%")
            print(f"  * Format Compliance:        {fmt_rate:.2f}%")
            print(f"  * Mean Completion Length:   {mean_tokens:.1f} tokens")
            print(f"  * Test Inference Wall Time: {wall_time:.1f} s")
        except Exception as e:
            logger.warning("Could not parse evaluation file: %s", e)

    # Inspect DAG if downloaded
    if Path(dag_dest).exists():
        try:
            with open(dag_dest, "r", encoding="utf-8") as f:
                dag_data = json.load(f)
            nodes = dag_data.get("nodes", [])
            print(f"\nPopulation Lineage Graph ({len(nodes)} registered nodes):")
            for node in nodes:
                nid = node.get("node_id")
                flops = node.get("cumulative_flops", 0.0)
                cap = node.get("capability_profile") or {}
                acc = cap.get("accuracy")
                acc_str = f"{acc * 100:.2f}%" if acc is not None else "N/A"
                print(f"  * Node: {nid:25} | Cumulative FLOPs: {flops:.2e} | Test Acc: {acc_str}")
        except Exception as e:
            logger.warning("Could not parse DAG file: %s", e)

    print(f"\nAll artifacts saved to: {dest_dir.resolve()}\n")


if __name__ == "__main__":
    main()
