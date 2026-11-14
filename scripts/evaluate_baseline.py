#!/usr/bin/env python3
"""CLI script to run baseline zero-shot / few-shot evaluation on GSM8K (EXP-0.1).

This script establishes the foundational baseline accuracy, format compliance,
and initial error index set (E_base) for the frozen base model (e.g. Qwen/Qwen3-0.6B)
prior to any RL training (EXP-1.1) or Knowledge Distillation (EXP-1.2).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
from pathlib import Path

# Ensure multiprocessing uses 'spawn' and vLLM uses spawn before any CUDA operations
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ==============================================================================
# Path Setup: Ensure project root is in sys.path for direct script invocation
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multigen.eval.evaluator import GSM8KEvaluator, split_gsm8k_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("evaluate_baseline")


def parse_args() -> argparse.Namespace:
    """Parse command line flags for the baseline evaluation experiment."""
    parser = argparse.ArgumentParser(
        description="Run GSM8K baseline zero-shot/few-shot evaluation (EXP-0.1)"
    )
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model ID or local directory (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional PEFT LoRA adapter checkpoint path (used in Phase 1+)",
    )
    # Dataset & split configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Dataset name (default: gsm8k)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "test"],
        default="test",
        help="Split to evaluate: 'dev' (500 problems) or 'test' (1,319 problems)",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=0,
        help="Number of in-context few-shot exemplars (0 for zero-shot, 4 for 4-shot)",
    )
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per solution (default: 512)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers"],
        default="vllm",
        help="Inference engine backend ('vllm' for GPU, 'transformers' for CPU/MPS)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for HuggingFace transformers backend (default: 16)",
    )
    # Diagnostics & budget control
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N examples (useful for rapid dry-runs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/dev split partitioning (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evals/exp0_1_baseline.json",
        help="Path to write evaluation results JSON (default: evals/exp0_1_baseline.json)",
    )
    return parser.parse_args()


def load_dataset_split(split_name: str, seed: int = 42):
    """Load GSM8K from HuggingFace datasets and apply deterministic train/dev splitting.

    Args:
        split_name: 'dev' or 'test'.
        seed: Random seed for train/dev partition.

    Returns:
        List of problem dictionaries for the requested split.
    """
    from datasets import load_dataset

    logger.info("Loading 'openai/gsm8k' (name='main') from Hugging Face datasets...")
    raw_ds = load_dataset("openai/gsm8k", "main")
    # Carve dev (500) from train (7,473 -> 6,973 train + 500 dev), keeping test (1,319) untouched
    splits = split_gsm8k_dataset(raw_ds, dev_size=500, seed=seed)
    return splits[split_name]


def main() -> None:
    """Execute baseline evaluation pipeline, report metrics, and persist JSON artifacts."""
    args = parse_args()

    # Step 1: Load and partition the benchmark dataset
    eval_data = load_dataset_split(args.split, seed=args.seed)
    if args.limit is not None and args.limit > 0:
        logger.info("Limiting evaluation to first %d examples", args.limit)
        eval_data = eval_data[: args.limit]

    logger.info(
        "Initializing evaluator for model '%s' (backend: %s, split: %s, total: %d, num_shots: %d)",
        args.model,
        args.backend,
        args.split,
        len(eval_data),
        args.num_shots,
    )

    # Step 2: Initialize inference engine (vLLM on CUDA or Transformers on CPU/MPS)
    evaluator = GSM8KEvaluator(
        model_name_or_path=args.model,
        adapter_path=args.adapter_path,
        backend=args.backend,
        max_tokens=args.max_tokens,
        temperature=0.0,  # Strict greedy decoding for deterministic baseline
        batch_size=args.batch_size,
    )

    # Step 3: Run greedy inference and score completions
    result = evaluator.evaluate(
        dataset=eval_data,
        split_name=args.split,
        num_shots=args.num_shots,
        record_details=True,
    )

    # Step 4: Display clean summary metrics table in terminal
    print("\n" + "=" * 65)
    print(f" EXP-0.1 Baseline Evaluation Summary: {args.model}")
    print("=" * 65)
    print(f" Split:                   {result.split} (N={result.total_examples})")
    print(f" Few-shot shots:          {result.num_shots}")
    print(f" Strict Accuracy:         {result.accuracy * 100:.2f}%")
    print(f" Permissive Accuracy:     {result.permissive_accuracy * 100:.2f}%")
    print(f" Format Compliance:       {result.format_compliance_rate * 100:.2f}%")
    print(f" Mean Completion Tokens:  {result.mean_completion_tokens:.1f}")
    print(f" Wall Clock Time:         {result.wall_time_seconds:.2f}s")
    print(f" Parser Exceptions:       {result.parser_exceptions}")
    print(f" Total Error Set Size:    {len(result.error_indices)}")
    print("=" * 65)

    # Step 5: Save full results artifact to output JSON file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.to_json(indent=2))
    logger.info("Evaluation results saved to %s", out_path)


if __name__ == "__main__":
    main()
