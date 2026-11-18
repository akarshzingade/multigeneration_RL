#!/usr/bin/env python3
"""CLI driver for Experiment 1.3: Mixed Learning (RL + Distillation) vs Baselines (Hypothesis H1).

Executes hybrid training (SFT Warmup -> GRPO Exploration) on Qwen/Qwen3-0.6B with PEFT LoRA
under strict compute budgeting (1.0e16 FLOPs). Supports seamless fault tolerance and crash
resumption (--resume), evaluates on GSM8K test split, and registers the hybrid node in the
Population Lineage DAG with dual-parent ancestry.

Usage:
  # Standard run with auto-resumption:
  python scripts/run_mixed_experiment.py --config configs/experiments/exp1_3_mixed_learning.yaml --seed 42 --resume

  # Fast debug run with small budget:
  python scripts/run_mixed_experiment.py --target-flops 1e14 --num-eval-samples 20
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import logging
import multiprocessing
import os
import sys

# Configure vLLM multiprocessing start method and disable experimental V1 engine
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "0"
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from pathlib import Path
from typing import Any, Dict, List, Optional
import datasets
import torch

from multigen.core.population_dag import PopulationDAG
from multigen.core.schema import CapabilityProfile, ModelNode, TrainingRecipe
from multigen.eval.evaluator import GSM8KEvaluator, format_gsm8k_prompt, split_gsm8k_dataset
from multigen.training.mixed_runner import MixedRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_mixed_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EXP-1.3 Mixed Learning (RL + Distillation) Benchmark (Hypothesis H1)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp1_3_mixed_learning.yaml",
        help="Path to YAML experiment configuration file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override for model initialization and data shuffling",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory override for checkpoints",
    )
    parser.add_argument(
        "--dag-path",
        type=str,
        default="population_dag.json",
        help="Path to population lineage DAG JSON file",
    )
    parser.add_argument(
        "--target-flops",
        type=float,
        default=None,
        help="Optional override for target FLOP compute budget",
    )
    parser.add_argument(
        "--no-flop-constraint",
        "--no-flop-budget",
        action="store_true",
        help="Train without FLOP compute budget ceiling (stops via plateau/epochs/steps)",
    )
    parser.add_argument(
        "--termination-criterion",
        type=str,
        choices=["plateau", "epochs", "steps", "flops"],
        default=None,
        help="Overall/Stage 2 termination criterion override",
    )
    parser.add_argument(
        "--stage-transition-criterion",
        type=str,
        choices=["plateau", "epochs", "steps", "flops"],
        default=None,
        help="Stage 1 SFT -> Stage 2 GRPO transition criterion override",
    )
    parser.add_argument(
        "--sft-epochs",
        type=float,
        default=None,
        help="Stage 1 SFT epochs override",
    )
    parser.add_argument(
        "--sft-steps",
        "--sft-max-steps",
        type=int,
        default=None,
        help="Stage 1 SFT max steps override",
    )
    parser.add_argument(
        "--sft-patience",
        type=int,
        default=None,
        help="Stage 1 SFT early stopping patience",
    )
    parser.add_argument(
        "--grpo-epochs",
        type=float,
        default=None,
        help="Stage 2 GRPO epochs override",
    )
    parser.add_argument(
        "--grpo-steps",
        "--grpo-max-steps",
        type=int,
        default=None,
        help="Stage 2 GRPO max steps override",
    )
    parser.add_argument(
        "--grpo-patience",
        type=int,
        default=None,
        help="Stage 2 GRPO early stopping patience",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Default early stopping patience for both SFT and GRPO stages",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Explicit run tag for checkpoints, eval JSONs, and DAG node IDs (auto-generated if omitted)",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        choices=["dev", "test"],
        default="test",
        help="Benchmark split to evaluate post-training (default: test)",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default="evals",
        help="Directory to save evaluation result JSON",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=None,
        help="Optional maximum number of evaluation samples (useful for quick debug)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume training from previous checkpoint/state journal if available (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Disable resumption and restart experiment from scratch",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip post-training evaluation and DAG registration",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interleaved_sequential", "teacher_regularized", "micro_interleaved"],
        default=None,
        help="Override hybrid learning mode",
    )
    parser.add_argument(
        "--traces-path",
        type=str,
        default=None,
        help="Optional path to pre-generated verified teacher traces JSONL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        logger.error("Configuration file not found: %s", args.config)
        sys.exit(1)

    recipe = TrainingRecipe.from_yaml(args.config)

    # Apply CLI overrides
    if args.seed is not None:
        recipe.seed = args.seed
    if args.no_flop_constraint:
        recipe.target_flop_budget = None
    elif args.target_flops is not None:
        recipe.target_flop_budget = args.target_flops

    if args.termination_criterion is not None:
        recipe.termination_criterion = args.termination_criterion
    elif args.no_flop_constraint and recipe.termination_criterion == "flops":
        recipe.termination_criterion = "plateau"

    if args.stage_transition_criterion is not None:
        recipe.stage_transition_criterion = args.stage_transition_criterion
    elif args.no_flop_constraint and recipe.stage_transition_criterion == "flops":
        recipe.stage_transition_criterion = "plateau"

    if args.patience is not None:
        recipe.sft_patience = args.patience
        recipe.grpo_patience = args.patience
        recipe.early_stopping_patience = args.patience
    if args.sft_patience is not None:
        recipe.sft_patience = args.sft_patience
    if args.grpo_patience is not None:
        recipe.grpo_patience = args.grpo_patience

    if args.sft_epochs is not None:
        recipe.sft_epochs = args.sft_epochs
    if args.sft_steps is not None:
        recipe.sft_max_steps = args.sft_steps
    if args.grpo_epochs is not None:
        recipe.grpo_epochs = args.grpo_epochs
    if args.grpo_steps is not None:
        recipe.grpo_max_steps = args.grpo_steps

    if args.mode is not None:
        recipe.mode = args.mode
    if args.traces_path is not None:
        recipe.teacher_trace_dataset_path = args.traces_path
    elif not recipe.teacher_trace_dataset_path or not os.path.exists(recipe.teacher_trace_dataset_path):
        if os.path.exists("/root/storage/data/teacher_traces_gsm8k.jsonl"):
            recipe.teacher_trace_dataset_path = "/root/storage/data/teacher_traces_gsm8k.jsonl"

    seed = recipe.seed
    raw_tag = (
        args.run_tag.strip().lstrip("_")
        if args.run_tag
        else ("unconstrained" if recipe.target_flop_budget is None else "baseline")
    )
    run_tag = f"_{raw_tag}"
    output_dir = args.output_dir or f"checkpoints/exp1_3_mixed{run_tag}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.eval_output_dir, exist_ok=True)

    budget_disp = (
        f"{recipe.target_flop_budget:.2e} FLOPs"
        if recipe.target_flop_budget
        else "None (unconstrained)"
    )

    logger.info("==================================================================")
    logger.info("MULTIGEN-RL: Experiment 1.3 Mixed Learning (RL + Distillation)")
    logger.info("Base Model:    %s", recipe.base_model_name_or_path)
    logger.info("Teacher Model: %s", recipe.teacher_model_name_or_path)
    logger.info("Teacher Traces: %s", recipe.teacher_trace_dataset_path)
    logger.info("Hybrid Mode:   %s", recipe.mode)
    logger.info("Budget Cap:    %s", budget_disp)
    logger.info("Transition:    %s", recipe.stage_transition_criterion)
    logger.info("Termination:   %s", recipe.termination_criterion)
    logger.info("Seed:          %d", seed)
    logger.info("Resume:        %s", args.resume)
    logger.info("Output Dir:    %s", output_dir)
    logger.info("==================================================================")

    # Load GSM8K splits
    logger.info("Loading GSM8K dataset: %s (%s)...", recipe.dataset_name, recipe.dataset_subset)
    raw_ds = datasets.load_dataset(recipe.dataset_name, recipe.dataset_subset)
    splits = split_gsm8k_dataset(raw_ds, dev_size=500, seed=42)

    train_data = splits["train"]
    dev_data = splits["dev"]
    test_data = splits["test"]

    logger.info(
        "Dataset partitioned: train=%d, dev=%d, official_test=%d",
        len(train_data),
        len(dev_data),
        len(test_data),
    )

    questions = [ex["question"] for ex in train_data]
    gold_answers = [ex["gold_answer"] for ex in train_data]

    # Format HF Datasets for GRPO / evaluation
    from datasets import Dataset

    train_prompts = [
        format_gsm8k_prompt(item["question"], num_shots=recipe.num_shots)
        for item in train_data
    ]
    train_answers = [item["gold_answer"] for item in train_data]

    dev_prompts = [
        format_gsm8k_prompt(item["question"], num_shots=recipe.num_shots)
        for item in dev_data
    ]
    dev_answers = [item["gold_answer"] for item in dev_data]

    hf_train = Dataset.from_dict({"prompt": train_prompts, "answer": train_answers})
    hf_dev = Dataset.from_dict({"prompt": dev_prompts, "answer": dev_answers})

    # Instantiate MixedRunner
    runner = MixedRunner(
        recipe=recipe,
        output_dir=output_dir,
        resume=args.resume,
    )

    # Execute Mixed Learning
    training_results = runner.run(
        questions=questions,
        gold_answers=gold_answers,
        train_dataset=hf_train,
        eval_dataset=hf_dev,
    )

    final_adapter_path = training_results.get("final_adapter_path")
    logger.info("Mixed training concluded. Final adapter path: %s", final_adapter_path)
    logger.info("Cumulative Compute: %.2e FLOPs", runner.accumulator.cumulative_flops)

    if args.skip_eval:
        logger.info("--skip-eval flag provided. Exiting without post-training evaluation.")
        return

    # Post-training evaluation
    eval_target_dataset = dev_data if args.eval_split == "dev" else test_data
    eval_questions = [ex["question"] for ex in eval_target_dataset]
    eval_gold_answers = [ex["gold_answer"] for ex in eval_target_dataset]

    if args.num_eval_samples is not None:
        eval_questions = eval_questions[: args.num_eval_samples]
        eval_gold_answers = eval_gold_answers[: args.num_eval_samples]
        logger.info("Truncated evaluation to %d samples as requested.", len(eval_questions))

    logger.info("Evaluating final checkpoint on GSM8K %s split (%d samples)...", args.eval_split, len(eval_questions))
    evaluator = GSM8KEvaluator(
        model_name_or_path=recipe.base_model_name_or_path,
        adapter_path=final_adapter_path,
    )
    profile = evaluator.evaluate(
        questions=eval_questions,
        gold_answers=eval_gold_answers,
        num_shots=recipe.num_shots,
    )

    logger.info("==================================================================")
    logger.info("EVALUATION RESULTS (EXP-1.3 Mixed Seed %d):", seed)
    logger.info("Strict Verifier Accuracy:  %.2f%%", profile.accuracy * 100)
    logger.info("Permissive Accuracy:       %.2f%%", profile.permissive_accuracy * 100)
    logger.info("XML Format Compliance:     %.2f%%", profile.format_compliance_rate * 100)
    logger.info("Mean Completion Tokens:    %.1f", profile.mean_completion_tokens)
    logger.info("Wall-clock Seconds:        %.1f", profile.wall_time_seconds)
    logger.info("==================================================================")

    # Save evaluation scorecard
    eval_filename = f"exp1_3_mixed{run_tag}_seed{seed}_{args.eval_split}.json"
    eval_path = os.path.join(args.eval_output_dir, eval_filename)
    eval_dict = profile.to_dict() if hasattr(profile, "to_dict") else asdict(profile)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=2, default=str)
    logger.info("Saved evaluation profile to %s", eval_path)

    # Convert to CapabilityProfile for Lineage DAG registration
    cap_profile = CapabilityProfile(
        benchmark="gsm8k",
        split=args.eval_split if args.eval_split in ("train", "dev", "test") else "test",
        num_shots=recipe.num_shots,
        accuracy=profile.accuracy,
        permissive_accuracy=profile.permissive_accuracy,
        format_compliance_rate=profile.format_compliance_rate,
        mean_completion_tokens=profile.mean_completion_tokens,
        error_indices=profile.error_indices,
        wall_time_seconds=profile.wall_time_seconds,
        metadata={"total_examples": profile.total_examples, "parser_exceptions": profile.parser_exceptions},
    )

    # Register in Population Lineage DAG
    logger.info("Registering hybrid model in Population Lineage DAG: %s", args.dag_path)
    node_id = f"exp1_3_mixed{run_tag}_seed{seed}"
    runner.register_in_population_dag(
        dag_path=args.dag_path,
        adapter_path=final_adapter_path,
        capability_profile=cap_profile,
        node_id=node_id,
    )

    logger.info("EXP-1.3 Mixed Learning benchmark successfully completed for seed %d!", seed)


if __name__ == "__main__":
    main()
