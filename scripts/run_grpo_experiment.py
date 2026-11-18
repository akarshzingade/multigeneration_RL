#!/usr/bin/env python3
"""CLI driver for Experiment 1.1: Standalone Reinforcement Learning (GRPO) Baseline.

Executes GRPO training on Qwen/Qwen3-0.6B with PEFT LoRA under strict compute budgeting
(1.0e16 FLOPs). Evaluates the trained adapter on held-out GSM8K and registers the new
child checkpoint node into the population lineage DAG.

Usage:
  python scripts/run_grpo_experiment.py --config configs/experiments/exp1_1_grpo_baseline.yaml --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import datasets
from datasets import Dataset

from multigen.core.population_dag import PopulationDAG
from multigen.core.schema import CapabilityProfile, ModelNode, TrainingRecipe
from multigen.eval.evaluator import GSM8KEvaluator, format_gsm8k_prompt, split_gsm8k_dataset
from multigen.training.grpo_runner import GRPORunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_grpo_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EXP-1.1 Standalone GRPO Reinforcement Learning Baseline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp1_1_grpo_baseline.yaml",
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
        help="Stopping criterion override (default: plateau if unconstrained, else flops)",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Number of training epochs override",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps override",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (eval rounds without improvement before halt)",
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
        "--skip-eval",
        action="store_true",
        help="Skip post-training evaluation on held-out benchmark split",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default="evals",
        help="Directory to save evaluation JSON artifacts",
    )
    return parser.parse_args()


def prepare_datasets(
    recipe: TrainingRecipe,
) -> Tuple[Dict[str, Dataset], Dict[str, List[Dict[str, str]]]]:
    """Load and format GSM8K dataset into prompt and answer columns."""
    logger.info("Loading dataset: %s (%s)", recipe.dataset_name, recipe.dataset_subset)
    raw_ds = datasets.load_dataset(recipe.dataset_name, recipe.dataset_subset)
    splits = split_gsm8k_dataset(raw_ds, dev_size=500, seed=recipe.seed)

    # Format training dataset
    train_prompts = [
        format_gsm8k_prompt(item["question"], num_shots=recipe.num_shots)
        for item in splits["train"]
    ]
    train_answers = [item["gold_answer"] for item in splits["train"]]

    # Format dev dataset for evaluation
    dev_prompts = [
        format_gsm8k_prompt(item["question"], num_shots=recipe.num_shots)
        for item in splits["dev"]
    ]
    dev_answers = [item["gold_answer"] for item in splits["dev"]]

    hf_train = Dataset.from_dict({"prompt": train_prompts, "answer": train_answers})
    hf_dev = Dataset.from_dict({"prompt": dev_prompts, "answer": dev_answers})

    logger.info(
        "Prepared datasets: train=%d samples, dev=%d samples, test=%d samples",
        len(hf_train),
        len(hf_dev),
        len(splits.get("test", [])),
    )
    return {"train": hf_train, "dev": hf_dev}, splits


def main() -> None:
    args = parse_args()

    # 1. Load Recipe and apply CLI overrides
    if not os.path.exists(args.config):
        logger.error("Configuration file not found: %s", args.config)
        sys.exit(1)

    recipe = TrainingRecipe.from_yaml(args.config)
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

    if args.epochs is not None:
        recipe.num_train_epochs = args.epochs
    if args.max_steps is not None:
        recipe.max_steps = args.max_steps
    if args.patience is not None:
        recipe.early_stopping_patience = args.patience

    raw_tag = (
        args.run_tag.strip().lstrip("_")
        if args.run_tag
        else ("unconstrained" if recipe.target_flop_budget is None else "baseline")
    )
    run_tag = f"_{raw_tag}"
    output_dir = args.output_dir or f"checkpoints/exp1_1_grpo{run_tag}_seed{recipe.seed}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.eval_output_dir, exist_ok=True)

    budget_disp = (
        f"{recipe.target_flop_budget:.2e} FLOPs"
        if recipe.target_flop_budget
        else "None (unconstrained)"
    )

    logger.info("=" * 70)
    logger.info("EXPERIMENT 1.1: Standalone GRPO Reinforcement Learning Baseline")
    logger.info("Base Model:    %s", recipe.base_model_name_or_path)
    logger.info("FLOP Budget:   %s", budget_disp)
    logger.info("Criterion:     %s", recipe.termination_criterion)
    logger.info("Group Size:    %d", recipe.group_size)
    logger.info("Seed:          %d", recipe.seed)
    logger.info("Output Dir:    %s", output_dir)
    logger.info("=" * 70)

    # 2. Prepare Training and Validation Data
    datasets_dict, raw_splits = prepare_datasets(recipe)

    # 3. Instantiate Runner and Execute Training
    runner = GRPORunner(recipe=recipe, output_dir=output_dir)
    train_result = runner.train(
        train_dataset=datasets_dict["train"],
        eval_dataset=datasets_dict["dev"],
    )

    logger.info("-" * 70)
    logger.info("Training Finished:")
    logger.info("  Cumulative FLOPs: %.3e", train_result["cumulative_flops"])
    logger.info("  Total Steps:      %d", train_result["total_steps"])
    logger.info("  Final Adapter:    %s", train_result["adapter_path"])
    logger.info("-" * 70)

    # 4. Post-Training Evaluation on Held-Out Split
    capability_profile: CapabilityProfile | None = None
    if not args.skip_eval:
        logger.info(
            "Evaluating final adapter checkpoint on GSM8K (%s split)...",
            args.eval_split,
        )
        del runner
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        eval_dataset = raw_splits.get(args.eval_split) or raw_splits.get("test", [])
        evaluator = GSM8KEvaluator(
            model_name_or_path=recipe.base_model_name_or_path,
            adapter_path=train_result["adapter_path"],
            temperature=0.0,
            max_tokens=recipe.max_completion_length,
            gpu_memory_utilization=0.6,
        )
        eval_result = evaluator.evaluate(
            dataset=eval_dataset,
            split_name=args.eval_split,
            num_shots=recipe.num_shots,
        )

        capability_profile = CapabilityProfile(
            benchmark="gsm8k",
            split=args.eval_split,
            num_shots=recipe.num_shots,
            accuracy=eval_result.accuracy,
            permissive_accuracy=eval_result.permissive_accuracy,
            format_compliance_rate=eval_result.format_compliance_rate,
            mean_completion_tokens=eval_result.mean_completion_tokens,
            error_indices=eval_result.error_indices,
            wall_time_seconds=eval_result.wall_time_seconds,
            metadata={
                "parser_exceptions": eval_result.parser_exceptions,
                "total_examples": eval_result.total_examples,
            },
        )

        # Save standalone evaluation JSON artifact
        eval_artifact_path = os.path.join(
            args.eval_output_dir,
            f"exp1_1_grpo{run_tag}_seed{recipe.seed}_{args.eval_split}.json",
        )
        with open(eval_artifact_path, "w", encoding="utf-8") as f:
            json.dump(eval_result.to_dict(), f, indent=2, default=str)
        logger.info("Saved evaluation artifact to %s", eval_artifact_path)

        logger.info("Evaluation Summary:")
        logger.info("  Strict Accuracy:           %.2f%%", eval_result.accuracy * 100)
        logger.info("  Permissive Accuracy:       %.2f%%", eval_result.permissive_accuracy * 100)
        logger.info("  Format Compliance Rate:    %.2f%%", eval_result.format_compliance_rate * 100)
        logger.info("  Mean Completion Tokens:    %.1f", eval_result.mean_completion_tokens)

    # 5. Register Model Node in Population DAG
    logger.info("Updating population lineage DAG: %s", args.dag_path)
    if os.path.exists(args.dag_path):
        dag = PopulationDAG.load_from_json(args.dag_path)
    else:
        dag = PopulationDAG(root_model_name=recipe.base_model_name_or_path)

    node_id = f"exp1_1_grpo{run_tag}_seed{recipe.seed}"
    node = ModelNode(
        node_id=node_id,
        parent_ids=["node_0"],
        depth=1,
        recipe=recipe,
        capability_profile=capability_profile,
        intermediate_evaluations=train_result.get("intermediate_checkpoints", []),
        adapter_path=train_result["adapter_path"],
        cumulative_flops=train_result["cumulative_flops"],
        metadata={
            "status": "completed",
            "total_steps": train_result["total_steps"],
            "is_budget_exhausted": train_result["is_exhausted"],
        },
    )
    dag.add_node(node)
    dag.save_to_json(args.dag_path)
    logger.info("Registered node %s (parent: node_0) in DAG -> %s", node_id, args.dag_path)

    logger.info("=" * 70)
    logger.info("EXP-1.1 Run Complete.")
    logger.info("=" * 70)

    # Clean up distributed process group if initialized by accelerate/TRL
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
