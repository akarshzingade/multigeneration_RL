#!/usr/bin/env python3
"""CLI driver for Experiment 1.2: Standalone Knowledge Distillation (KD) Baseline.

Executes knowledge distillation from Qwen/Qwen2.5-Math-7B-Instruct teacher to
Qwen/Qwen3-0.6B student with PEFT LoRA under strict compute budgeting (1.0e16 FLOPs).
Evaluates the trained adapter on held-out GSM8K and registers the new child checkpoint
node and teacher reference node into the population lineage DAG.

Usage:
  # Full run: generate/load traces, train student, and evaluate
  python scripts/run_distill_experiment.py --config configs/experiments/exp1_2_distill_baseline.yaml --seed 42

  # Trace generation only
  python scripts/run_distill_experiment.py --config configs/experiments/exp1_2_distill_baseline.yaml --generate-traces-only
"""

from __future__ import annotations

import argparse
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
from typing import Any, Dict, List, Tuple
import datasets
import torch

from multigen.core.population_dag import PopulationDAG
from multigen.core.schema import CapabilityProfile, ModelNode, TrainingRecipe
from multigen.eval.evaluator import GSM8KEvaluator, split_gsm8k_dataset
from multigen.training.distill_runner import DistillRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_distill_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EXP-1.2 Standalone Knowledge Distillation (KD) Baseline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp1_2_distill_baseline.yaml",
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
        "--generate-traces-only",
        action="store_true",
        help="Only run Stage 1 (generate and verify teacher traces) and exit",
    )
    parser.add_argument(
        "--force-regenerate-traces",
        action="store_true",
        help="Force regeneration of teacher traces even if cached dataset exists",
    )
    parser.add_argument(
        "--traces-path",
        type=str,
        default=None,
        help="Path to save or load verified teacher traces JSONL",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Maximum training problems to generate/use (useful for quick test runs)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Maximum test problems to evaluate (useful for quick test runs)",
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


def load_gsm8k_problems(
    recipe: TrainingRecipe,
) -> Tuple[List[str], List[str], Dict[str, List[Dict[str, str]]]]:
    """Load GSM8K dataset and extract training question-answer pairs."""
    logger.info("Loading GSM8K dataset: %s (%s)", recipe.dataset_name, recipe.dataset_subset)
    raw_ds = datasets.load_dataset(recipe.dataset_name, recipe.dataset_subset)
    splits = split_gsm8k_dataset(raw_ds, dev_size=500, seed=recipe.seed)

    questions = [item["question"] for item in splits["train"]]
    answers = [item["gold_answer"] for item in splits["train"]]

    logger.info(
        "Extracted %d train problems, %d dev problems, %d test problems",
        len(questions),
        len(splits.get("dev", [])),
        len(splits.get("test", [])),
    )
    return questions, answers, splits


def main() -> None:
    args = parse_args()

    # 1. Load Recipe & apply overrides
    if not os.path.exists(args.config):
        logger.error("Configuration file not found: %s", args.config)
        sys.exit(1)

    recipe = TrainingRecipe.from_yaml(args.config)
    if args.seed is not None:
        recipe.seed = args.seed
    if args.target_flops is not None:
        recipe.target_flop_budget = args.target_flops
    if args.traces_path is not None:
        recipe.teacher_trace_dataset_path = args.traces_path

    output_dir = args.output_dir or f"checkpoints/exp1_2_distill_seed{recipe.seed}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.eval_output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("EXPERIMENT 1.2: Standalone Knowledge Distillation (KD) Baseline")
    logger.info("Student Model:  %s", recipe.base_model_name_or_path)
    logger.info("Teacher Model:  %s", recipe.teacher_model_name_or_path)
    logger.info("FLOP Budget:    %.2e FLOPs (%s mode)", recipe.target_flop_budget, recipe.accounting_mode)
    logger.info("Seed:           %d", recipe.seed)
    logger.info("Output Dir:     %s", output_dir)
    logger.info("=" * 70)

    # 2. Load GSM8K dataset
    train_questions, train_answers, raw_splits = load_gsm8k_problems(recipe)
    if args.max_train_samples is not None and args.max_train_samples > 0:
        logger.info("Subsampling training questions to %d samples for quick test run.", args.max_train_samples)
        train_questions = train_questions[:args.max_train_samples]
        train_answers = train_answers[:args.max_train_samples]

    # 3. Instantiate DistillRunner
    distill_runner = DistillRunner(recipe=recipe, output_dir=output_dir)

    # If traces-only requested, run Stage 1 and exit
    if args.generate_traces_only:
        logger.info("Running Stage 1 only: Generating and verifying teacher traces...")
        traces, stats = distill_runner.generate_and_filter_teacher_traces(
            questions=train_questions,
            gold_answers=train_answers,
            output_path=args.traces_path,
            force_regenerate=args.force_regenerate_traces,
        )
        logger.info("Stage 1 complete: %d verified traces saved.", len(traces))
        return

    # 4. Execute Full Two-Stage Distillation
    # SFT training terminates strictly on the compute budget; held-out evaluation is executed in Step 5
    result = distill_runner.run(
        questions=train_questions,
        gold_answers=train_answers,
        eval_dataset=None,
        output_path=args.traces_path,
        force_regenerate=args.force_regenerate_traces,
    )

    final_adapter = result["final_adapter_path"]
    budget_summary = result["budget_summary"]

    logger.info("-" * 70)
    logger.info("Distillation Training Finished:")
    logger.info("  Cumulative FLOPs: %.3e", budget_summary["cumulative_flops"])
    logger.info("  Final Adapter:    %s", final_adapter)
    logger.info("-" * 70)

    # 5. Post-Training Evaluation on Held-Out Split
    capability_profile: CapabilityProfile | None = None
    if not args.skip_eval:
        logger.info(
            "Evaluating distilled student adapter on GSM8K (%s split)...",
            args.eval_split,
        )
        del distill_runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        eval_dataset = raw_splits.get(args.eval_split) or raw_splits.get("test", [])
        if args.max_eval_samples is not None and args.max_eval_samples > 0:
            logger.info("Subsampling evaluation set to %d samples for quick test run.", args.max_eval_samples)
            eval_dataset = eval_dataset[:args.max_eval_samples]
        evaluator = GSM8KEvaluator(
            model_name_or_path=recipe.base_model_name_or_path,
            adapter_path=final_adapter,
            temperature=0.0,
            max_tokens=recipe.max_seq_length,
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
            f"exp1_2_distill_seed{recipe.seed}_{args.eval_split}.json",
        )
        with open(eval_artifact_path, "w", encoding="utf-8") as f:
            json.dump(eval_result.to_dict(), f, indent=2, default=str)
        logger.info("Saved evaluation artifact to %s", eval_artifact_path)

        logger.info("Evaluation Summary:")
        logger.info("  Strict Accuracy:           %.2f%%", eval_result.accuracy * 100)
        logger.info("  Permissive Accuracy:       %.2f%%", eval_result.permissive_accuracy * 100)
        logger.info("  Format Compliance Rate:    %.2f%%", eval_result.format_compliance_rate * 100)
        logger.info("  Mean Completion Tokens:    %.1f", eval_result.mean_completion_tokens)

    # 6. Register Teacher & Distilled Student in Population DAG
    logger.info("Updating population lineage DAG: %s", args.dag_path)
    if os.path.exists(args.dag_path):
        dag = PopulationDAG.load_from_json(args.dag_path)
    else:
        dag = PopulationDAG(root_model_name=recipe.base_model_name_or_path)

    # Register external teacher reference node
    teacher_id = "teacher_qwen25_math_7b"
    dag.register_teacher_node(
        teacher_id=teacher_id,
        model_name_or_path=recipe.teacher_model_name_or_path or "Qwen/Qwen2.5-Math-7B-Instruct",
        params=7.61e9,
        metadata={
            "description": "External frozen teacher model",
            "k_samples": recipe.teacher_k_samples,
            "traces_path": recipe.teacher_trace_dataset_path,
        },
    )

    # Register distilled student node
    node_id = f"exp1_2_distill_seed{recipe.seed}"
    student_node = ModelNode(
        node_id=node_id,
        parent_ids=["node_0"],
        depth=1,
        recipe=recipe,
        capability_profile=capability_profile,
        intermediate_evaluations=result.get("intermediate_checkpoints", []),
        adapter_path=final_adapter,
        cumulative_flops=budget_summary["cumulative_flops"],
        metadata={
            "status": "completed",
            "distillation_teacher": teacher_id,
            "accounting_mode": recipe.accounting_mode,
            "teacher_generation_flops": budget_summary["breakdown"].get("teacher_flops", 0.0),
            "student_training_flops": budget_summary["breakdown"].get("train_flops", 0.0),
            "is_budget_exhausted": budget_summary["is_exhausted"],
        },
    )
    dag.add_node(student_node, edge_type="mutation")
    dag.add_edge(teacher_id, node_id, edge_type="distillation")
    dag.save_to_json(args.dag_path)

    logger.info("Registered student %s with mutation edge from node_0 and distillation edge from %s in DAG.", node_id, teacher_id)
    logger.info("=" * 70)
    logger.info("EXP-1.2 Run Complete.")
    logger.info("=" * 70)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
