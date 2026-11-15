"""Unit tests for DistillRunner and two-stage distillation pipeline (multigen/training/distill_runner.py)."""

from __future__ import annotations

import json
import os
import tempfile
import pytest

from multigen.accounting.tracker import FLOPAccumulator
from multigen.core.population_dag import PopulationDAG
from multigen.core.schema import ModelNode, TrainingRecipe
from multigen.training.distill_runner import DistillRunner


def test_distill_runner_initialization():
    recipe = TrainingRecipe(
        algorithm="distill",
        teacher_model_name_or_path="Qwen/Qwen2.5-Math-7B-Instruct",
        teacher_k_samples=4,
        target_flop_budget=1.0e16,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = DistillRunner(recipe=recipe, output_dir=tmp_dir)
        assert runner.recipe.teacher_k_samples == 4
        assert runner.recipe.teacher_model_name_or_path == "Qwen/Qwen2.5-Math-7B-Instruct"
        assert runner.accumulator.target_budget == 1.0e16


def test_filter_and_format_traces_best_of_n():
    recipe = TrainingRecipe(algorithm="distill")
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = DistillRunner(recipe=recipe, output_dir=tmp_dir)

        questions = [
            "Janet's ducks lay 16 eggs per day. How many eggs in 3 days?",
            "A robe takes 2 bolts of blue fiber and half that of white. How many total bolts?",
        ]
        gold_answers = ["48", "3"]

        candidates = [
            # Problem 1 candidates: 1 wrong, 1 verbose correct, 1 concise correct
            [
                "<think>16 + 3 = 19 eggs.</think><answer>19</answer>",  # Wrong
                "<think>Let us calculate carefully step by step. Janet has ducks. 16 times 3 is forty eight.</think><answer>48</answer>",  # Long
                "<think>16 * 3 = 48.</think><answer>48</answer>",  # Short (Best)
            ],
            # Problem 2 candidates: 1 no tags, 1 correct
            [
                "The total is 3 bolts of fiber.",  # Malformed (no think tags)
                "<think>2 + 1 = 3 bolts.</think><answer>3</answer>",  # Correct
            ],
        ]

        verified, stats = runner.filter_and_format_traces(
            questions=questions,
            gold_answers=gold_answers,
            candidate_lists=candidates,
        )

        assert stats["total_problems"] == 2
        assert stats["problems_with_verified"] == 2
        assert stats["verification_coverage"] == 1.0
        assert len(verified) == 2

        # Check Best-of-N picked shortest valid trace for problem 1
        assert verified[0]["question"] == questions[0]
        assert "<think>" in verified[0]["response"]
        assert "16 * 3 = 48." in verified[0]["response"]
        assert "<answer>48</answer>" in verified[0]["response"]

        # Check problem 2 (which was wrapped from candidate that had tags)
        assert verified[1]["question"] == questions[1]
        assert "<think>" in verified[1]["response"]
        assert "2 + 1 = 3 bolts." in verified[1]["response"]
        assert "<answer>3</answer>" in verified[1]["response"]


def test_normalize_teacher_trace():
    from multigen.training.distill_runner import normalize_teacher_trace

    # Case 1: Standard CoT ending in \\boxed{}
    raw_1 = "To find the number of clips, Natalia has 48 and then 24. 48 + 24 = 72. The answer is \\boxed{72}."
    norm_1 = normalize_teacher_trace(raw_1, "72")
    assert norm_1 is not None
    assert norm_1.startswith("<think>\n")
    assert norm_1.endswith("\n<answer>72</answer>")
    assert "\\boxed" not in norm_1
    assert "48 + 24 = 72" in norm_1

    # Case 2: Candidate with <think> tags already
    raw_2 = "Intro text <think>Step 1: add numbers. 1 + 1 = 2.</think> extra text <answer>2</answer>"
    norm_2 = normalize_teacher_trace(raw_2, "2")
    assert norm_2 is not None
    assert norm_2 == "<think>\nStep 1: add numbers. 1 + 1 = 2.\n</think>\n<answer>2</answer>"


def test_filter_and_format_traces_all_incorrect():
    recipe = TrainingRecipe(algorithm="distill")
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = DistillRunner(recipe=recipe, output_dir=tmp_dir)

        questions = ["What is 10 + 10?"]
        gold_answers = ["20"]
        candidates = [
            [
                "<think>10 + 10 = 99.</think><answer>99</answer>",
                "<think>10 + 10 = 100.</think><answer>100</answer>",
            ]
        ]

        verified, stats = runner.filter_and_format_traces(
            questions=questions,
            gold_answers=gold_answers,
            candidate_lists=candidates,
        )

        assert len(verified) == 0
        assert stats["problems_with_verified"] == 0
        assert stats["verification_coverage"] == 0.0


def test_distillation_budget_debiting_modes():
    # Test strict split mode
    recipe_strict = TrainingRecipe(
        algorithm="distill",
        accounting_mode="strict_split",
        target_flop_budget=1.0e16,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = DistillRunner(recipe=recipe_strict, output_dir=tmp_dir)
        # Mock teacher generation
        candidates = [["<think>4*5=20.</think><answer>20</answer>"]]
        verified, stats = runner.filter_and_format_traces(["4*5?"], ["20"], candidates)

        # Debit teacher FLOPs
        teacher_flops = 2.0 * runner.teacher_params * (60 + stats["total_teacher_tokens"])
        runner.accumulator.debit(teacher_flops, category="teacher", tokens=stats["total_teacher_tokens"])

        assert runner.accumulator.cumulative_flops == teacher_flops
        assert runner.accumulator.teacher_flops == teacher_flops

    # Test amortized mode
    recipe_amortized = TrainingRecipe(
        algorithm="distill",
        accounting_mode="amortized",
        target_flop_budget=1.0e16,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = DistillRunner(recipe=recipe_amortized, output_dir=tmp_dir)
        # In amortized mode, teacher inference is logged in teacher ledger without debiting active training
        runner.accumulator.teacher_flops += 5.0e15
        assert runner.accumulator.cumulative_flops == 0.0
        assert runner.accumulator.teacher_flops == 5.0e15


def test_teacher_traces_caching():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_file = os.path.join(tmp_dir, "traces.jsonl")
        sample_data = [
            {"question": "Q1", "response": "<think>...</think><answer>1</answer>", "answer": "1"}
        ]
        with open(cache_file, "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")

        recipe = TrainingRecipe(algorithm="distill", teacher_trace_dataset_path=cache_file)
        runner = DistillRunner(recipe=recipe, output_dir=tmp_dir)

        # Calling generate_and_filter should load from cache without generating
        traces, stats = runner.generate_and_filter_teacher_traces(
            questions=["Q1"],
            gold_answers=["1"],
            output_path=cache_file,
            force_regenerate=False,
        )
        assert len(traces) == 1
        assert stats.get("cached") is True
        assert traces[0]["question"] == "Q1"


def test_dag_teacher_and_distill_edge():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dag_path = os.path.join(tmp_dir, "dag.json")
        dag = PopulationDAG()

        # Register teacher node
        teacher_node = dag.register_teacher_node(
            teacher_id="teacher_qwen25_math_7b",
            model_name_or_path="Qwen/Qwen2.5-Math-7B-Instruct",
        )
        assert dag.has_node("teacher_qwen25_math_7b")
        assert teacher_node.node_id == "teacher_qwen25_math_7b"

        # Register student node
        recipe = TrainingRecipe(algorithm="distill")
        student_node = ModelNode(
            node_id="exp1_2_distill_seed42",
            parent_ids=["node_0"],
            recipe=recipe,
        )
        dag.add_node(student_node, edge_type="mutation")
        dag.add_edge("teacher_qwen25_math_7b", "exp1_2_distill_seed42", edge_type="distillation")

        assert dag.has_node("exp1_2_distill_seed42")
        edge_data = dag.graph.get_edge_data("teacher_qwen25_math_7b", "exp1_2_distill_seed42")
        assert edge_data["edge_type"] == "distillation"

        # Save and reload
        dag.save_to_json(dag_path)
        reloaded = PopulationDAG.load_from_json(dag_path)
        assert reloaded.has_node("teacher_qwen25_math_7b")
        assert reloaded.has_node("exp1_2_distill_seed42")
        reloaded_edge = reloaded.graph.get_edge_data("teacher_qwen25_math_7b", "exp1_2_distill_seed42")
        assert reloaded_edge["edge_type"] == "distillation"
