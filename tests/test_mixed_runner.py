"""Unit tests for MixedRunner and mixed learning orchestration (multigen/training/mixed_runner.py)."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from multigen.accounting.tracker import FLOPAccumulator
from multigen.core.population_dag import PopulationDAG
from multigen.core.schema import CapabilityProfile, ModelNode, TrainingRecipe
from multigen.training.mixed_runner import MixedRunner
from multigen.training.recovery import ExperimentStateJournal


def test_mixed_runner_initialization_from_yaml():
    yaml_content = """
experiment_id: "exp1_3_test"
algorithm: "mixed"
mode: "interleaved_sequential"
num_cycles: 4
ref_policy_source: "sft_warmup"
optimizer_transition: "reset_and_warmup"

model:
  name_or_path: "Qwen/Qwen3-0.6B"

teacher:
  name_or_path: "Qwen/Qwen2.5-Math-7B-Instruct"

sft_phase:
  distill_budget_ratio: 0.25
  learning_rate: 2.0e-4

grpo_phase:
  learning_rate: 2.0e-5
  group_size: 8

budget:
  target_flops: 1.0e16

resumption:
  enabled: true
  auto_resume: true
  checkpoint_interval_steps: 50
  enable_step_checkpoints: true
  enable_milestone_checkpoints: true
  enable_signal_trapping: false
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = os.path.join(tmp_dir, "test_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        recipe = TrainingRecipe.from_yaml(config_path)
        assert recipe.algorithm == "mixed"
        assert recipe.mode == "interleaved_sequential"
        assert recipe.distill_budget_ratio == 0.25
        assert recipe.ref_policy_source == "sft_warmup"
        assert recipe.optimizer_transition == "reset_and_warmup"
        assert recipe.enable_step_checkpoints is True
        assert recipe.checkpoint_interval_steps == 50
        assert recipe.enable_signal_trapping is False

        runner = MixedRunner(recipe=recipe, output_dir=tmp_dir)
        assert runner.recipe.mode == "interleaved_sequential"
        assert runner.accumulator.target_budget == 1.0e16
        assert runner.signal_trap is None  # Since enable_signal_trapping is False


def test_mixed_runner_checkpoint_saving():
    recipe = TrainingRecipe(
        algorithm="mixed",
        target_flop_budget=1.0e16,
        enable_signal_trapping=False,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = MixedRunner(recipe=recipe, output_dir=tmp_dir)
        runner.accumulator.debit(3.0e15, category="train")

        ckpt_path = runner.save_checkpoint(tag="test", step=100)
        assert os.path.exists(ckpt_path)
        assert os.path.exists(os.path.join(ckpt_path, "flops_ledger.json"))

        # Verify journal recorded ledger
        journal = ExperimentStateJournal(os.path.join(tmp_dir, "recovery_state.json"))
        assert journal.state["ledger"]["cumulative_flops"] == 3.0e15


def test_mixed_runner_rehydration_on_resume():
    recipe = TrainingRecipe(
        algorithm="mixed",
        target_flop_budget=1.0e16,
        enable_signal_trapping=False,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = os.path.join(tmp_dir, "recovery_state.json")
        journal = ExperimentStateJournal(journal_path)

        acc = FLOPAccumulator(target_budget=1.0e16)
        acc.debit(2.5e15, category="train", tokens=250000, step=300)
        journal.update_ledger(acc)
        journal.update_phase("sft", status="completed", checkpoint_path="/saved/sft/adapter")

        # Now instantiate MixedRunner with resume=True
        runner = MixedRunner(recipe=recipe, output_dir=tmp_dir, resume=True)

        assert runner.accumulator.cumulative_flops == 2.5e15
        assert runner.accumulator.total_tokens_processed == 250000
        assert runner.journal.is_phase_completed("sft")
        assert runner.journal.get_phase("sft")["checkpoint_path"] == "/saved/sft/adapter"


def test_mixed_runner_interleaved_resumption_skips_completed_sft():
    recipe = TrainingRecipe(
        algorithm="mixed",
        mode="interleaved_sequential",
        target_flop_budget=1.0e16,
        distill_budget_ratio=0.25,
        enable_signal_trapping=False,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Pre-populate journal with completed SFT phase
        journal_path = os.path.join(tmp_dir, "recovery_state.json")
        journal = ExperimentStateJournal(journal_path)
        mock_sft_adapter = os.path.join(tmp_dir, "mock_sft_adapter")
        os.makedirs(mock_sft_adapter, exist_ok=True)

        acc = FLOPAccumulator(target_budget=1.0e16)
        acc.debit(2.5e15, category="train", step=300)
        journal.update_ledger(acc)
        journal.update_phase("sft", status="completed", checkpoint_path=mock_sft_adapter)

        runner = MixedRunner(recipe=recipe, output_dir=tmp_dir, resume=True)

        # Mock GRPORunner so we don't need actual GPU execution
        mock_final_adapter = os.path.join(tmp_dir, "mock_grpo_adapter")
        os.makedirs(mock_final_adapter, exist_ok=True)

        with patch("multigen.training.mixed_runner.GRPORunner") as mock_grpo_cls:
            mock_grpo_instance = MagicMock()
            mock_grpo_instance.train.return_value = {
                "adapter_path": mock_final_adapter,
                "cumulative_flops": 1.0e16,
            }
            mock_grpo_cls.return_value = mock_grpo_instance

            # Also ensure SFTRunner/DistillRunner is NEVER called because SFT was already completed!
            with patch("multigen.training.mixed_runner.SFTRunner") as mock_sft_cls:
                with patch("multigen.training.mixed_runner.DistillRunner") as mock_distill_cls:
                    result = runner.run()

                    mock_sft_cls.assert_not_called()
                    mock_distill_cls.assert_not_called()
                    mock_grpo_cls.assert_called_once()

                    # Verify grpo was configured with parent_adapter_path set to mock_sft_adapter
                    call_kwargs = mock_grpo_cls.call_args[1]
                    assert call_kwargs["recipe"].parent_adapter_path == mock_sft_adapter
                    assert result["sft_adapter_path"] == mock_sft_adapter
                    assert result["final_adapter_path"] == mock_final_adapter
                    assert runner.journal.is_phase_completed("grpo")


def test_mixed_runner_register_in_population_dag_dual_parents():
    recipe = TrainingRecipe(
        algorithm="mixed",
        seed=42,
        teacher_model_name_or_path="Qwen/Qwen2.5-Math-7B-Instruct",
        target_flop_budget=1.0e16,
        enable_signal_trapping=False,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = MixedRunner(recipe=recipe, output_dir=tmp_dir)
        runner.accumulator.debit(1.0e16, category="train")

        dag_path = os.path.join(tmp_dir, "population_dag.json")
        dag = PopulationDAG.load_or_create(dag_path)
        # Root node_0
        dag.add_node(
            ModelNode(
                node_id="node_0",
                parent_ids=[],
                depth=0,
                recipe=TrainingRecipe(),
                cumulative_flops=0.0,
            )
        )
        dag.save(dag_path)

        profile = CapabilityProfile(
            benchmark="gsm8k",
            accuracy=0.55,
            permissive_accuracy=0.58,
            format_compliance_rate=0.98,
        )

        child_node = runner.register_in_population_dag(
            dag_path=dag_path,
            adapter_path="/fake/adapter",
            capability_profile=profile,
        )

        assert child_node.node_id == "exp1_3_mixed_seed42"
        # Verify dual parent lineage (Decision 4.7)
        assert "node_0" in child_node.parent_ids
        assert "teacher_qwen25_math_7b" in child_node.parent_ids
        assert child_node.capability_profile.accuracy == 0.55

        # Verify DAG state reloads with the dual parents
        reloaded_dag = PopulationDAG.load_or_create(dag_path)
        assert reloaded_dag.has_node("exp1_3_mixed_seed42")
        assert reloaded_dag.has_node("teacher_qwen25_math_7b")
        node_entry = reloaded_dag.get_node("exp1_3_mixed_seed42")
        assert node_entry.parent_ids == ["node_0", "teacher_qwen25_math_7b"]
