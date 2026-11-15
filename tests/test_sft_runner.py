"""Unit tests for SFT runner and FLOP budget callback integration (multigen/training/sft_runner.py)."""

from __future__ import annotations

import tempfile
import pytest

from multigen.accounting.tracker import FLOPAccumulator
from multigen.core.schema import TrainingRecipe
from multigen.training.sft_runner import SFTRunner, SFTBudgetTrackingCallback


class DummyControl:
    def __init__(self):
        self.should_training_stop = False
        self.should_save = False


class DummyState:
    def __init__(self, step=1):
        self.global_step = step


def test_sft_runner_initialization():
    recipe = TrainingRecipe(
        algorithm="distill",
        learning_rate=2e-4,
        batch_size=8,
        target_flop_budget=1.0e16,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = SFTRunner(recipe=recipe, output_dir=tmp_dir)
        assert runner.recipe.learning_rate == 2e-4
        assert runner.recipe.batch_size == 8
        assert runner.accumulator.target_budget == 1.0e16
        assert runner.accumulator.cumulative_flops == 0.0


def test_sft_runner_prepare_dataset():
    recipe = TrainingRecipe(response_template="<think>")
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = SFTRunner(recipe=recipe, output_dir=tmp_dir)

        # 1. Empty or None datasets
        assert runner.prepare_dataset(None) is None
        assert runner.prepare_dataset([]) is None

        # 2. Raw list of dicts (like teacher traces or dev sets)
        raw_data = [
            {
                "question": "What is 2 + 2?",
                "response": "<think>2+2 is 4.</think><answer>4</answer>",
            },
            {
                "question": "What is 5 * 6?",
                "response": "<think>5*6 is 30.</think><answer>30</answer>",
            },
        ]
        prepared = runner.prepare_dataset(raw_data)
        assert len(prepared) == 2
        assert hasattr(prepared, "column_names")
        assert "text" in prepared.column_names
        assert "Question: What is 2 + 2?" in prepared[0]["text"]
        assert "<think>2+2 is 4.</think><answer>4</answer>" in prepared[0]["text"]

        # 3. Already converted Hugging Face Dataset idempotency
        re_prepared = runner.prepare_dataset(prepared)
        assert re_prepared is prepared


def test_sft_budget_callback_halting():
    accumulator = FLOPAccumulator(target_budget=1.0e14)
    callback = SFTBudgetTrackingCallback(
        accumulator=accumulator,
        intermediate_interval_flops=2.0e13,
    )

    control = DummyControl()
    state = DummyState(step=5)

    # Step 1: Under budget
    accumulator.debit(5.0e13, category="train")
    callback.on_step_end(args=None, state=state, control=control)
    assert control.should_training_stop is False

    # Step 2: Crosses budget
    accumulator.debit(6.0e13, category="train")
    assert accumulator.is_exhausted is True
    callback.on_step_end(args=None, state=state, control=control)
    assert control.should_training_stop is True
    assert control.should_save is True


def test_sft_budget_callback_intermediate_checkpoints():
    saved_milestones = []

    def mock_save_checkpoint(milestone_flops: float, step: int) -> str:
        saved_milestones.append((milestone_flops, step))
        return f"/mock/checkpoint_{int(milestone_flops):.0e}"

    accumulator = FLOPAccumulator(target_budget=1.0e15)
    callback = SFTBudgetTrackingCallback(
        accumulator=accumulator,
        intermediate_interval_flops=2.0e14,
        save_checkpoint_fn=mock_save_checkpoint,
    )

    control = DummyControl()
    state = DummyState(step=10)

    # Debit past milestone 1 (2.0e14)
    accumulator.debit(2.5e14, category="train")
    callback.on_step_end(args=None, state=state, control=control)

    assert len(saved_milestones) == 1
    assert saved_milestones[0][0] == 2.0e14

    # Debit past milestone 2 (4.0e14)
    accumulator.debit(2.0e14, category="train")
    callback.on_step_end(args=None, state=state, control=control)

    assert len(saved_milestones) == 2
    assert saved_milestones[1][0] == 4.0e14
