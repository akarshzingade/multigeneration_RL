"""Unit tests for unconstrained (no FLOP constraint) training, plateau callbacks, and stage transitions."""

from __future__ import annotations

import os
from unittest.mock import MagicMock
import pytest

from multigen.accounting.tracker import FLOPAccumulator, FLOPBudgetCallback
from multigen.core.schema import TrainingRecipe
from multigen.training.grpo_runner import GRPOPlateauCallback, GRPORunner
from multigen.training.mixed_runner import MixedRunner
from multigen.training.sft_runner import SFTRunner


def test_load_unconstrained_yaml_configs():
    """Verify that all three unconstrained YAML configs parse correctly."""
    r1 = TrainingRecipe.from_yaml("configs/experiments/exp1_1_grpo_unconstrained.yaml")
    assert r1.target_flop_budget is None
    assert r1.termination_criterion == "plateau"
    assert r1.early_stopping_patience == 3

    r2 = TrainingRecipe.from_yaml("configs/experiments/exp1_2_distill_unconstrained.yaml")
    assert r2.target_flop_budget is None
    assert r2.termination_criterion == "plateau"
    assert r2.early_stopping_metric == "eval_loss"

    r3 = TrainingRecipe.from_yaml("configs/experiments/exp1_3_mixed_unconstrained.yaml")
    assert r3.target_flop_budget is None
    assert r3.stage_transition_criterion == "plateau"
    assert r3.termination_criterion == "plateau"
    assert r3.sft_epochs == 1.0
    assert r3.grpo_epochs == 2.0


def test_flop_accumulator_unconstrained():
    """FLOPAccumulator with target_budget=None should never be exhausted but still track FLOPs."""
    acc = FLOPAccumulator(target_budget=None)
    assert not acc.is_exhausted
    assert acc.remaining_flops == float("inf")
    assert acc.budget_utilization == 0.0

    acc.debit(1e15, category="train")
    assert acc.cumulative_flops == 1e15
    assert not acc.is_exhausted

    # FLOPBudgetCallback should not trigger training stop
    callback = FLOPBudgetCallback(accumulator=acc)
    control = MagicMock()
    control.should_training_stop = False
    control.should_save = False

    callback.on_step_end(args=MagicMock(), state=MagicMock(), control=control)
    assert not control.should_training_stop


def test_grpo_plateau_callback():
    """GRPOPlateauCallback triggers early stopping after patience checks without improvement."""
    callback = GRPOPlateauCallback(patience=3, threshold=0.01, metric_name="reward", mode="max", window_size=1)
    control = MagicMock()
    control.should_training_stop = False
    control.should_save = False
    state = MagicMock()
    state.global_step = 10

    # Step 1: initial reward 0.50
    callback.on_log(args=MagicMock(), state=state, control=control, logs={"reward": 0.50})
    assert callback.patience_counter == 0
    assert not control.should_training_stop

    # Step 2: improved reward 0.60
    state.global_step = 20
    callback.on_log(args=MagicMock(), state=state, control=control, logs={"reward": 0.60})
    assert callback.patience_counter == 0
    assert not control.should_training_stop

    # Step 3: no improvement (0.601, diff < threshold 0.01) -> patience = 1
    state.global_step = 30
    callback.on_log(args=MagicMock(), state=state, control=control, logs={"reward": 0.601})
    assert callback.patience_counter == 1
    assert not control.should_training_stop

    # Step 4: lower reward (0.59) -> patience = 2
    state.global_step = 40
    callback.on_log(args=MagicMock(), state=state, control=control, logs={"reward": 0.59})
    assert callback.patience_counter == 2
    assert not control.should_training_stop

    # Step 5: plateau sustained (0.58) -> patience = 3 (triggers stop!)
    state.global_step = 50
    callback.on_log(args=MagicMock(), state=state, control=control, logs={"reward": 0.58})
    assert callback.patience_counter == 3
    assert callback.is_plateaued
    assert control.should_training_stop
    assert control.should_save


def test_mixed_runner_unconstrained_setup(tmp_path):
    """Verify MixedRunner properly initializes without FLOP constraint."""
    recipe = TrainingRecipe(
        algorithm="mixed",
        target_flop_budget=None,
        stage_transition_criterion="plateau",
        termination_criterion="plateau",
        sft_epochs=1.0,
        grpo_epochs=2.0,
    )
    runner = MixedRunner(recipe=recipe, output_dir=str(tmp_path / "mixed_out"))
    assert runner.accumulator.target_budget is None
    assert runner.recipe.stage_transition_criterion == "plateau"
