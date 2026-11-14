"""Unit tests for GRPO runner and FLOP budget callback integration (multigen/training/grpo_runner.py)."""

from __future__ import annotations

import tempfile
import pytest

from multigen.accounting.tracker import FLOPAccumulator
from multigen.core.schema import TrainingRecipe
from multigen.training.grpo_runner import GRPORunner, GRPOBudgetTrackingCallback


class DummyControl:
    def __init__(self):
        self.should_training_stop = False
        self.should_save = False


class DummyState:
    def __init__(self, step=1):
        self.global_step = step


def test_grpo_runner_initialization():
    recipe = TrainingRecipe(group_size=8, target_flop_budget=1.0e16)
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = GRPORunner(recipe=recipe, output_dir=tmp_dir)
        assert runner.recipe.group_size == 8
        assert runner.accumulator.target_budget == 1.0e16
        assert runner.accumulator.cumulative_flops == 0.0


def test_grpo_runner_reward_functions_and_flop_debit():
    recipe = TrainingRecipe(
        correctness_reward_weight=1.0,
        format_reward_weight=0.1,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = GRPORunner(recipe=recipe, output_dir=tmp_dir)

        # Build reward functions with mock parameter counts
        n_base = 500_000_000
        n_lora = 5_000_000
        # per-token coefficient = 8 * 5e8 + 6 * 5e6 = 4.0e9 + 3.0e7 = 4.03e9
        correctness_fn, format_fn = runner.build_reward_functions(
            tokenizer=None,
            n_base=n_base,
            n_lora=n_lora,
        )

        prompts = ["What is 2 + 2?", "What is 5 * 5?"]
        completions = [
            "<think>2 + 2 is 4</think><answer>4</answer>",
            "<think>5 * 5 is 25</think><answer>20</answer>",
        ]
        answers = ["4", "25"]

        # Call format function
        format_scores = format_fn(prompts, completions)
        assert len(format_scores) == 2
        assert format_scores[0] == pytest.approx(0.1)
        assert format_scores[1] == pytest.approx(0.1)

        # Call correctness function
        correctness_scores = correctness_fn(prompts, completions, answer=answers)
        assert len(correctness_scores) == 2
        assert correctness_scores[0] == pytest.approx(1.0)
        assert correctness_scores[1] == pytest.approx(0.0)

        # Verify FLOP accumulator debited compute
        assert runner.accumulator.cumulative_flops > 0
        assert runner.accumulator.total_tokens_processed > 0


def test_grpo_budget_callback_halting():
    accumulator = FLOPAccumulator(target_budget=1.0e14)
    callback = GRPOBudgetTrackingCallback(
        accumulator=accumulator,
        intermediate_interval_flops=2.0e13,
    )

    control = DummyControl()
    state = DummyState(step=5)

    # Step 1: Under budget -> should not stop
    accumulator.debit(5.0e13, category="train")
    callback.on_step_end(args=None, state=state, control=control)
    assert control.should_training_stop is False

    # Step 2: Crosses budget -> must signal stop and save
    accumulator.debit(6.0e13, category="train")
    assert accumulator.is_exhausted is True
    callback.on_step_end(args=None, state=state, control=control)
    assert control.should_training_stop is True
    assert control.should_save is True


def test_grpo_budget_callback_intermediate_checkpoints():
    saved_milestones = []

    def mock_save_checkpoint(milestone_flops: float, step: int) -> str:
        saved_milestones.append((milestone_flops, step))
        return f"/mock/checkpoint_{int(milestone_flops):.0e}"

    accumulator = FLOPAccumulator(target_budget=1.0e15)
    callback = GRPOBudgetTrackingCallback(
        accumulator=accumulator,
        intermediate_interval_flops=2.0e14,
        save_checkpoint_fn=mock_save_checkpoint,
    )

    control = DummyControl()
    state = DummyState(step=10)

    # Debit past first milestone (2.0e14)
    accumulator.debit(2.5e14, category="train")
    callback.on_step_end(args=None, state=state, control=control)

    assert len(saved_milestones) == 1
    assert saved_milestones[0][0] == 2.0e14
    assert len(callback.saved_milestones) == 1


def test_vllm_detection_fallback(monkeypatch):
    import torch

    recipe = TrainingRecipe(use_vllm=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = GRPORunner(recipe=recipe, output_dir=tmp_dir)

        # Force cuda.is_available() to return False
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        # Should gracefully fall back to False
        assert runner._determine_vllm_usage() is False


def test_grpo_config_and_train_pipeline_wiring(monkeypatch):
    """End-to-end integration test ensuring runner.train() wires GRPOConfig and GRPOTrainer cleanly."""
    import inspect
    from unittest.mock import MagicMock
    import torch
    from trl import GRPOConfig

    # 1. Assert that max_prompt_length is NOT in GRPOConfig signature (the bug that failed on Modal)
    sig = inspect.signature(GRPOConfig.__init__)
    assert "num_generations" in sig.parameters
    assert "max_prompt_length" not in sig.parameters, (
        "max_prompt_length must not be passed to GRPOConfig"
    )

    recipe = TrainingRecipe(group_size=4, target_flop_budget=1.0e14)
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = GRPORunner(recipe=recipe, output_dir=tmp_dir)

        # Mock model and tokenizer to avoid downloading 1.2GB weights during unit tests
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        monkeypatch.setattr(runner, "setup_model_and_tokenizer", lambda: (mock_model, mock_tokenizer))

        # Use a mock GRPOConfig class with the exact signature of real GRPOConfig.__init__
        # to avoid local macOS PyTorch < 2.6 / numpy 2 C-extension incompatibility during unit tests
        captured_grpo_kwargs = {}
        class MockGRPOConfig:
            def __init__(self, **kwargs):
                captured_grpo_kwargs.update(kwargs)
                for k, v in kwargs.items():
                    setattr(self, k, v)

        MockGRPOConfig.__init__.__signature__ = inspect.signature(GRPOConfig.__init__)

        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.state.global_step = 42
        mock_trainer_cls.return_value = mock_trainer_instance

        import multigen.training.grpo_runner as grpo_module
        monkeypatch.setattr(grpo_module, "HAS_TRL", True)
        monkeypatch.setattr(grpo_module, "GRPOTrainer", mock_trainer_cls)
        monkeypatch.setattr(grpo_module, "GRPOConfig", MockGRPOConfig)

        # Execute train() with dummy dataset
        dummy_dataset = [{"prompt": "test prompt", "answer": "42"}]
        result = runner.train(train_dataset=dummy_dataset)

        # Verify GRPOTrainer was initialized with properly filtered arguments
        assert mock_trainer_cls.called
        call_kwargs = mock_trainer_cls.call_args[1]
        training_args = call_kwargs["args"]
        assert isinstance(training_args, MockGRPOConfig)
        assert training_args.num_generations == 4
        assert training_args.per_device_train_batch_size == 4
        assert training_args.generation_batch_size == 4
        assert captured_grpo_kwargs["generation_batch_size"] % training_args.num_generations == 0
        assert training_args.max_completion_length == 512
        assert "max_prompt_length" not in captured_grpo_kwargs

        # Verify output dictionary
        assert result["total_steps"] == 42
        assert "adapter_path" in result
        assert mock_model.save_pretrained.called

