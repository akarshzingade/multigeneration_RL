"""Unit tests for compute accounting formulas, parameter extraction, and budget callbacks.

Supports both pytest and standard library unittest for cross-environment execution.
"""

from __future__ import annotations

import math
import unittest
from typing import List, Tuple

from multigen.accounting.flops import (
    ModelParamBreakdown,
    compute_grpo_step_flops,
    compute_rollout_flops,
    compute_teacher_flops,
    compute_train_step_flops,
    extract_param_counts,
)
from multigen.accounting.tracker import (
    FLOPAccumulator,
    FLOPBudgetCallback,
    TrainerControl,
    TrainerState,
)


class MockTensor:
    """Lightweight mock tensor for parameter counting without requiring PyTorch."""

    def __init__(self, shape: Tuple[int, ...], requires_grad: bool = True) -> None:
        self.shape = shape
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return math.prod(self.shape)


class MockModel:
    """Mock PyTorch / Hugging Face model."""

    def __init__(self, named_params: List[Tuple[str, MockTensor]]) -> None:
        self._named_params = named_params

    def named_parameters(self):
        return iter(self._named_params)

    def parameters(self):
        return iter([p for _, p in self._named_params])


class TestFLOPAccounting(unittest.TestCase):
    """Test suite for FLOP calculations, parameter counting, and budget callbacks."""

    def test_extract_param_counts_full_model(self):
        """Verify parameter extraction for standard full-parameter model."""
        params = [
            ("embed_tokens.weight", MockTensor((1000, 128), requires_grad=True)),  # 128,000
            ("layers.0.mlp.weight", MockTensor((128, 512), requires_grad=True)),   # 65,536
        ]
        model = MockModel(params)
        breakdown = extract_param_counts(model)

        self.assertEqual(breakdown.total_params, 193536)
        self.assertEqual(breakdown.trainable_params, 193536)
        self.assertEqual(breakdown.frozen_params, 0)
        self.assertFalse(breakdown.is_lora)
        self.assertEqual(breakdown.base_params, 193536)

    def test_extract_param_counts_lora_model(self):
        """Verify parameter extraction for model with frozen base and LoRA adapters."""
        params = [
            ("base_model.layer.weight", MockTensor((512, 512), requires_grad=False)),         # 262,144 frozen
            ("base_model.layer.lora_A.weight", MockTensor((512, 8), requires_grad=True)),      # 4,096 trainable
            ("base_model.layer.lora_B.weight", MockTensor((8, 512), requires_grad=True)),      # 4,096 trainable
        ]
        model = MockModel(params)
        breakdown = extract_param_counts(model)

        self.assertEqual(breakdown.total_params, 262144 + 8192)
        self.assertEqual(breakdown.frozen_params, 262144)
        self.assertEqual(breakdown.trainable_params, 8192)
        self.assertTrue(breakdown.is_lora)
        self.assertEqual(breakdown.lora_params, 8192)
        self.assertEqual(breakdown.base_params, 262144)

    def test_compute_rollout_flops(self):
        """Verify rollout FLOP formula: 2 * N * (T_prompt + T_gen)."""
        n_params = 600_000_000  # 600M params
        prompt_tokens = 50
        gen_tokens = 150
        expected = 2.0 * 600_000_000 * 200  # 2.4e11

        flops = compute_rollout_flops(n_params, prompt_tokens, gen_tokens)
        self.assertEqual(flops, expected)

        # Linearity check: doubling gen tokens
        flops_doubled = compute_rollout_flops(n_params, prompt_tokens, gen_tokens * 2)
        self.assertEqual(flops_doubled, 2.0 * 600_000_000 * 350)

    def test_compute_rollout_flops_with_attention(self):
        """Verify quadratic attention option adds expected FLOPs."""
        n_params = 100_000_000
        prompt_t = 100
        gen_t = 100
        layers = 12
        hidden = 768

        dense_only = compute_rollout_flops(n_params, prompt_t, gen_t, include_attention_flops=False)
        with_attn = compute_rollout_flops(
            n_params, prompt_t, gen_t, include_attention_flops=True, num_layers=layers, hidden_size=hidden
        )

        expected_attn = 2.0 * layers * (200**2) * hidden
        self.assertEqual(with_attn, dense_only + expected_attn)

    def test_compute_train_step_flops_full_vs_lora(self):
        """Verify full fine-tuning (6NT) vs exact LoRA fine-tuning (4N_base*T + 6N_lora*T)."""
        n_total = 600_000_000
        n_lora = 6_000_000  # 1% adapter params
        seq_len = 512

        # Full fine-tuning: 6 * N * T
        flops_full = compute_train_step_flops(num_params=n_total, seq_len=seq_len, is_lora=False)
        self.assertEqual(flops_full, 6.0 * n_total * seq_len)

        # LoRA fine-tuning: 4 * N_base * T + 6 * N_lora * T
        flops_lora = compute_train_step_flops(
            num_params=n_total, seq_len=seq_len, is_lora=True, trainable_params=n_lora
        )
        expected_lora = (4.0 * (n_total - n_lora) * seq_len) + (6.0 * n_lora * seq_len)
        self.assertEqual(flops_lora, expected_lora)

        # LoRA compute must be strictly less than full fine-tuning
        self.assertLess(flops_lora, flops_full)
        ratio = flops_lora / flops_full
        self.assertTrue(0.66 < ratio < 0.68)

    def test_compute_teacher_flops(self):
        """Verify teacher rationale generation: 2 * N_teacher * (T_prompt + T_gen)."""
        teacher_params = 7_000_000_000  # 7B
        prompt_tokens = 100
        gen_tokens = 400

        flops = compute_teacher_flops(teacher_params, prompt_tokens, gen_tokens)
        self.assertEqual(flops, 2.0 * 7_000_000_000 * 500)

    def test_compute_grpo_step_flops(self):
        """Verify multi-candidate GRPO step accounting across all phases."""
        student_params = 600_000_000
        prompt_tokens = 50
        completions = [100, 150, 200, 250]  # G=4 completions
        lora_params = 6_000_000

        # With reference model counted (Decision 2)
        step_with_ref = compute_grpo_step_flops(
            student_params=student_params,
            prompt_tokens=prompt_tokens,
            completion_tokens_list=completions,
            is_lora=True,
            student_trainable_params=lora_params,
            count_ref_model=True,
        )

        self.assertGreater(step_with_ref["rollout_flops"], 0)
        self.assertGreater(step_with_ref["ref_flops"], 0)
        self.assertGreater(step_with_ref["train_flops"], 0)
        self.assertEqual(
            step_with_ref["total_flops"],
            step_with_ref["rollout_flops"] + step_with_ref["ref_flops"] + step_with_ref["train_flops"],
        )

        # Without reference model counted
        step_no_ref = compute_grpo_step_flops(
            student_params=student_params,
            prompt_tokens=prompt_tokens,
            completion_tokens_list=completions,
            is_lora=True,
            student_trainable_params=lora_params,
            count_ref_model=False,
        )
        self.assertEqual(step_no_ref["ref_flops"], 0.0)
        self.assertLess(step_no_ref["total_flops"], step_with_ref["total_flops"])

    def test_flop_accumulator_budget_and_eval_exemption(self):
        """Verify accumulator tracking, budget exhaustion, and evaluation exemption (Decision 4)."""
        target = 1.0e12
        acc = FLOPAccumulator(target_budget=target)

        self.assertFalse(acc.is_exhausted)
        self.assertEqual(acc.budget_utilization, 0.0)
        self.assertEqual(acc.remaining_flops, target)

        # Debit 6.0e11 training FLOPs (60%)
        acc.debit(6.0e11, category="train", tokens=1000)
        self.assertEqual(acc.cumulative_flops, 6.0e11)
        self.assertEqual(acc.budget_utilization, 0.6)
        self.assertFalse(acc.is_exhausted)

        # Debit 3.0e11 evaluation FLOPs (Decision 4: MUST be exempt from training budget)
        acc.record_eval_pass(flops=3.0e11, tokens=500)
        self.assertEqual(acc.eval_flops, 3.0e11)
        self.assertEqual(acc.cumulative_flops, 6.0e11)  # Did NOT increase
        self.assertEqual(acc.budget_utilization, 0.6)    # Still 60%
        self.assertFalse(acc.is_exhausted)

        # Debit 4.5e11 rollout FLOPs (crosses 100% budget -> 1.05e12)
        acc.debit(4.5e11, category="rollout", tokens=800)
        self.assertTrue(acc.is_exhausted)
        self.assertEqual(acc.cumulative_flops, 1.05e12)
        self.assertEqual(acc.remaining_flops, 0.0)
        self.assertTrue(math.isclose(acc.overshoot_fraction, 0.05, rel_tol=1e-3))

        summary = acc.get_summary()
        self.assertTrue(summary["is_exhausted"])
        self.assertEqual(summary["breakdown"]["eval_flops"], 3.0e11)
        self.assertEqual(summary["breakdown"]["train_flops"], 6.0e11)
        self.assertEqual(summary["breakdown"]["rollout_flops"], 4.5e11)

    def test_flop_budget_callback_graceful_stop(self):
        """Verify TrainerCallback triggers graceful halt at step boundary (Decision 5)."""
        target = 5.0e11
        acc = FLOPAccumulator(target_budget=target)
        callback = FLOPBudgetCallback(accumulator=acc)

        control = TrainerControl()
        state = TrainerState()

        # Step 1: Under budget
        acc.debit(2.0e11, category="train")
        control = callback.on_step_end(args=None, state=state, control=control)
        self.assertFalse(control.should_training_stop)
        self.assertFalse(control.should_save)

        # Step 2: Under budget
        acc.debit(2.0e11, category="train")
        control = callback.on_step_end(args=None, state=state, control=control)
        self.assertFalse(control.should_training_stop)

        # Step 3: Hits and slightly exceeds budget
        acc.debit(1.01e11, category="train")
        self.assertTrue(acc.is_exhausted)
        control = callback.on_step_end(args=None, state=state, control=control)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(control.should_save)

    def test_linear_scaling_linearity_r_squared(self):
        """Verify that theoretical FLOPs scale strictly linearly with token length (R^2 >= 0.9999)."""
        num_params = 600_000_000
        seq_lens = [64, 128, 256, 512, 1024, 2048]
        flops_values = [compute_train_step_flops(num_params, s, is_lora=False) for s in seq_lens]

        n = len(seq_lens)
        mean_x = sum(seq_lens) / n
        mean_y = sum(flops_values) / n

        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(seq_lens, flops_values))
        var_x = sum((x - mean_x) ** 2 for x in seq_lens)
        var_y = sum((y - mean_y) ** 2 for y in flops_values)

        r = cov_xy / math.sqrt(var_x * var_y)
        r_squared = r**2

        self.assertGreaterEqual(r_squared, 0.9999)


if __name__ == "__main__":
    unittest.main()
