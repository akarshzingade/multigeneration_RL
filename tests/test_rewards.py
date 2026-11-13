"""Unit test suite for math reward functions and verification callbacks.

Validates that math-verify / fallback logic satisfies 100% of calibration
edge cases specified in EXP-0.1. Can be run with either pytest or python3 -m unittest.
"""

import unittest
from multigen.eval.rewards import (
    HAS_MATH_VERIFY,
    extract_answer,
    verify_math_answer,
    correctness_reward_fn,
    format_reward_fn,
)


class TestAnswerExtraction(unittest.TestCase):
    """Tests for extract_answer with balanced braces and tag priority."""

    def test_answer_tag_priority(self):
        text = "Step 1: 10.\n#### 20\n<answer>42</answer>"
        self.assertEqual(extract_answer(text), "42")

    def test_boxed_balanced_braces(self):
        text = r"The solution is \boxed{\frac{3}{4}}."
        self.assertEqual(extract_answer(text), r"\frac{3}{4}")

    def test_boxed_nested_braces(self):
        text = r"Result: \boxed{\textbf{\frac{1}{2}}}"
        self.assertEqual(extract_answer(text), r"\textbf{\frac{1}{2}}")

    def test_gsm8k_hash_marker(self):
        text = "The total cost is 50 dollars.\n#### 50"
        self.assertEqual(extract_answer(text), "50")

    def test_partial_number_trap_strict(self):
        text = "Step 1: 10. The intermediate sum is 20.\n<answer>12</answer>"
        # Must extract 12, never 10 or 20
        self.assertEqual(extract_answer(text), "12")

    def test_permissive_fallback_disabled(self):
        text = "The answer is 42."
        self.assertIsNone(extract_answer(text, allow_last_number_fallback=False))

    def test_permissive_fallback_enabled(self):
        text = "The answer is 42."
        self.assertEqual(extract_answer(text, allow_last_number_fallback=True), "42")

    def test_empty_and_malformed(self):
        self.assertIsNone(extract_answer(""))
        self.assertIsNone(extract_answer(None))
        self.assertIsNone(extract_answer("<answer></answer>"))
        self.assertIsNone(extract_answer(r"\boxed{\frac{1}{"))


class TestVerifyMathAnswer(unittest.TestCase):
    """Tests for verify_math_answer correctness and normalization."""

    def test_exact_integer(self):
        self.assertEqual(verify_math_answer("42", "42"), 1.0)
        self.assertEqual(verify_math_answer("42", "#### 42"), 1.0)

    def test_negative_integer(self):
        self.assertEqual(verify_math_answer("-17", "-17"), 1.0)

    def test_commas_and_currency(self):
        self.assertEqual(verify_math_answer("$1,250", "1250"), 1.0)
        self.assertEqual(verify_math_answer("1250", "$1,250"), 1.0)

    def test_fraction_and_decimal(self):
        self.assertEqual(verify_math_answer(r"\frac{1}{2}", "0.5"), 1.0)
        self.assertEqual(verify_math_answer("0.5", r"\frac{1}{2}"), 1.0)
        self.assertEqual(verify_math_answer("1/2", "0.5"), 1.0)
        self.assertEqual(verify_math_answer(r"\frac{3}{4}", "0.75"), 1.0)

    def test_unit_bearing_answer(self):
        # When math-verify is installed, it extracts numeric values from simple unit strings ("12 apples" -> 12).
        # In fallback mode without math-verify, non-normalized unit strings evaluate strictly to 0.0.
        expected = 1.0 if HAS_MATH_VERIFY else 0.0
        self.assertEqual(verify_math_answer("12 apples", "12"), expected)
        # Mismatched unit-bearing answer must always evaluate to 0.0
        self.assertEqual(verify_math_answer("15 apples", "12"), 0.0)

    def test_partial_number_mismatch(self):
        # Must not reward gold=10 when prediction is 12
        self.assertEqual(verify_math_answer("12", "10"), 0.0)

    def test_empty_or_none(self):
        self.assertEqual(verify_math_answer(None, "42"), 0.0)
        self.assertEqual(verify_math_answer("", "42"), 0.0)
        self.assertEqual(verify_math_answer("   ", "42"), 0.0)


class TestTRLRewardCallbacks(unittest.TestCase):
    """Tests for correctness_reward_fn and format_reward_fn."""

    def test_correctness_reward_fn_raw_strings(self):
        completions = [
            "<think>...</think><answer>42</answer>",
            "<think>...</think><answer>10</answer>",
            "No tags 42",
        ]
        rewards = correctness_reward_fn(
            prompts=[""] * 3,
            completions=completions,
            answer=["42", "42", "42"],
        )
        self.assertEqual(rewards, [1.0, 0.0, 0.0])

    def test_correctness_reward_fn_conversational_format(self):
        completions = [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "<think>2+2=4</think><answer>4</answer>"},
            ],
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "<think>2+2=5</think><answer>5</answer>"},
            ],
        ]
        rewards = correctness_reward_fn(
            prompts=[""] * 2,
            completions=completions,
            answer="4",
        )
        self.assertEqual(rewards, [1.0, 0.0])

    def test_format_reward_fn_valid(self):
        valid_completions = [
            "<think>2 + 2 = 4</think><answer>4</answer>",
            "  <think>\nMulti-step reasoning\n</think>\n<answer>42</answer>  ",
        ]
        rewards = format_reward_fn(prompts=[""] * 2, completions=valid_completions)
        self.assertEqual(rewards, [1.0, 1.0])

    def test_format_reward_fn_invalid(self):
        invalid_completions = [
            "<think></think><answer>4</answer>",  # empty think
            "<think>reasoning</think><answer></answer>",  # empty answer
            "The answer is 4.",  # no tags
            "<answer>4</answer>",  # missing think
            "<think>reasoning</think>",  # missing answer
            "<think>reasoning</think><answer>4</answer> extra text",  # trailing text
        ]
        rewards = format_reward_fn(prompts=[""] * len(invalid_completions), completions=invalid_completions)
        self.assertEqual(rewards, [0.0] * len(invalid_completions))


if __name__ == "__main__":
    unittest.main()
