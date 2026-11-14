"""Unit tests for GSM8K evaluator, prompt formatting, and dataset splitting."""

import json
import unittest
from multigen.eval.evaluator import (
    BaselineEvalResult,
    format_gsm8k_prompt,
    split_gsm8k_dataset,
)


class TestEvaluatorPromptsAndSplits(unittest.TestCase):
    """Test prompt construction and deterministic dataset splitting."""

    def test_format_gsm8k_prompt_0shot(self):
        prompt = format_gsm8k_prompt("What is 10 + 5?", num_shots=0)
        self.assertIn("Solve the following math problem step by step.", prompt)
        self.assertIn("<think>...</think>", prompt)
        self.assertIn("<answer>...</answer>", prompt)
        self.assertIn("Problem: What is 10 + 5?", prompt)
        # 0-shot should not have Natalia exemplar
        self.assertNotIn("Natalia sold clips", prompt)

    def test_format_gsm8k_prompt_4shot(self):
        prompt = format_gsm8k_prompt("What is 10 + 5?", num_shots=4)
        self.assertIn("Natalia sold clips", prompt)
        self.assertIn("Weng earns $12 an hour", prompt)
        self.assertIn("Betty is saving money", prompt)
        self.assertIn("A deep-sea monster", prompt)
        self.assertTrue(prompt.endswith("Problem: What is 10 + 5?\n"))

    def test_split_gsm8k_dataset_reproducibility(self):
        # Create synthetic dataset with 7,473 train and 1,319 test
        mock_data = {
            "train": [
                {"question": f"Q{i}", "answer": f"Reasoning {i}\n#### {i}"}
                for i in range(7473)
            ],
            "test": [
                {"question": f"T{i}", "answer": f"Test Reasoning {i}\n#### {i * 10}"}
                for i in range(1319)
            ],
        }

        split1 = split_gsm8k_dataset(mock_data, dev_size=500, seed=42)
        split2 = split_gsm8k_dataset(mock_data, dev_size=500, seed=42)

        # Check split sizes
        self.assertEqual(len(split1["train"]), 6973)
        self.assertEqual(len(split1["dev"]), 500)
        self.assertEqual(len(split1["test"]), 1319)

        # Check disjointness of train and dev
        train_qs = {item["question"] for item in split1["train"]}
        dev_qs = {item["question"] for item in split1["dev"]}
        self.assertEqual(len(train_qs.intersection(dev_qs)), 0)

        # Check reproducibility with same seed
        self.assertEqual(
            [item["question"] for item in split1["dev"]],
            [item["question"] for item in split2["dev"]],
        )

        # Check gold extraction
        self.assertEqual(split1["test"][0]["gold_answer"], "0")
        self.assertEqual(split1["test"][5]["gold_answer"], "50")

    def test_baseline_eval_result_serialization(self):
        res = BaselineEvalResult(
            model_name="Qwen/Qwen3-0.6B",
            split="test",
            num_shots=0,
            total_examples=100,
            accuracy=0.45,
            permissive_accuracy=0.52,
            format_compliance_rate=0.88,
            error_indices=[1, 4, 9],
            mean_completion_tokens=150.5,
            wall_time_seconds=25.2,
            parser_exceptions=0,
        )

        d = res.to_dict()
        self.assertEqual(d["model_name"], "Qwen/Qwen3-0.6B")
        self.assertEqual(d["accuracy"], 0.45)
        self.assertEqual(d["permissive_accuracy"], 0.52)
        self.assertEqual(d["error_indices"], [1, 4, 9])

        raw_json = res.to_json()
        loaded = json.loads(raw_json)
        self.assertEqual(loaded["model_name"], "Qwen/Qwen3-0.6B")

    def test_evaluator_evaluate_signature_and_fallback(self):
        from unittest.mock import MagicMock, patch
        from multigen.eval.evaluator import GSM8KEvaluator

        with patch.object(GSM8KEvaluator, "_init_backend", return_value=None):
            evaluator = GSM8KEvaluator(
                model_name_or_path="Qwen/Qwen3-0.6B",
                backend="transformers",
            )
            evaluator.generate_batch = MagicMock(return_value=(
                ["<think>10+5=15</think><answer>15</answer>"],
                [10],
            ))

            test_data = [{"question": "What is 10 + 5?", "gold_answer": "15"}]

            # 1. Test evaluate with explicit dataset and split_name
            res1 = evaluator.evaluate(dataset=test_data, split_name="test")
            self.assertEqual(res1.accuracy, 1.0)
            self.assertEqual(res1.format_compliance_rate, 1.0)

            # 2. Test evaluate with 'split' keyword alias
            res2 = evaluator.evaluate(dataset=test_data, split="dev")
            self.assertEqual(res2.split, "dev")
            self.assertEqual(res2.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
