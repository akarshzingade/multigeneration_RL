"""Unit tests for Pydantic core data contracts (multigen/core/schema.py)."""

from __future__ import annotations

import tempfile
import pytest
from multigen.core.schema import CapabilityProfile, ModelNode, TrainingRecipe


def test_training_recipe_defaults():
    recipe = TrainingRecipe()
    assert recipe.algorithm == "grpo"
    assert recipe.base_model_name_or_path == "Qwen/Qwen3-0.6B"
    assert recipe.lora_rank == 16
    assert recipe.lora_alpha == 32
    assert "q_proj" in recipe.target_modules
    assert "down_proj" in recipe.target_modules
    assert recipe.group_size == 8
    assert recipe.target_flop_budget == 1.0e16
    assert recipe.correctness_reward_weight == 1.0
    assert recipe.format_reward_weight == 0.1
    assert recipe.num_shots == 0


def test_training_recipe_yaml_roundtrip():
    recipe = TrainingRecipe(
        algorithm="grpo",
        base_model_name_or_path="Qwen/Qwen3-0.6B",
        lora_rank=32,
        lora_alpha=64,
        learning_rate=3e-5,
        target_flop_budget=5.0e15,
        seed=101,
    )
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        recipe.to_yaml(f.name)
        loaded = TrainingRecipe.from_yaml(f.name)

    assert loaded.lora_rank == 32
    assert loaded.lora_alpha == 64
    assert loaded.learning_rate == 3e-5
    assert loaded.target_flop_budget == 5.0e15
    assert loaded.seed == 101


def test_training_recipe_from_experiment_yaml():
    yaml_content = """
experiment_id: "test_exp"
model:
  name_or_path: "Qwen/Qwen3-0.6B"
peft:
  rank: 8
  alpha: 16
  target_modules: ["q_proj", "v_proj"]
grpo:
  group_size: 4
  learning_rate: 1.0e-5
  kl_beta: 0.02
rewards:
  correctness_weight: 1.0
  format_weight: 0.2
budget:
  target_flops: 2.0e15
seed: 99
"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        recipe = TrainingRecipe.from_yaml(f.name)

    assert recipe.base_model_name_or_path == "Qwen/Qwen3-0.6B"
    assert recipe.lora_rank == 8
    assert recipe.lora_alpha == 16
    assert recipe.target_modules == ["q_proj", "v_proj"]
    assert recipe.group_size == 4
    assert recipe.learning_rate == 1.0e-5
    assert recipe.kl_beta == 0.02
    assert recipe.format_reward_weight == 0.2
    assert recipe.target_flop_budget == 2.0e15
    assert recipe.seed == 99


def test_distill_training_recipe_from_experiment_yaml():
    yaml_content = """
experiment_id: "exp1_2_distill_baseline"
model:
  name_or_path: "Qwen/Qwen3-0.6B"
teacher:
  name_or_path: "Qwen/Qwen2.5-Math-7B-Instruct"
  k_samples: 4
  temperature: 0.7
  max_new_tokens: 512
  traces_path: "data/teacher_traces_gsm8k.jsonl"
peft:
  rank: 16
  alpha: 32
sft:
  learning_rate: 2.0e-4
  batch_size: 8
  gradient_accumulation_steps: 2
  response_template: "<think>"
budget:
  target_flops: 1.0e16
  accounting_mode: "amortized"
seed: 42
"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        recipe = TrainingRecipe.from_yaml(f.name)

    assert recipe.algorithm == "distill"
    assert recipe.teacher_model_name_or_path == "Qwen/Qwen2.5-Math-7B-Instruct"
    assert recipe.teacher_k_samples == 4
    assert recipe.teacher_temperature == 0.7
    assert recipe.teacher_max_tokens == 512
    assert recipe.teacher_trace_dataset_path == "data/teacher_traces_gsm8k.jsonl"
    assert recipe.learning_rate == 2.0e-4
    assert recipe.batch_size == 8
    assert recipe.gradient_accumulation_steps == 2
    assert recipe.response_template == "<think>"
    assert recipe.accounting_mode == "amortized"
    assert recipe.target_flop_budget == 1.0e16
    assert recipe.seed == 42


def test_capability_profile_creation():
    profile = CapabilityProfile(
        benchmark="gsm8k",
        split="test",
        accuracy=0.68,
        permissive_accuracy=0.72,
        format_compliance_rate=0.98,
        mean_completion_tokens=220.5,
        error_indices=[3, 14, 29],
        wall_time_seconds=124.0,
    )
    assert profile.accuracy == 0.68
    assert profile.format_compliance_rate == 0.98
    assert len(profile.error_indices) == 3
    data = profile.model_dump()
    assert data["benchmark"] == "gsm8k"


def test_model_node_creation_and_validation():
    recipe = TrainingRecipe(seed=42)
    node = ModelNode(
        node_id="exp1_1_grpo_seed42",
        parent_ids=["node_0"],
        depth=1,
        recipe=recipe,
        adapter_path="checkpoints/exp1_1_grpo_seed42/final_adapter",
        cumulative_flops=1.0e16,
    )
    assert node.node_id == "exp1_1_grpo_seed42"
    assert node.parent_ids == ["node_0"]
    assert node.depth == 1
    assert node.cumulative_flops == 1.0e16

    dump = node.model_dump()
    assert dump["node_id"] == "exp1_1_grpo_seed42"
    loaded = ModelNode.model_validate(dump)
    assert loaded.recipe.lora_rank == 16
