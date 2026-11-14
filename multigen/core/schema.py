"""Pydantic data contracts for population lineage and experiment recipes.

Defines validated, immutable models representing:
  - TrainingRecipe: Specification of hyperparameter configurations and compute budgets.
  - CapabilityProfile: Standardized evaluation metrics on mathematical benchmarks.
  - ModelNode: Checkpoint nodes forming the population lineage graph.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TrainingRecipe(BaseModel):
    """Specification of training parameters and compute budget that produced a model node."""

    model_config = ConfigDict(extra="ignore")

    algorithm: Literal["grpo", "sft", "mixed"] = "grpo"
    base_model_name_or_path: str = "Qwen/Qwen3-0.6B"
    parent_adapter_path: Optional[str] = None

    # LoRA Hyperparameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # GRPO Optimization Hyperparameters
    group_size: int = 8
    learning_rate: float = 2e-5
    kl_beta: float = 0.04
    clip_epsilon: float = 0.2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_prompt_length: int = 512
    max_completion_length: int = 512
    temperature: float = 0.9

    # Reward Formulation
    correctness_reward_weight: float = 1.0
    format_reward_weight: float = 0.1

    # Prompting & Dataset
    num_shots: int = 0
    dataset_name: str = "openai/gsm8k"
    dataset_subset: str = "main"

    # Compute Budget & Checkpointing
    target_flop_budget: float = 1.0e16
    intermediate_eval_interval_flops: float = 2.0e15
    seed: int = 42

    # Hardware & Backend Execution
    use_vllm: Optional[bool] = None

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingRecipe":
        """Load recipe from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Flatten nested experiment YAML structures if present
        flat: Dict[str, Any] = {}
        if "model" in raw and isinstance(raw["model"], dict):
            flat["base_model_name_or_path"] = raw["model"].get(
                "name_or_path", "Qwen/Qwen3-0.6B"
            )
        if "peft" in raw and isinstance(raw["peft"], dict):
            peft = raw["peft"]
            if "rank" in peft:
                flat["lora_rank"] = peft["rank"]
            if "alpha" in peft:
                flat["lora_alpha"] = peft["alpha"]
            if "dropout" in peft:
                flat["lora_dropout"] = peft["dropout"]
            if "target_modules" in peft:
                flat["target_modules"] = peft["target_modules"]
        if "grpo" in raw and isinstance(raw["grpo"], dict):
            grpo = raw["grpo"]
            for k in [
                "group_size",
                "learning_rate",
                "kl_beta",
                "clip_epsilon",
                "weight_decay",
                "warmup_ratio",
                "max_prompt_length",
                "max_completion_length",
                "temperature",
                "use_vllm",
            ]:
                if k in grpo:
                    flat[k] = grpo[k]
        if "rewards" in raw and isinstance(raw["rewards"], dict):
            rew = raw["rewards"]
            if "correctness_weight" in rew:
                flat["correctness_reward_weight"] = rew["correctness_weight"]
            if "format_weight" in rew:
                flat["format_reward_weight"] = rew["format_weight"]
        if "budget" in raw and isinstance(raw["budget"], dict):
            bud = raw["budget"]
            if "target_flops" in bud:
                flat["target_flop_budget"] = float(bud["target_flops"])
            if "intermediate_eval_interval_flops" in bud:
                flat["intermediate_eval_interval_flops"] = float(
                    bud["intermediate_eval_interval_flops"]
                )
        if "dataset" in raw and isinstance(raw["dataset"], dict):
            ds = raw["dataset"]
            if "name" in ds:
                flat["dataset_name"] = ds["name"]
            if "subset" in ds:
                flat["dataset_subset"] = ds["subset"]
            if "num_shots" in ds:
                flat["num_shots"] = ds["num_shots"]
        if "seed" in raw:
            flat["seed"] = raw["seed"]

        # Merge any root-level keys
        for k, v in raw.items():
            if k in cls.model_fields and k not in flat:
                flat[k] = v

        return cls(**flat)

    def to_yaml(self, path: str) -> None:
        """Serialize recipe to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)


class CapabilityProfile(BaseModel):
    """Standardized capability scorecard for a model checkpoint."""

    model_config = ConfigDict(extra="ignore")

    benchmark: str = "gsm8k"
    split: Literal["train", "dev", "test"] = "test"
    num_shots: int = 0
    accuracy: float
    permissive_accuracy: float
    format_compliance_rate: float
    mean_completion_tokens: float = 0.0
    error_indices: List[int] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=_utc_now)
    wall_time_seconds: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelNode(BaseModel):
    """A checkpoint node in the population lineage DAG."""

    model_config = ConfigDict(extra="ignore")

    node_id: str
    parent_ids: List[str] = Field(default_factory=list)
    depth: int = 0
    recipe: TrainingRecipe
    capability_profile: Optional[CapabilityProfile] = None
    intermediate_evaluations: List[Dict[str, Any]] = Field(default_factory=list)
    adapter_path: Optional[str] = None
    cumulative_flops: float = 0.0
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
