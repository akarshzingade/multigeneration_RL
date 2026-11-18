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

    algorithm: Literal["grpo", "sft", "distill", "mixed"] = "grpo"
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

    # Distillation & Teacher Hyperparameters (EXP-1.2)
    teacher_model_name_or_path: Optional[str] = "Qwen/Qwen2.5-Math-7B-Instruct"
    teacher_k_samples: int = 4
    teacher_temperature: float = 0.7
    teacher_max_tokens: int = 512
    teacher_trace_dataset_path: Optional[str] = "data/teacher_traces_gsm8k.jsonl"
    teacher_budget_ratio: float = 0.20

    # Mixed Learning Hyperparameters (EXP-1.3)
    mode: Literal["interleaved_sequential", "teacher_regularized", "micro_interleaved"] = "interleaved_sequential"
    num_cycles: int = 4
    ref_policy_source: Literal["sft_warmup", "base_model", "ema"] = "sft_warmup"
    ema_decay: float = 0.99
    optimizer_transition: Literal["reset_and_warmup", "carryover_rescale"] = "reset_and_warmup"
    distill_budget_ratio: float = 0.25

    # Resumption & Fault Tolerance (EXP-1.3)
    resumption_enabled: bool = True
    auto_resume: bool = True
    journal_filename: str = "recovery_state.json"
    enable_step_checkpoints: bool = True
    checkpoint_interval_steps: int = 50
    enable_milestone_checkpoints: bool = True
    enable_signal_trapping: bool = True
    save_on_signals: List[str] = Field(default_factory=lambda: ["SIGINT", "SIGTERM"])

    # SFT Optimization Hyperparameters (EXP-1.2)
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_seq_length: int = 1024
    response_template: str = "<think>"

    # Reward Formulation (GRPO)
    correctness_reward_weight: float = 1.0
    format_reward_weight: float = 0.1

    # Prompting & Dataset
    num_shots: int = 0
    dataset_name: str = "openai/gsm8k"
    dataset_subset: str = "main"

    # Compute Budget & Checkpointing (Supports Unconstrained Training)
    target_flop_budget: Optional[float] = 1.0e16
    accounting_mode: Literal["amortized", "strict_split"] = "amortized"
    intermediate_eval_interval_flops: float = 2.0e15
    seed: int = 42

    # Termination & Stopping Criteria (Without FLOP constraint)
    termination_criterion: Literal["flops", "plateau", "epochs", "steps"] = "flops"
    num_train_epochs: Optional[float] = None
    max_steps: Optional[int] = None

    # Plateau Detection & Early Stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    early_stopping_metric: str = "eval_loss"
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_total_limit: int = 2
    load_best_model_at_end: bool = True

    # Mixed Learning Unconstrained Stage Transition (EXP-1.3)
    stage_transition_criterion: Literal["plateau", "epochs", "steps", "flops"] = "plateau"
    sft_epochs: Optional[float] = None
    sft_max_steps: Optional[int] = None
    sft_patience: int = 3
    grpo_epochs: Optional[float] = None
    grpo_max_steps: Optional[int] = None
    grpo_patience: int = 3

    # Hardware & Backend Execution
    use_vllm: Optional[bool] = None

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingRecipe":
        """Load recipe from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Flatten nested experiment YAML structures if present
        flat: Dict[str, Any] = {}
        if "algorithm" in raw:
            flat["algorithm"] = raw["algorithm"]
        elif "experiment_id" in raw and "distill" in str(raw["experiment_id"]):
            flat["algorithm"] = "distill"

        if "model" in raw and isinstance(raw["model"], dict):
            flat["base_model_name_or_path"] = raw["model"].get(
                "name_or_path", "Qwen/Qwen3-0.6B"
            )
        if "teacher" in raw and isinstance(raw["teacher"], dict):
            t = raw["teacher"]
            if "name_or_path" in t:
                flat["teacher_model_name_or_path"] = t["name_or_path"]
            if "k_samples" in t:
                flat["teacher_k_samples"] = int(t["k_samples"])
            if "temperature" in t:
                flat["teacher_temperature"] = float(t["temperature"])
            if "max_new_tokens" in t:
                flat["teacher_max_tokens"] = int(t["max_new_tokens"])
            elif "max_tokens" in t:
                flat["teacher_max_tokens"] = int(t["max_tokens"])
            if "traces_path" in t:
                flat["teacher_trace_dataset_path"] = t["traces_path"]
            elif "dataset_path" in t:
                flat["teacher_trace_dataset_path"] = t["dataset_path"]

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

        if "sft" in raw and isinstance(raw["sft"], dict):
            sft = raw["sft"]
            for k in [
                "learning_rate",
                "batch_size",
                "gradient_accumulation_steps",
                "weight_decay",
                "warmup_ratio",
                "max_seq_length",
                "response_template",
                "use_vllm",
            ]:
                if k in sft:
                    flat[k] = sft[k]

        if "sft_phase" in raw and isinstance(raw["sft_phase"], dict):
            sft_p = raw["sft_phase"]
            if "distill_budget_ratio" in sft_p:
                flat["distill_budget_ratio"] = float(sft_p["distill_budget_ratio"])
            for k in [
                "learning_rate",
                "batch_size",
                "gradient_accumulation_steps",
                "weight_decay",
                "warmup_ratio",
                "max_seq_length",
                "response_template",
                "use_vllm",
            ]:
                if k in sft_p:
                    flat[k] = sft_p[k]

        if "grpo_phase" in raw and isinstance(raw["grpo_phase"], dict):
            grpo_p = raw["grpo_phase"]
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
                if k in grpo_p:
                    flat[k] = grpo_p[k]

        if "resumption" in raw and isinstance(raw["resumption"], dict):
            res = raw["resumption"]
            if "enabled" in res:
                flat["resumption_enabled"] = bool(res["enabled"])
            if "auto_resume" in res:
                flat["auto_resume"] = bool(res["auto_resume"])
            if "journal_filename" in res:
                flat["journal_filename"] = str(res["journal_filename"])
            if "enable_step_checkpoints" in res:
                flat["enable_step_checkpoints"] = bool(res["enable_step_checkpoints"])
            if "checkpoint_interval_steps" in res:
                flat["checkpoint_interval_steps"] = int(res["checkpoint_interval_steps"])
            if "enable_milestone_checkpoints" in res:
                flat["enable_milestone_checkpoints"] = bool(res["enable_milestone_checkpoints"])
            if "enable_signal_trapping" in res:
                flat["enable_signal_trapping"] = bool(res["enable_signal_trapping"])
            if "save_on_signals" in res:
                flat["save_on_signals"] = list(res["save_on_signals"])

        if "rewards" in raw and isinstance(raw["rewards"], dict):
            rew = raw["rewards"]
            if "correctness_weight" in rew:
                flat["correctness_reward_weight"] = rew["correctness_weight"]
            if "format_weight" in rew:
                flat["format_reward_weight"] = rew["format_weight"]

        if "budget" in raw and isinstance(raw["budget"], dict):
            bud = raw["budget"]
            if "target_flops" in bud:
                val = bud["target_flops"]
                if val is None or str(val).lower() in ("null", "none"):
                    flat["target_flop_budget"] = None
                else:
                    flat["target_flop_budget"] = float(val)
            if "accounting_mode" in bud:
                flat["accounting_mode"] = str(bud["accounting_mode"])
            if "teacher_budget_ratio" in bud:
                flat["teacher_budget_ratio"] = float(bud["teacher_budget_ratio"])
            if "intermediate_eval_interval_flops" in bud:
                val = bud["intermediate_eval_interval_flops"]
                if val is None or str(val).lower() in ("null", "none"):
                    flat["intermediate_eval_interval_flops"] = 0.0
                else:
                    flat["intermediate_eval_interval_flops"] = float(val)

        if "termination" in raw and isinstance(raw["termination"], dict):
            term = raw["termination"]
            if "criterion" in term:
                flat["termination_criterion"] = term["criterion"]
            if "num_train_epochs" in term:
                flat["num_train_epochs"] = float(term["num_train_epochs"]) if term["num_train_epochs"] is not None else None
            elif "epochs" in term:
                flat["num_train_epochs"] = float(term["epochs"]) if term["epochs"] is not None else None
            if "max_steps" in term:
                flat["max_steps"] = int(term["max_steps"]) if term["max_steps"] is not None else None
            if "early_stopping_patience" in term:
                flat["early_stopping_patience"] = int(term["early_stopping_patience"])
            elif "patience" in term:
                flat["early_stopping_patience"] = int(term["patience"])
            if "early_stopping_threshold" in term:
                flat["early_stopping_threshold"] = float(term["early_stopping_threshold"])
            elif "threshold" in term:
                flat["early_stopping_threshold"] = float(term["threshold"])
            if "early_stopping_metric" in term:
                flat["early_stopping_metric"] = str(term["early_stopping_metric"])
            if "eval_steps" in term:
                flat["eval_steps"] = int(term["eval_steps"])
            if "eval_strategy" in term:
                flat["eval_strategy"] = str(term["eval_strategy"])

        if "stage_transition" in raw and isinstance(raw["stage_transition"], dict):
            st = raw["stage_transition"]
            if "criterion" in st:
                flat["stage_transition_criterion"] = st["criterion"]
            if "sft_epochs" in st:
                flat["sft_epochs"] = float(st["sft_epochs"]) if st["sft_epochs"] is not None else None
            if "sft_max_steps" in st:
                flat["sft_max_steps"] = int(st["sft_max_steps"]) if st["sft_max_steps"] is not None else None
            if "sft_patience" in st:
                flat["sft_patience"] = int(st["sft_patience"])
            if "grpo_epochs" in st:
                flat["grpo_epochs"] = float(st["grpo_epochs"]) if st["grpo_epochs"] is not None else None
            if "grpo_max_steps" in st:
                flat["grpo_max_steps"] = int(st["grpo_max_steps"]) if st["grpo_max_steps"] is not None else None
            if "grpo_patience" in st:
                flat["grpo_patience"] = int(st["grpo_patience"])

        # Auto-switch termination criterion to plateau if target_flop_budget is None
        # and criterion was not explicitly set to something else
        if flat.get("target_flop_budget") is None and "termination_criterion" not in flat:
            flat["termination_criterion"] = "plateau"

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
