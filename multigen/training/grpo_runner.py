"""GRPO training runner orchestrating TRL GRPOTrainer with PEFT LoRA and FLOP budgeting.

Implements Experiment 1.1 specification:
  - Base model: Qwen/Qwen3-0.6B (or configured base model)
  - Parameter-efficient fine-tuning: LoRA on all linear projections
  - Reward callbacks: Deterministic mathematical verification + format adherence
  - FLOP budget accounting: Exact hardware-invariant step debiting and early halting
  - Checkpoint management: Intermediate FLOP-interval and final adapter persistence
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional
import torch

from multigen.accounting.flops import extract_param_counts
from multigen.accounting.tracker import FLOPAccumulator, FLOPBudgetCallback
from multigen.core.schema import TrainingRecipe
from multigen.eval.rewards import correctness_reward_fn, format_reward_fn

logger = logging.getLogger(__name__)

# Safe imports for optional heavy libraries
try:
    from peft import LoraConfig, PeftModel, get_peft_model
    HAS_PEFT = True
except (ImportError, NameError, Exception) as e:
    HAS_PEFT = False
    LoraConfig = Any  # type: ignore[assignment,misc]
    PeftModel = Any  # type: ignore[assignment,misc]
    get_peft_model = Any  # type: ignore[assignment,misc]
    logger.debug("peft import failed: %s", e)

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )
    HAS_TRANSFORMERS = True
except (ImportError, NameError, Exception) as e:
    HAS_TRANSFORMERS = False
    AutoModelForCausalLM = Any  # type: ignore[assignment,misc]
    AutoTokenizer = Any  # type: ignore[assignment,misc]
    logger.debug("transformers import failed: %s", e)

try:
    from trl import GRPOConfig, GRPOTrainer
    HAS_TRL = True
except (ImportError, NameError, Exception) as e:
    HAS_TRL = False
    GRPOConfig = Any  # type: ignore[assignment,misc]
    GRPOTrainer = Any  # type: ignore[assignment,misc]
    logger.debug("trl import failed: %s", e)


class GRPOPlateauCallback(TrainerCallback):
    """Monitors reward or loss during GRPO training and stops when improvement plateaus."""

    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.001,
        metric_name: str = "reward",
        mode: str = "max",
        window_size: int = 5,
    ) -> None:
        self.patience = patience
        self.threshold = threshold
        self.metric_name = metric_name
        self.mode = mode
        self.window_size = window_size
        self.history: List[float] = []
        self.smoothed_history: List[float] = []
        self.best_value: Optional[float] = None
        self.patience_counter: int = 0
        self.is_plateaued: bool = False

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if logs is None or not isinstance(logs, dict):
            return

        target_val = None
        if self.metric_name in logs:
            target_val = logs[self.metric_name]
        elif f"eval_{self.metric_name}" in logs:
            target_val = logs[f"eval_{self.metric_name}"]
        elif self.metric_name == "reward":
            for k in logs:
                if "reward" in k.lower():
                    target_val = logs[k]
                    break

        if target_val is None or not isinstance(target_val, (int, float)):
            return

        val = float(target_val)
        self.history.append(val)

        recent = self.history[-self.window_size :]
        smoothed = sum(recent) / len(recent)
        self.smoothed_history.append(smoothed)

        if self.best_value is None:
            self.best_value = smoothed
            self.patience_counter = 0
            return

        if self.mode == "max":
            improved = (smoothed - self.best_value) > self.threshold
        else:
            improved = (self.best_value - smoothed) > self.threshold

        if improved:
            self.best_value = smoothed
            self.patience_counter = 0
            logger.info(
                f"[GRPOPlateauCallback] Step {state.global_step}: Metric improved to {smoothed:.4f}. Resetting patience counter."
            )
        else:
            self.patience_counter += 1
            logger.info(
                f"[GRPOPlateauCallback] Step {state.global_step}: Metric plateaued at {smoothed:.4f} (best: {self.best_value:.4f}). "
                f"Patience: {self.patience_counter}/{self.patience}"
            )
            if self.patience_counter >= self.patience:
                control.should_training_stop = True
                control.should_save = True
                self.is_plateaued = True
                logger.info(
                    f"[GRPOPlateauCallback] Early stopping triggered! Plateau detected on '{self.metric_name}' after {self.patience} checks without improvement."
                )


class GRPOBudgetTrackingCallback(FLOPBudgetCallback):
    """Callback that monitors FLOP budget and triggers intermediate checkpoints."""

    def __init__(
        self,
        accumulator: FLOPAccumulator,
        intermediate_interval_flops: float = 2.0e15,
        save_checkpoint_fn: Optional[Callable[[float, int], str]] = None,
    ) -> None:
        super().__init__(accumulator)
        self.intermediate_interval_flops = intermediate_interval_flops
        self.save_checkpoint_fn = save_checkpoint_fn
        self.last_saved_milestone: float = 0.0
        self.saved_milestones: List[Dict[str, Any]] = []

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        super().on_step_end(args, state, control, **kwargs)

        # Check for intermediate evaluation/checkpoint milestones
        current_flops = self.accumulator.cumulative_flops
        if self.intermediate_interval_flops > 0:
            next_milestone = self.last_saved_milestone + self.intermediate_interval_flops
            if current_flops >= next_milestone and next_milestone <= (self.accumulator.target_budget or float("inf")):
                self.last_saved_milestone = next_milestone
                if self.save_checkpoint_fn is not None:
                    try:
                        saved_path = self.save_checkpoint_fn(next_milestone, state.global_step)
                        self.saved_milestones.append(
                            {
                                "milestone_flops": next_milestone,
                                "current_flops": current_flops,
                                "step": state.global_step,
                                "checkpoint_path": saved_path,
                            }
                        )
                        logger.info(
                            "[GRPOBudgetCallback] Saved intermediate checkpoint at %.2e FLOPs (step %d) -> %s",
                            next_milestone,
                            state.global_step,
                            saved_path,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to save intermediate checkpoint at %.2e FLOPs: %s",
                            next_milestone,
                            e,
                        )
        return control


class GRPORunner:
    """Orchestrates GRPO training with strict compute budget enforcement."""

    def __init__(
        self,
        recipe: TrainingRecipe,
        output_dir: str,
        accumulator: Optional[FLOPAccumulator] = None,
    ) -> None:
        self.recipe = recipe
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if accumulator is not None:
            self.accumulator = accumulator
        else:
            self.accumulator = FLOPAccumulator(target_budget=recipe.target_flop_budget)
        self.intermediate_checkpoints: List[Dict[str, Any]] = []

    def _determine_vllm_usage(self) -> bool:
        """Determine whether vLLM rollout acceleration should be enabled."""
        if self.recipe.use_vllm is False:
            return False
        if not torch.cuda.is_available():
            if self.recipe.use_vllm is True:
                logger.warning(
                    "use_vllm=True was requested, but CUDA is unavailable. "
                    "Falling back to PyTorch generation."
                )
            return False
        return True

    def build_reward_functions(self, tokenizer: Any, n_base: int, n_lora: int) -> List[Callable]:
        """Construct reward functions with automatic FLOP accounting instrumentation.

        Each batch of completions evaluated by the verifiers is measured for token length
        and debited to the cumulative FLOP accumulator under the hardware-invariant
        GRPO accounting formula:
            FLOPs_seq = (8 * N_base + 6 * N_lora) * (T_prompt + T_gen)
        """
        # Precompute per-token FLOP coefficient
        flop_per_token = (8 * n_base) + (6 * n_lora)

        def correctness_wrapper(
            prompts: List[str], completions: Any, answer: Any = None, **kwargs: Any
        ) -> List[float]:
            # Extract plain text strings
            comp_texts: List[str] = []
            if isinstance(completions, list):
                for c in completions:
                    if isinstance(c, str):
                        comp_texts.append(c)
                    elif isinstance(c, list) and len(c) > 0 and isinstance(c[-1], dict):
                        comp_texts.append(c[-1].get("content", ""))
                    else:
                        comp_texts.append(str(c))
            else:
                comp_texts = [str(completions)]

            # Tokenize or estimate token counts for exact accounting
            total_batch_tokens = 0
            for p, c in zip(prompts, comp_texts):
                p_tokens = len(tokenizer.encode(p, add_special_tokens=False)) if tokenizer else max(1, len(p) // 4)
                c_tokens = len(tokenizer.encode(c, add_special_tokens=False)) if tokenizer else max(1, len(c) // 4)
                total_batch_tokens += (p_tokens + c_tokens)

            # Debit to accumulator
            batch_flops = total_batch_tokens * flop_per_token
            self.accumulator.debit(
                batch_flops,
                category="train",
                tokens=total_batch_tokens,
            )

            # Compute standard correctness reward
            raw_correctness = correctness_reward_fn(
                prompts, completions, answer=answer, **kwargs
            )
            return [float(r) * self.recipe.correctness_reward_weight for r in raw_correctness]

        def format_wrapper(prompts: List[str], completions: Any, **kwargs: Any) -> List[float]:
            raw_format = format_reward_fn(prompts, completions, **kwargs)
            return [float(r) * self.recipe.format_reward_weight for r in raw_format]

        return [correctness_wrapper, format_wrapper]

    def setup_model_and_tokenizer(self) -> Any:
        """Initialize base model, tokenizer, and LoRA adapter."""
        if not HAS_TRANSFORMERS or not HAS_PEFT:
            raise ImportError("transformers and peft are required to run GRPORunner.")

        tokenizer = AutoTokenizer.from_pretrained(
            self.recipe.base_model_name_or_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            self.recipe.base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Attach LoRA adapter
        if self.recipe.parent_adapter_path:
            logger.info("Loading parent LoRA adapter from %s", self.recipe.parent_adapter_path)
            model = PeftModel.from_pretrained(
                model,
                self.recipe.parent_adapter_path,
                is_trainable=True,
            )
        else:
            peft_config = LoraConfig(
                r=self.recipe.lora_rank,
                lora_alpha=self.recipe.lora_alpha,
                lora_dropout=self.recipe.lora_dropout,
                target_modules=self.recipe.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

        return model, tokenizer

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute GRPO training until FLOP budget is exhausted."""
        if not HAS_TRL:
            raise ImportError("trl is required to run GRPORunner.")

        model, tokenizer = self.setup_model_and_tokenizer()
        param_counts = extract_param_counts(model)
        n_base = param_counts.frozen_params or param_counts.total_params
        n_lora = param_counts.trainable_params

        use_vllm = self._determine_vllm_usage()

        # Build instrumentation
        reward_funcs = self.build_reward_functions(tokenizer, n_base, n_lora)

        def save_checkpoint(milestone_flops: float, step: int) -> str:
            cp_dir = os.path.join(
                self.output_dir, f"intermediate_flops_{int(milestone_flops):.0e}"
            )
            os.makedirs(cp_dir, exist_ok=True)
            model.save_pretrained(cp_dir)
            tokenizer.save_pretrained(cp_dir)
            return cp_dir

        callback = GRPOBudgetTrackingCallback(
            accumulator=self.accumulator,
            intermediate_interval_flops=self.recipe.intermediate_eval_interval_flops,
            save_checkpoint_fn=save_checkpoint,
        )
        callbacks = [callback]

        plateau_callback: Optional[GRPOPlateauCallback] = None
        if self.recipe.termination_criterion == "plateau":
            plateau_callback = GRPOPlateauCallback(
                patience=self.recipe.early_stopping_patience,
                threshold=self.recipe.early_stopping_threshold,
                metric_name="reward",
                mode="max",
            )
            callbacks.append(plateau_callback)

        import inspect

        candidate_grpo_args = {
            "output_dir": self.output_dir,
            "learning_rate": self.recipe.learning_rate,
            "num_generations": self.recipe.group_size,
            "max_completion_length": self.recipe.max_completion_length,
            "temperature": self.recipe.temperature,
            "beta": self.recipe.kl_beta,
            "epsilon": self.recipe.clip_epsilon,
            "weight_decay": self.recipe.weight_decay,
            "warmup_ratio": self.recipe.warmup_ratio,
            "logging_steps": 10,
            "save_strategy": "no",  # Checkpoints are managed by FLOP milestones or final save
            "per_device_train_batch_size": self.recipe.group_size,
            "gradient_accumulation_steps": 1,
            "generation_batch_size": self.recipe.group_size,
            "use_vllm": use_vllm,
            "bf16": torch.cuda.is_available(),
            "fp16": False,
            "seed": self.recipe.seed,
        }

        # Handle termination criteria: epochs, steps, or plateau
        if self.recipe.num_train_epochs is not None:
            candidate_grpo_args["num_train_epochs"] = self.recipe.num_train_epochs
        elif self.recipe.termination_criterion == "plateau":
            candidate_grpo_args["num_train_epochs"] = 5.0  # Generous upper bound; halted by plateau callback
        elif self.recipe.target_flop_budget is None:
            candidate_grpo_args["num_train_epochs"] = 1.0  # Default 1 epoch if unconstrained and no plateau

        if self.recipe.max_steps is not None and self.recipe.max_steps > 0:
            candidate_grpo_args["max_steps"] = self.recipe.max_steps

        # Filter strictly to parameters accepted by the installed TRL GRPOConfig version
        sig = inspect.signature(GRPOConfig.__init__)
        valid_grpo_args = {
            k: v for k, v in candidate_grpo_args.items() if k in sig.parameters
        }
        training_args = GRPOConfig(**valid_grpo_args)

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

        budget_disp = (
            f"{self.recipe.target_flop_budget:.2e} FLOPs"
            if self.recipe.target_flop_budget
            else "None (unconstrained)"
        )
        logger.info(
            "Starting GRPO training: model=%s, group_size=%d, target_budget=%s, termination_criterion=%s",
            self.recipe.base_model_name_or_path,
            self.recipe.group_size,
            budget_disp,
            self.recipe.termination_criterion,
        )

        trainer.train()

        # Save final adapter
        final_adapter_dir = os.path.join(self.output_dir, "final_adapter")
        os.makedirs(final_adapter_dir, exist_ok=True)
        model.save_pretrained(final_adapter_dir)
        tokenizer.save_pretrained(final_adapter_dir)

        self.intermediate_checkpoints = callback.saved_milestones
        total_steps = trainer.state.global_step if hasattr(trainer, "state") else 0
        is_plateaued = getattr(plateau_callback, "is_plateaued", False) if plateau_callback else False

        # Free trainer memory and flush CUDA cache before evaluation starts
        del trainer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "adapter_path": final_adapter_dir,
            "cumulative_flops": self.accumulator.cumulative_flops,
            "total_steps": total_steps,
            "is_exhausted": self.accumulator.is_exhausted,
            "is_plateaued": is_plateaued,
            "intermediate_checkpoints": self.intermediate_checkpoints,
            "summary": self.accumulator.get_summary(),
        }
