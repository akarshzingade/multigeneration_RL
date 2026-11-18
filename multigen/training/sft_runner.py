"""SFT training runner orchestrating TRL SFTTrainer with PEFT LoRA and FLOP budgeting.

Implements Experiment 1.2 specification:
  - Base model: Qwen/Qwen3-0.6B (or configured base model)
  - PEFT: LoRA on all linear projections (matching EXP-1.1 architecture)
  - Completion-only loss masking: DataCollatorForCompletionOnlyLM targeting "<think>"
  - FLOP budget accounting: Exact hardware-invariant step debiting (4*N_base + 6*N_lora)*T
  - Early graceful halting & intermediate checkpoint persistence at FLOP intervals
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import torch

from multigen.accounting.flops import extract_param_counts
from multigen.accounting.tracker import FLOPAccumulator, FLOPBudgetCallback
from multigen.core.schema import TrainingRecipe

logger = logging.getLogger(__name__)

# Safe imports for heavy optional dependencies
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
        EarlyStoppingCallback,
        Trainer,
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
    EarlyStoppingCallback = Any  # type: ignore[assignment,misc]
    Trainer = Any  # type: ignore[assignment,misc]
    TrainingArguments = Any  # type: ignore[assignment,misc]
    TrainerCallback = Any  # type: ignore[assignment,misc]
    logger.debug("transformers import failed: %s", e)

try:
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
    HAS_TRL = True
except (ImportError, NameError, Exception) as e:
    HAS_TRL = False
    DataCollatorForCompletionOnlyLM = None  # type: ignore[assignment,misc]
    SFTConfig = None  # type: ignore[assignment,misc]
    SFTTrainer = None  # type: ignore[assignment,misc]
    logger.debug("trl import failed: %s", e)


class FallbackCompletionOnlyDataCollator:
    """Fallback data collator for completion-only loss masking when TRL is unavailable.

    Masks prompt tokens (sets labels to -100) and computes cross-entropy loss exclusively on
    the completion response tokens starting from `response_template`.
    """

    def __init__(self, response_template: str, tokenizer: Any, max_length: int = 1024) -> None:
        self.response_template = response_template
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [f["text"] for f in features]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()

        # Find response_template in each sequence and mask preceding prompt tokens with -100
        for i, input_id_seq in enumerate(batch["input_ids"]):
            seq_list = input_id_seq.tolist()
            match_idx = -1
            m = len(self.response_token_ids)
            for j in range(len(seq_list) - m + 1):
                if seq_list[j : j + m] == self.response_token_ids:
                    match_idx = j
                    break
            if match_idx != -1:
                labels[i, :match_idx] = -100
            if self.tokenizer.pad_token_id is not None:
                labels[i, input_id_seq == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch


class SFTBudgetTrackingCallback(FLOPBudgetCallback):
    """Callback that monitors FLOP budget and triggers intermediate checkpoints for SFT."""

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

        current_flops = self.accumulator.cumulative_flops
        if self.intermediate_interval_flops > 0:
            next_milestone = self.last_saved_milestone + self.intermediate_interval_flops
            if current_flops >= next_milestone and next_milestone <= (
                self.accumulator.target_budget or float("inf")
            ):
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
                            "[SFTBudgetCallback] Saved intermediate checkpoint at %.2e FLOPs (step %d) -> %s",
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


BaseTrainerClass = SFTTrainer if (HAS_TRL and SFTTrainer is not None) else Trainer


if HAS_TRANSFORMERS:
    class BudgetedSFTTrainer(BaseTrainerClass):
        """Trainer subclass that automatically debits FLOPs per training step."""

        def __init__(
            self,
            *args: Any,
            accumulator: Optional[FLOPAccumulator] = None,
            n_base: int = 0,
            n_lora: int = 0,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.accumulator = accumulator
            self.n_base = n_base
            self.n_lora = n_lora
            # Formula: 4 * N_base * T + 6 * N_lora * T
            self.flops_per_token = (4.0 * float(n_base)) + (6.0 * float(n_lora))

        def training_step(
            self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
        ) -> torch.Tensor:
            loss = super().training_step(model, inputs)
            if self.accumulator is not None and "input_ids" in inputs:
                input_ids = inputs["input_ids"]
                if hasattr(input_ids, "numel"):
                    tokens = int(input_ids.numel())
                    step_flops = float(tokens) * self.flops_per_token
                    self.accumulator.debit(step_flops, category="train", tokens=tokens)
            return loss
else:
    class BudgetedSFTTrainer:  # type: ignore[no-redef]
        pass


class SFTRunner:
    """Orchestrates LoRA Supervised Fine-Tuning with strict compute budget enforcement."""

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

    def setup_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Load and configure student model and tokenizer with LoRA adapters."""
        if not HAS_TRANSFORMERS or not HAS_PEFT:
            raise ImportError("transformers and peft are required to run SFTRunner.")

        tokenizer = AutoTokenizer.from_pretrained(
            self.recipe.base_model_name_or_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model_kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.recipe.base_model_name_or_path,
                dtype=torch_dtype,
                **model_kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                self.recipe.base_model_name_or_path,
                torch_dtype=torch_dtype,
                **model_kwargs,
            )

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

    def format_prompt(self, question: str) -> str:
        """Format input question according to standard zero-shot protocol."""
        return f"Question: {question.strip()}\n\n"

    def prepare_dataset(self, raw_dataset: Any) -> Any:
        """Standardize training dataset into prompt + completion text pairs."""
        if not raw_dataset:
            return None

        from datasets import Dataset

        if isinstance(raw_dataset, Dataset):
            if "text" in raw_dataset.column_names:
                return raw_dataset

        formatted_rows = []
        for row in raw_dataset:
            q = row.get("question") or row.get("prompt") or ""
            r = row.get("response") or row.get("completion") or row.get("rationale") or ""
            if not r and "text" in row:
                full_text = row["text"]
            else:
                full_text = f"{self.format_prompt(q)}{r.strip()}"

            formatted_rows.append({"text": full_text})

        return Dataset.from_list(formatted_rows)

    def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute SFT training until the FLOP budget is hit."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required to run SFTRunner.")

        model, tokenizer = self.setup_model_and_tokenizer()
        param_counts = extract_param_counts(model)
        n_base = param_counts.frozen_params or param_counts.total_params
        n_lora = param_counts.trainable_params

        # Build completion-only data collator
        response_template = self.recipe.response_template or "<think>"
        if HAS_TRL and DataCollatorForCompletionOnlyLM is not None:
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=tokenizer,
            )
        else:
            data_collator = FallbackCompletionOnlyDataCollator(
                response_template=response_template,
                tokenizer=tokenizer,
                max_length=self.recipe.max_seq_length,
            )

        def save_checkpoint(milestone_flops: float, step: int) -> str:
            cp_dir = os.path.join(
                self.output_dir, f"intermediate_flops_{int(milestone_flops):.0e}"
            )
            os.makedirs(cp_dir, exist_ok=True)
            model.save_pretrained(cp_dir)
            tokenizer.save_pretrained(cp_dir)
            return cp_dir

        callback = SFTBudgetTrackingCallback(
            accumulator=self.accumulator,
            intermediate_interval_flops=self.recipe.intermediate_eval_interval_flops,
            save_checkpoint_fn=save_checkpoint,
        )
        callbacks = [callback]

        dataset = self.prepare_dataset(train_dataset)
        eval_data = self.prepare_dataset(eval_dataset) if eval_dataset else None

        # Configure SFT arguments
        candidate_args = {
            "output_dir": self.output_dir,
            "learning_rate": self.recipe.learning_rate,
            "per_device_train_batch_size": self.recipe.batch_size,
            "gradient_accumulation_steps": self.recipe.gradient_accumulation_steps,
            "weight_decay": self.recipe.weight_decay,
            "warmup_ratio": self.recipe.warmup_ratio,
            "max_seq_length": self.recipe.max_seq_length,
            "gradient_checkpointing": True,
            "dataset_text_field": "text",
            "logging_steps": 10,
            "bf16": torch.cuda.is_available(),
            "fp16": False,
            "seed": self.recipe.seed,
            "report_to": "none",
            "remove_unused_columns": False,
        }

        # Plateau detection / early stopping setup
        if self.recipe.termination_criterion == "plateau":
            if eval_data is not None:
                candidate_args["evaluation_strategy"] = self.recipe.eval_strategy
                candidate_args["eval_strategy"] = self.recipe.eval_strategy
                candidate_args["eval_steps"] = self.recipe.eval_steps
                candidate_args["save_strategy"] = self.recipe.eval_strategy
                candidate_args["save_steps"] = self.recipe.eval_steps
                candidate_args["save_total_limit"] = self.recipe.save_total_limit
                candidate_args["metric_for_best_model"] = "loss"
                candidate_args["greater_is_better"] = False
                candidate_args["load_best_model_at_end"] = self.recipe.load_best_model_at_end
                if HAS_TRANSFORMERS and EarlyStoppingCallback is not None:
                    callbacks.append(
                        EarlyStoppingCallback(
                            early_stopping_patience=self.recipe.early_stopping_patience,
                            early_stopping_threshold=self.recipe.early_stopping_threshold,
                        )
                    )
            else:
                candidate_args["save_strategy"] = "no"
                logger.warning(
                    "[SFTRunner] Termination criterion is 'plateau' but no eval_dataset was provided. "
                    "Evaluation and EarlyStoppingCallback disabled."
                )
            candidate_args["num_train_epochs"] = self.recipe.num_train_epochs or 5.0
            if self.recipe.max_steps is not None and self.recipe.max_steps > 0:
                candidate_args["max_steps"] = self.recipe.max_steps
        elif self.recipe.termination_criterion == "epochs":
            candidate_args["num_train_epochs"] = self.recipe.num_train_epochs or 1.0
            candidate_args["save_strategy"] = "no"
            if self.recipe.max_steps is not None and self.recipe.max_steps > 0:
                candidate_args["max_steps"] = self.recipe.max_steps
        elif self.recipe.termination_criterion == "steps":
            candidate_args["max_steps"] = self.recipe.max_steps or 500
            candidate_args["save_strategy"] = "no"
        else:  # "flops"
            candidate_args["save_strategy"] = "no"
            candidate_args["max_steps"] = 10000
            if self.recipe.num_train_epochs is not None:
                candidate_args["num_train_epochs"] = self.recipe.num_train_epochs

        # Safe SFTConfig / TrainingArguments dispatch
        if HAS_TRL and hasattr(SFTConfig, "__init__"):
            sig = inspect.signature(SFTConfig.__init__)
            valid_args = {k: v for k, v in candidate_args.items() if k in sig.parameters}
            training_args = SFTConfig(**valid_args)
        else:
            sig = inspect.signature(TrainingArguments.__init__)
            valid_args = {k: v for k, v in candidate_args.items() if k in sig.parameters}
            training_args = TrainingArguments(**valid_args)

        trainer_kwargs: Dict[str, Any] = {
            "model": model,
            "args": training_args,
            "train_dataset": dataset,
            "eval_dataset": eval_data,
            "data_collator": data_collator,
            "callbacks": callbacks,
            "accumulator": self.accumulator,
            "n_base": n_base,
            "n_lora": n_lora,
        }

        if HAS_TRL and SFTTrainer is not None and hasattr(SFTTrainer, "__init__"):
            sft_sig = inspect.signature(SFTTrainer.__init__)
            if "processing_class" in sft_sig.parameters:
                trainer_kwargs["processing_class"] = tokenizer
            elif "tokenizer" in sft_sig.parameters:
                trainer_kwargs["tokenizer"] = tokenizer
            if "dataset_text_field" in sft_sig.parameters:
                trainer_kwargs["dataset_text_field"] = "text"
            if "max_seq_length" in sft_sig.parameters:
                trainer_kwargs["max_seq_length"] = self.recipe.max_seq_length

        trainer = BudgetedSFTTrainer(**trainer_kwargs)

        budget_disp = (
            f"{self.recipe.target_flop_budget:.2e} FLOPs"
            if self.recipe.target_flop_budget
            else "None (unconstrained)"
        )
        logger.info(
            "Starting SFT distillation training on student (%s). Target budget: %s, Criterion: %s",
            self.recipe.base_model_name_or_path,
            budget_disp,
            self.recipe.termination_criterion,
        )
        trainer.train()

        # Save final adapter checkpoint
        final_adapter_dir = os.path.join(self.output_dir, "final_adapter")
        os.makedirs(final_adapter_dir, exist_ok=True)
        model.save_pretrained(final_adapter_dir)
        tokenizer.save_pretrained(final_adapter_dir)
        logger.info("Saved final SFT adapter checkpoint to %s", final_adapter_dir)

        self.intermediate_checkpoints = callback.saved_milestones

        return {
            "final_adapter_path": final_adapter_dir,
            "intermediate_checkpoints": self.intermediate_checkpoints,
            "budget_summary": self.accumulator.get_summary(),
        }
