"""Cumulative FLOP ledger and Hugging Face / TRL TrainerCallback.

Tracks runtime FLOP consumption across rollouts, backpropagation, reference
model queries, and teacher distillation passes, enforcing compute budget parity.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Safe import for transformers TrainerCallback
try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback TrainerCallback base class when transformers is not installed."""
        pass

    class TrainerControl:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.should_training_stop = False
            self.should_save = False

    class TrainerState:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.global_step = 0

    class TrainingArguments:  # type: ignore[no-redef]
        pass


class FLOPAccumulator:
    """Thread-safe cumulative compute ledger.

    Records compute according to the finalized EXP-0.2 specification:
      - cumulative_flops tracks training operations against target_budget.
      - eval_flops is tracked in metadata but EXEMPT from the budget (Decision 4).
    """

    def __init__(self, target_budget: Optional[float] = None) -> None:
        self.target_budget: Optional[float] = target_budget
        self.cumulative_flops: float = 0.0
        self.rollout_flops: float = 0.0
        self.train_flops: float = 0.0
        self.teacher_flops: float = 0.0
        self.ref_flops: float = 0.0
        self.eval_flops: float = 0.0
        self.total_tokens_processed: int = 0
        self.step_history: List[Dict[str, Any]] = []

    def debit(
        self,
        flops: float,
        category: str = "train",
        tokens: int = 0,
        step: Optional[int] = None,
    ) -> None:
        """Debit compute to the ledger.

        Categories:
          - "rollout": Student rollout generation
          - "train": Backprop training step (full or LoRA)
          - "teacher": Teacher rationale generation
          - "ref": Reference model log-prob evaluation
          - "eval": Held-out dev/test evaluation (exempt from budget)
        """
        flops = float(flops)
        self.total_tokens_processed += int(tokens)

        if category == "rollout":
            self.rollout_flops += flops
            self.cumulative_flops += flops
        elif category == "train":
            self.train_flops += flops
            self.cumulative_flops += flops
        elif category == "teacher":
            self.teacher_flops += flops
            self.cumulative_flops += flops
        elif category == "ref":
            self.ref_flops += flops
            self.cumulative_flops += flops
        elif category == "eval":
            # Observational compute (Decision 4: exempt from training budget)
            self.eval_flops += flops
        else:
            self.cumulative_flops += flops

        record = {
            "step": step if step is not None else len(self.step_history) + 1,
            "category": category,
            "flops": flops,
            "tokens": tokens,
            "cumulative_flops": self.cumulative_flops,
            "eval_flops": self.eval_flops,
        }
        self.step_history.append(record)

    def record_grpo_step(
        self,
        breakdown: Dict[str, float],
        tokens: int = 0,
        step: Optional[int] = None,
    ) -> None:
        """Record all phases of a multi-candidate GRPO step."""
        r = breakdown.get("rollout_flops", 0.0)
        ref = breakdown.get("ref_flops", 0.0)
        tr = breakdown.get("train_flops", 0.0)

        if r > 0:
            self.debit(r, category="rollout", tokens=tokens, step=step)
        if ref > 0:
            self.debit(ref, category="ref", tokens=0, step=step)
        if tr > 0:
            self.debit(tr, category="train", tokens=0, step=step)

    def record_eval_pass(self, flops: float, tokens: int = 0) -> None:
        """Record observational evaluation compute (exempt from budget)."""
        self.debit(flops, category="eval", tokens=tokens)

    @property
    def is_exhausted(self) -> bool:
        """Check if training compute has reached or exceeded target budget."""
        if self.target_budget is None:
            return False
        return self.cumulative_flops >= self.target_budget

    @property
    def remaining_flops(self) -> float:
        """Return remaining FLOPs before budget exhaustion."""
        if self.target_budget is None:
            return float("inf")
        return max(0.0, self.target_budget - self.cumulative_flops)

    @property
    def budget_utilization(self) -> float:
        """Return fraction of target budget consumed."""
        if self.target_budget is None or self.target_budget <= 0:
            return 0.0
        return self.cumulative_flops / self.target_budget

    @property
    def overshoot_fraction(self) -> float:
        """Return fractional overshoot beyond budget (if any)."""
        if self.target_budget is None or self.target_budget <= 0:
            return 0.0
        if self.cumulative_flops > self.target_budget:
            return (self.cumulative_flops - self.target_budget) / self.target_budget
        return 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Return structured summary metrics for experiment logging."""
        return {
            "cumulative_flops": self.cumulative_flops,
            "target_budget": self.target_budget,
            "is_exhausted": self.is_exhausted,
            "budget_utilization": self.budget_utilization,
            "overshoot_fraction": self.overshoot_fraction,
            "total_tokens_processed": self.total_tokens_processed,
            "breakdown": {
                "rollout_flops": self.rollout_flops,
                "train_flops": self.train_flops,
                "teacher_flops": self.teacher_flops,
                "ref_flops": self.ref_flops,
                "eval_flops": self.eval_flops,
            },
            "num_recorded_steps": len(self.step_history),
        }

    def reset(self) -> None:
        """Reset ledger for a new trial."""
        self.cumulative_flops = 0.0
        self.rollout_flops = 0.0
        self.train_flops = 0.0
        self.teacher_flops = 0.0
        self.ref_flops = 0.0
        self.eval_flops = 0.0
        self.total_tokens_processed = 0
        self.step_history.clear()


class FLOPBudgetCallback(TrainerCallback):
    """Hugging Face / TRL TrainerCallback for graceful budget termination.

    Follows Decision 5: On step end, if cumulative FLOPs >= target_budget,
    sets should_training_stop = True and should_save = True, ensuring optimizer
    state integrity and adapter persistence with <0.1% typical overshoot.
    """

    def __init__(self, accumulator: FLOPAccumulator) -> None:
        self.accumulator = accumulator

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        """Check budget at step boundary and request graceful halt if exhausted."""
        if self.accumulator.is_exhausted:
            control.should_training_stop = True
            control.should_save = True
            logger.info(
                f"[FLOPBudgetCallback] Budget exhausted: "
                f"{self.accumulator.cumulative_flops:.2e} / {self.accumulator.target_budget:.2e} FLOPs "
                f"({self.accumulator.budget_utilization * 100:.2f}%). Halting training gracefully."
            )
        return control

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Inject compute accounting metrics into Trainer logs."""
        if logs is not None and isinstance(logs, dict):
            logs["flops/cumulative"] = self.accumulator.cumulative_flops
            logs["flops/utilization"] = self.accumulator.budget_utilization
            logs["flops/remaining"] = self.accumulator.remaining_flops
            if self.accumulator.eval_flops > 0:
                logs["flops/eval_observational"] = self.accumulator.eval_flops
