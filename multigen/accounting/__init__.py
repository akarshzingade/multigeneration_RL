"""Compute Accounting Subsystem for Multigen-RL.

Provides hardware-invariant FLOP accounting, parameter counting,
and compute budget tracking callbacks for strict compute parity.
"""

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
)

__all__ = [
    "ModelParamBreakdown",
    "extract_param_counts",
    "compute_rollout_flops",
    "compute_train_step_flops",
    "compute_teacher_flops",
    "compute_grpo_step_flops",
    "FLOPAccumulator",
    "FLOPBudgetCallback",
]
