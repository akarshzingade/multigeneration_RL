"""Hardware-invariant FLOP accounting formulas and parameter extractors.

Implements parameter-accurate accounting based on Kaplan et al. (2020) and
Chinchilla scaling (Hoffmann et al., 2022) as specified in EXP-0.2:
  - Rollout generation: 2 * N * (T_prompt + T_gen)
  - Full fine-tuning:   6 * N * T_seq
  - LoRA fine-tuning:   4 * N_base * T_seq + 6 * N_lora * T_seq
  - Teacher inference:  2 * N_teacher * (T_prompt + T_gen)
  - GRPO step:          Rollouts + Reference Passes (optional) + Policy Step
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass(frozen=True)
class ModelParamBreakdown:
    """Breakdown of model parameters for FLOP calculations."""

    total_params: int
    trainable_params: int
    frozen_params: int
    is_lora: bool
    lora_params: int = 0
    non_embedding_params: Optional[int] = None

    @property
    def base_params(self) -> int:
        """Parameters of the frozen base architecture when using adapters."""
        if self.is_lora:
            return self.total_params - self.lora_params
        return self.total_params


def extract_param_counts(model: Any) -> ModelParamBreakdown:
    """Accurately extract total, trainable, and frozen parameters from a model.

    Supports PyTorch nn.Module, Hugging Face PreTrainedModel, PEFT models,
    and duck-typed mock models for unit tests.
    """
    total = 0
    trainable = 0
    lora_params = 0
    is_lora = False

    # Check for PEFT / LoRA wrappers
    if hasattr(model, "peft_config") or hasattr(model, "active_peft_config"):
        is_lora = True

    # 1. Named parameters iteration if available
    if hasattr(model, "named_parameters") and callable(model.named_parameters):
        for name, param in model.named_parameters():
            if param is None:
                continue
            numel = getattr(param, "numel", None)
            if callable(numel):
                n = numel()
            elif hasattr(param, "shape"):
                n = 1
                for dim in param.shape:
                    n *= dim
            else:
                n = int(getattr(param, "size", 0))

            total += n
            req_grad = getattr(param, "requires_grad", True)
            if req_grad:
                trainable += n
                if "lora_" in name.lower() or "adapter" in name.lower():
                    lora_params += n
                    is_lora = True

    # 2. Plain parameters iteration
    elif hasattr(model, "parameters") and callable(model.parameters):
        for param in model.parameters():
            if param is None:
                continue
            numel = getattr(param, "numel", None)
            if callable(numel):
                n = numel()
            elif hasattr(param, "shape"):
                n = 1
                for dim in param.shape:
                    n *= dim
            else:
                n = int(getattr(param, "size", 0))

            total += n
            req_grad = getattr(param, "requires_grad", True)
            if req_grad:
                trainable += n

    # 3. Duck-typed config or dictionary representations
    elif hasattr(model, "num_parameters") and callable(model.num_parameters):
        total = int(model.num_parameters())
        trainable = total
    elif isinstance(model, dict):
        total = int(model.get("total_params", 0))
        trainable = int(model.get("trainable_params", total))
        lora_params = int(model.get("lora_params", 0))
        is_lora = bool(model.get("is_lora", lora_params > 0))

    # Detect LoRA via partial trainable count if not already marked
    if not is_lora and 0 < trainable < total:
        is_lora = True
        if lora_params == 0:
            lora_params = trainable

    frozen = total - trainable
    if is_lora and lora_params == 0:
        lora_params = trainable

    return ModelParamBreakdown(
        total_params=total,
        trainable_params=trainable,
        frozen_params=frozen,
        is_lora=is_lora,
        lora_params=lora_params,
    )


def compute_rollout_flops(
    num_params: int,
    prompt_tokens: int,
    gen_tokens: int,
    include_attention_flops: bool = False,
    num_layers: Optional[int] = None,
    hidden_size: Optional[int] = None,
) -> float:
    """Compute forward FLOPs for autoregressive rollout generation.

    Formula:
        FLOPs = 2 * N * (T_prompt + T_gen)

    Optional Attention Quadratic component (Decision 3, default False):
        FLOPs_attn = 2 * L * (T_prompt + T_gen)^2 * d_model
    """
    total_tokens = float(prompt_tokens + gen_tokens)
    dense_flops = 2.0 * float(num_params) * total_tokens

    if include_attention_flops and num_layers is not None and hidden_size is not None:
        attn_flops = 2.0 * float(num_layers) * (total_tokens**2) * float(hidden_size)
        return dense_flops + attn_flops

    return dense_flops


def compute_train_step_flops(
    num_params: int,
    seq_len: int,
    is_lora: bool = False,
    trainable_params: Optional[int] = None,
) -> float:
    """Compute training step FLOPs (forward + backward passes).

    Decisions Implemented (Decision 1):
      - Full fine-tuning:
          FLOPs = 6 * N * T_seq (2*N*T forward + 4*N*T backward)
      - Exact LoRA fine-tuning:
          FLOPs = 4 * N_base * T_seq + 6 * N_lora * T_seq
          (Saves computation of base weight gradients 2 * N_base * T)
    """
    t = float(seq_len)
    if not is_lora:
        return 6.0 * float(num_params) * t

    n_lora = float(trainable_params if trainable_params is not None else 0)
    n_base = float(num_params) - n_lora
    if n_base < 0:
        n_base = float(num_params)

    # 2*N_base*T (fwd) + 2*N_base*T (bwd activation) + 6*N_lora*T (fwd + bwd act + bwd weights)
    return 4.0 * n_base * t + 6.0 * n_lora * t


def compute_teacher_flops(
    teacher_params: int,
    prompt_tokens: int,
    gen_tokens: int,
) -> float:
    """Compute inference FLOPs for teacher rationale generation.

    Formula:
        FLOPs = 2 * N_teacher * (T_prompt + T_gen)
    """
    return 2.0 * float(teacher_params) * float(prompt_tokens + gen_tokens)


def compute_grpo_step_flops(
    student_params: int,
    prompt_tokens: int,
    completion_tokens_list: Sequence[int],
    is_lora: bool = True,
    student_trainable_params: Optional[int] = None,
    ref_params: Optional[int] = None,
    count_ref_model: bool = True,
) -> Dict[str, float]:
    """Compute total FLOPs consumed in a single GRPO step.

    A GRPO step consists of:
      1. Rollouts: Generating G completions from student policy
      2. Reference passes: Evaluating log-probs under frozen reference policy
         (Decision 2: counted against budget by default for fair parity)
      3. Policy updates: Backpropagating policy loss over all G completions
    """
    g = len(completion_tokens_list)
    if g == 0:
        return {
            "rollout_flops": 0.0,
            "ref_flops": 0.0,
            "train_flops": 0.0,
            "total_flops": 0.0,
        }

    # Phase 1: Rollout generation (G samples)
    rollout_flops = 0.0
    for comp_len in completion_tokens_list:
        rollout_flops += compute_rollout_flops(
            num_params=student_params,
            prompt_tokens=prompt_tokens,
            gen_tokens=comp_len,
        )

    # Phase 2: Reference policy forward pass
    ref_flops = 0.0
    if count_ref_model:
        n_ref = ref_params if ref_params is not None else student_params
        for comp_len in completion_tokens_list:
            ref_flops += compute_rollout_flops(
                num_params=n_ref,
                prompt_tokens=prompt_tokens,
                gen_tokens=comp_len,
            )

    # Phase 3: Policy backward updates
    train_flops = 0.0
    for comp_len in completion_tokens_list:
        seq_len = prompt_tokens + comp_len
        train_flops += compute_train_step_flops(
            num_params=student_params,
            seq_len=seq_len,
            is_lora=is_lora,
            trainable_params=student_trainable_params,
        )

    total_flops = rollout_flops + ref_flops + train_flops
    return {
        "rollout_flops": rollout_flops,
        "ref_flops": ref_flops,
        "train_flops": train_flops,
        "total_flops": total_flops,
    }


def compute_sft_step_flops(
    student_params: int,
    batch_seq_lengths: Sequence[int],
    is_lora: bool = True,
    student_trainable_params: Optional[int] = None,
) -> float:
    """Compute total FLOPs consumed in a single SFT training step across a batch.

    Formula per token:
      - Full fine-tuning: 6 * N_student * T
      - LoRA fine-tuning: 4 * N_base * T + 6 * N_lora * T
    """
    total = 0.0
    for seq_len in batch_seq_lengths:
        total += compute_train_step_flops(
            num_params=student_params,
            seq_len=seq_len,
            is_lora=is_lora,
            trainable_params=student_trainable_params,
        )
    return total


def compute_distillation_budget_split(
    target_budget_flops: float = 1.0e16,
    teacher_ratio: float = 0.20,
    accounting_mode: str = "amortized",
) -> Dict[str, float]:
    """Compute budget allocation for teacher generation vs student SFT.

    In 'amortized' mode:
      - student_budget = target_budget_flops (1.0e16)
      - teacher_budget is tracked as offline curation compute.
    In 'strict_split' mode:
      - teacher_budget = teacher_ratio * target_budget_flops
      - student_budget = (1.0 - teacher_ratio) * target_budget_flops
    """
    if accounting_mode == "strict_split":
        teacher_budget = target_budget_flops * teacher_ratio
        student_budget = target_budget_flops - teacher_budget
    else:
        teacher_budget = target_budget_flops * teacher_ratio
        student_budget = target_budget_flops

    return {
        "target_budget_flops": target_budget_flops,
        "accounting_mode": accounting_mode,
        "teacher_budget_flops": teacher_budget,
        "student_budget_flops": student_budget,
    }
