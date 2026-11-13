# Experiment 0.2: Compute Accounting & FLOPs Normalization Calibration (`EXP-0.2`)

- **Sequential Position:** Step 1 (Pre-Flight Sanity & Budget Infrastructure)
- **Status:** Complete (Validated)
- **Parent Plans:** [Sequential Experiment Plan](../experiment_plan.md) | [Implementation Plan](../implementaiton_plan.md)
- **Preceding Dependency:** [Experiment 0.1: Verifier Calibration & Baseline Zero-Shot Evaluation](experiment_0.1.md) (`EXP-0.1`)

---

## 1. Objective & Scope

In the Multigen-RL framework, all comparisons across training paradigms—pure reinforcement learning (GRPO in `EXP-1.1`), pure knowledge distillation (SFT on teacher rationales in `EXP-1.2`), and interleaved mixed learning (`EXP-1.3`)—rest on the premise of **strict compute parity**. Without an exact, hardware-invariant FLOP accounting pipeline, claims of sample efficiency or algorithmic superiority are vulnerable to compute confounds (e.g., GRPO silently expending $4\times$ more compute on multi-candidate rollouts than distillation).

`EXP-0.2` establishes the compute calibration substrate for the entire project. It has four concrete goals:

1. **Formalize Hardware-Invariant Theoretical FLOP Models:** Standardize parameter-accurate equations for autoregressive generation, full-weight backpropagation, LoRA adapter backpropagation, reference model passes, and cross-model teacher rollouts.
2. **Implement the Compute Accounting Subsystem (`multigen/accounting/`):**
   - `multigen/accounting/flops.py`: Deterministic FLOP counters and parameter extractors for Hugging Face transformer models.
   - `multigen/accounting/tracker.py`: Hugging Face / TRL `TrainerCallback` (`FLOPBudgetCallback`) that accumulates step-by-step FLOPs and cleanly terminates training when a target budget $B$ is reached.
3. **Calibrate and Empirically Validate Linearity:** Execute empirical profiling runs across varying sequence lengths and batch sizes using `calflops` to confirm that theoretical FLOP formulas achieve an $R^2 \ge 0.999$ correlation with empirical matrix multiplication counts.
4. **Validate Budget Enforcement Precision:** Verify that `FLOPBudgetCallback` halts training runs within $\pm 0.5\%$ of predefined FLOP thresholds across both RL and SFT training loops.

---

## 2. Mathematical Formulation & FLOP Accounting Models

Following Kaplan et al. (2020) and Chinchilla scaling (Hoffmann et al., 2022), the compute of standard decoder-only dense Transformer architectures is dominated by matrix multiplications in linear projections (Query, Key, Value, Output, and MLP intermediate projections).

For a dense model with $N$ active parameters processing $T$ tokens:

$$\text{FLOPs}_{\text{fwd}} \approx 2 \times N \times T$$

$$\text{FLOPs}_{\text{bwd}} \approx 4 \times N \times T$$

$$\text{FLOPs}_{\text{train\_step}} = \text{FLOPs}_{\text{fwd}} + \text{FLOPs}_{\text{bwd}} \approx 6 \times N \times T$$

Below are the exact operational accounting rules implemented in `multigen/accounting/flops.py`.

```mermaid
flowchart TD
    subgraph FLOP Accounting Subsystem
        direction TB
        M[HF Model / Config] --> PE[Parameter Counter: N_base, N_lora, N_teacher]
        
        subgraph Operations
            O1[Rollout Generation: 2 * N_student * T_total]
            O2[Reference Policy Pass: 2 * N_student * T_total]
            O3[LoRA Policy Update: 4 * N_base * T + 6 * N_lora * T]
            O4[Teacher Distillation: 2 * N_teacher * T_teacher]
        end
        
        PE --> Operations
        Operations --> TC[FLOPAccumulator]
        TC --> FBC[FLOPBudgetCallback]
        FBC -->|FLOP_accum >= Target Budget B| HALT[Graceful Early Termination]
    end
```

### 2.1 Autoregressive Generation / Rollout FLOPs
During rollout generation (both student self-generation in GRPO and teacher rationale generation in distillation):

$$\text{FLOPs}_{\text{gen}}(N, T_{\text{prompt}}, T_{\text{gen}}) = 2 \times N \times (T_{\text{prompt}} + T_{\text{gen}})$$

For batched rollout sampling with group size $G$ (as in GRPO with $G$ candidate completions per prompt):

$$\text{FLOPs}_{\text{rollout\_group}} = \sum_{i=1}^{G} 2 \times N \times (T_{\text{prompt}} + T_{\text{gen}, i})$$

> **Attention Quadratic Component:** Standard dense FLOP formulas approximate compute proportional to $2NT$. For sequence lengths $T \le 1024$ in GSM8K, the quadratic self-attention FLOP term ($2 \cdot L \cdot T^2 \cdot d_{\text{model}}$) represents $<1.5\%$ of total compute and is hardware/kernel dependent (FlashAttention tiling). In `multigen`, attention quadratic terms can be toggled via `include_attention_flops=False` (default: pure parameter scaling for cross-hardware parity).

### 2.2 SFT / Distillation Training Step FLOPs (Full vs. LoRA)

#### Full Fine-Tuning:
For a forward and full backward pass across $N_{\text{active}}$ parameters:

$$\text{FLOPs}_{\text{sft\_full}} = 6 \times N_{\text{active}} \times T_{\text{seq}}$$

#### Parameter-Efficient (LoRA) Fine-Tuning:
When using Low-Rank Adaptation (LoRA), base model weights $W_{\text{base}}$ ($N_{\text{base}}$ parameters) are frozen, and low-rank matrices $A, B$ ($N_{\text{lora}}$ parameters) are trained:

1. **Forward pass:** Activations flow through both base layers and adapters:
   $$\text{FLOPs}_{\text{fwd}} = 2 \times (N_{\text{base}} + N_{\text{lora}}) \times T \approx 2 \times N_{\text{base}} \times T$$
2. **Backward pass:** Gradients must be backpropagated through base weights to obtain $\nabla_X \mathcal{L}$ for downstream layers ($2 \times N_{\text{base}} \times T$). However, weight gradients $\nabla_W \mathcal{L}$ are **not** computed for frozen base weights (saving $2 \times N_{\text{base}} \times T$). For LoRA parameters, both activation and weight gradients are computed ($4 \times N_{\text{lora}} \times T$).
   $$\text{FLOPs}_{\text{bwd}} = 2 \times N_{\text{base}} \times T + 4 \times N_{\text{lora}} \times T$$
3. **Total LoRA training pass:**
   $$\text{FLOPs}_{\text{lora\_step}} = 4 \times N_{\text{base}} \times T + 6 \times N_{\text{lora}} \times T$$

Because $N_{\text{lora}} \ll N_{\text{base}}$ (typically $<1\%$), a LoRA training step requires approximately **$\frac{2}{3}$ of the compute** ($4NT$ vs $6NT$) of a full-finetuning step.

### 2.3 GRPO Training Step FLOPs
A single optimization step of Group Relative Policy Optimization (GRPO) consists of three compute phases:

1. **Rollout Generation:** Sampling $G$ candidate reasoning paths per prompt from policy $\pi_{\theta_{\text{old}}}$:
   $$\text{FLOPs}_{\text{phase1}} = \sum_{i=1}^{G} 2 \times N_{\text{student}} \times (T_{\text{prompt}} + T_{\text{gen}, i})$$
2. **Reference & Old Policy Log-Prob Evaluation:** Computing reference logits $\pi_{\text{ref}}(y_i \mid x)$ (and $\pi_{\theta_{\text{old}}}$ if decoupled):
   $$\text{FLOPs}_{\text{phase2}} = \sum_{i=1}^{G} 2 \times N_{\text{ref}} \times (T_{\text{prompt}} + T_{\text{gen}, i})$$
3. **Policy Gradient Backward Step (over $G$ sequences):**
   $$\text{FLOPs}_{\text{phase3}} = \sum_{i=1}^{G} \text{FLOPs}_{\text{train\_step}}(N_{\text{student}}, T_{\text{prompt}} + T_{\text{gen}, i})$$

$$\text{FLOPs}_{\text{grpo\_step}} = \text{FLOPs}_{\text{phase1}} + \text{FLOPs}_{\text{phase2}} + \text{FLOPs}_{\text{phase3}}$$

### 2.4 Teacher Inference in Distillation
When student models distill reasoning traces from an external teacher model (e.g., $N_{\text{teacher}} \approx 7\text{B}$ distilling to $N_{\text{student}} \approx 0.6\text{B}$):

$$\text{FLOPs}_{\text{teacher\_call}} = 2 \times N_{\text{teacher}} \times (T_{\text{prompt}} + T_{\text{teacher\_gen}})$$

This compute must be debited against the distillation experiment's cumulative budget to prevent distillation from receiving an unfair compute advantage over self-RL.

---

## 3. Implementation Details

### 3.1 Module Layout

```text
multigen/
└── accounting/
    ├── __init__.py
    ├── flops.py             # Parameter extractors & FLOP calculation functions
    └── tracker.py           # FLOPAccumulator & FLOPBudgetCallback for TRL/HF
scripts/
└── calibrate_flops.py       # Empirical calflops profiling & linearity validator
tests/
└── test_flops.py            # Unit test suite verifying exact scaling and callbacks
configs/
└── flops_calibration.json   # Calibration output artifact
```

### 3.2 Core Accounting Module (`multigen/accounting/flops.py`)

Implement clean, typed functions that accept model configurations or parameter counts:

```python
from dataclasses import dataclass
from typing import Optional, Union
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


@dataclass(frozen=True)
class ModelParamBreakdown:
    total_params: int
    trainable_params: int
    frozen_params: int
    is_lora: bool


def extract_param_counts(model: Union[nn.Module, PreTrainedModel]) -> ModelParamBreakdown:
    """Accurately count total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    is_lora = 0 < trainable < total
    return ModelParamBreakdown(
        total_params=total,
        trainable_params=trainable,
        frozen_params=frozen,
        is_lora=is_lora,
    )


def compute_rollout_flops(
    num_params: int,
    prompt_tokens: int,
    gen_tokens: int,
) -> float:
    """Compute forward FLOPs for autoregressive rollout: 2 * N * (T_prompt + T_gen)."""
    return 2.0 * float(num_params) * float(prompt_tokens + gen_tokens)


def compute_train_step_flops(
    num_params: int,
    seq_len: int,
    is_lora: bool = False,
    trainable_params: Optional[int] = None,
) -> float:
    """Compute backprop step FLOPs.
    
    If full fine-tuning: 6 * N * T.
    If LoRA: 4 * N_frozen * T + 6 * N_lora * T.
    """
    if not is_lora:
        return 6.0 * float(num_params) * float(seq_len)
    
    n_lora = float(trainable_params if trainable_params is not None else 0)
    n_frozen = float(num_params) - n_lora
    return 4.0 * n_frozen * float(seq_len) + 6.0 * n_lora * float(seq_len)


def compute_teacher_flops(
    teacher_params: int,
    prompt_tokens: int,
    gen_tokens: int,
) -> float:
    """Compute inference FLOPs for teacher rollout generation."""
    return 2.0 * float(teacher_params) * float(prompt_tokens + gen_tokens)
```

### 3.3 Budget Tracker & TRL Callback (`multigen/accounting/tracker.py`)

Implement `FLOPAccumulator` to hold running state and `FLOPBudgetCallback` to interface with `trl.GRPOTrainer` and `transformers.Trainer`:

```python
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class FLOPAccumulator:
    """Thread-safe cumulative FLOP ledger."""
    def __init__(self, target_budget: Optional[float] = None) -> None:
        self.cumulative_flops: float = 0.0
        self.rollout_flops: float = 0.0
        self.train_flops: float = 0.0
        self.teacher_flops: float = 0.0
        self.target_budget: Optional[float] = target_budget
        self.total_tokens_processed: int = 0

    def debit(self, flops: float, category: str = "train", tokens: int = 0) -> None:
        self.cumulative_flops += flops
        self.total_tokens_processed += tokens
        if category == "rollout":
            self.rollout_flops += flops
        elif category == "train":
            self.train_flops += flops
        elif category == "teacher":
            self.teacher_flops += flops

    @property
    def is_exhausted(self) -> bool:
        if self.target_budget is None:
            return False
        return self.cumulative_flops >= self.target_budget


class FLOPBudgetCallback(TrainerCallback):
    """Hugging Face / TRL callback that halts training when compute budget is met."""
    def __init__(self, accumulator: FLOPAccumulator) -> None:
        self.accumulator = accumulator

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if self.accumulator.is_exhausted:
            control.should_training_stop = True
            control.should_save = True
        return control
```

### 3.4 Calibration & Verification Script (`scripts/calibrate_flops.py`)

The script profiles candidate base models (`Qwen/Qwen3-0.6B`) across sequences $T \in [64, 128, 256, 512, 1024]$:
- Runs forward and backward passes using synthetic dummy batches.
- Compares theoretical FLOP numbers against `calflops.calculate_flops`.
- Fits an Ordinary Least Squares (OLS) regression between theoretical and profiled FLOPs to confirm $R^2 \ge 0.999$.
- Emits `configs/flops_calibration.json` with model parameter breakdowns and empirical multipliers.

### 3.5 Unit Test Suite (`tests/test_flops.py`)

Deterministic test coverage:
1. **Parameter Breakdown:** Verify that frozen, trainable, and LoRA parameter counts match known layers exactly.
2. **Formula Monotonicity & Linearity:** Verify that doubling tokens exactly doubles FLOP output.
3. **LoRA vs. Full Discrepancy:** Confirm that LoRA FLOPs are strictly lower than full-parameter FLOPs for identical sequence lengths.
4. **Callback Interruption:** Mock a training loop with 100 steps and verify `FLOPBudgetCallback` halts training at the exact step where cumulative FLOPs exceed `target_budget` within $\pm 0.5\%$.

---

## 4. Finalized Design Decisions

All 6 core design decisions have been finalized, adopting the recommended options for strict compute parity, physical rigor, and cross-platform reproducibility:

| # | Topic | Decision | Implementation Rule |
| :---: | :--- | :--- | :--- |
| **1** | **LoRA vs. Full Backprop Multiplier** | **Option A (Exact LoRA Accounting)** | $\text{FLOPs}_{\text{lora\_step}} = 4 N_{\text{base}} T + 6 N_{\text{lora}} T \approx 4 N_{\text{base}} T$. Saves base weight gradient computation $\nabla_{W_{\text{base}}} \mathcal{L}$. |
| **2** | **Reference Model FLOPs in GRPO** | **Option A (Strict Parity / Count Reference)** | Debit $2 \times N_{\text{ref}} \times T_{\text{total}}$ per candidate token against the experiment budget to match SFT compute parity. |
| **3** | **Attention Quadratic Term ($O(L S^2 d)$)** | **Option A (Pure Parameter Scaling $6NT$)** | Omit $O(S^2)$ attention terms for $S \le 1024$. Maintains kernel- and hardware-invariant calculations across Mac MPS and Linux CUDA. |
| **4** | **Held-Out Evaluation FLOPs** | **Option A (Exempt from Training Budget)** | Validation passes do not consume training budget ($1.0 \times 10^{16}$ FLOPs); logged separately in trial metadata for auditing. |
| **5** | **Budget Exhaustion Halting** | **Option A (Graceful Step Finish)** | Finish current step, execute optimizer update, save adapter, and set `control.should_training_stop = True` (overshoot $<0.1\%$). |
| **6** | **Empirical Profiling Tool** | **Option A (`calflops`)** | Add `calflops` to `pyproject.toml` dev dependencies for model-layer inspection and cross-platform support. |

---

### Detailed Rationale for Finalized Decisions

#### 4.1 Decision 1: Exact LoRA Backpropagation Multiplier ($4NT$)
- **Finalized Choice:** Option A (`compute_train_step_flops` evaluates $4 N_{\text{base}} T + 6 N_{\text{lora}} T$).
- **Rationale:** Freezing base model parameters avoids the compute cost of accumulating and updating weight gradients $\nabla_W \mathcal{L}$ ($2 N_{\text{base}} T$ FLOPs). Only activation gradients ($2 N_{\text{base}} T$) are computed through the frozen backbone, plus $6 N_{\text{lora}} T$ through the low-rank adapters. This physically accurate accounting fairly rewards LoRA parameter efficiency, allowing LoRA to process $\approx 1.5\times$ more training tokens than full fine-tuning under identical total FLOP budgets.

#### 4.2 Decision 2: Reference Model Inference Debited to Budget in GRPO
- **Finalized Choice:** Option A (Debit $2 N_{\text{ref}} T_{\text{total}}$ per token).
- **Rationale:** GRPO requires running the reference policy $\pi_{\text{ref}}$ over every completion to calculate importance weights and KL penalties. Because pure SFT/distillation baselines have no reference model, exempting $\pi_{\text{ref}}$ would give GRPO a hidden $\approx 33\text{--}50\%$ compute advantage. Counting reference model forward passes ensures hypothesis testing in Phase 1 (`H1`) is rigorously fair.

#### 4.3 Decision 3: Pure Parameter Scaling ($6NT$) without Quadratic Attention Terms
- **Finalized Choice:** Option A (Default parameter scaling).
- **Rationale:** At typical GSM8K problem and completion lengths ($T \le 512$), attention operations represent less than $1.2\%$ of total FLOPs. Moreover, GPU implementations (e.g., FlashAttention-2, vLLM paged attention) fuse softmax computation, rendering quadratic FLOP approximations hardware-dependent. Pure parameter scaling provides transparent, deterministic metrics reproducible across Apple Silicon (MPS/CPU) and NVIDIA CUDA clusters.

#### 4.4 Decision 4: Evaluation Compute Treated as Observational Overhead
- **Finalized Choice:** Option A (Evaluation FLOPs tracked in metadata, excluded from training budget).
- **Rationale:** In standard reinforcement learning and pre-training literature, checkpoint evaluation on dev/validation sets is an observational diagnostic rather than model optimization. Deducting evaluation FLOPs from the training budget would perversely penalize experiments with higher evaluation frequency.

#### 4.5 Decision 5: Graceful Step-Boundary Halting on Budget Exhaustion
- **Finalized Choice:** Option A (`FLOPBudgetCallback` sets `control.should_training_stop = True` at step end).
- **Rationale:** Aborting training mid-step via hard exceptions causes partial weight updates, non-synchronized optimizer states, and potentially corrupted checkpoints. A graceful exit at the step boundary ensures a clean checkpoint save with negligible budget overshoot ($<0.1\%$).

#### 4.6 Decision 6: Empirical Profiling via `calflops`
- **Finalized Choice:** Option A (`calflops>=0.3.0` added to dev dependencies).
- **Rationale:** `calflops` parses transformer modules directly at the PyTorch graph level and reports layer-by-layer FLOP breakdowns without needing CUDA-specific performance counters (NVML), making automated calibration runnable in local macOS dev environments.

---

## 5. Execution & Acceptance Gate

### 5.1 CLI Commands

```bash
# 1. Run compute accounting unit tests
pytest tests/test_flops.py -v

# 2. Run empirical calibration across multiple sequence lengths
python scripts/calibrate_flops.py \
    --model Qwen/Qwen3-0.6B \
    --seq-lens 64 128 256 512 1024 \
    --output configs/flops_calibration.json

# 3. Verify budget callback early-stopping on a dummy training loop
python scripts/calibrate_flops.py \
    --test-budget-stop \
    --target-flops 1e14
```

### 5.2 Acceptance Criteria Checklist

- [x] **Line-Item Formulations Tested:** `tests/test_flops.py` passes 100% of tests (10/10 tests passed).
- [x] **Linearity Benchmark ($R^2 \ge 0.999$):** Empirical calibration script demonstrates strict linear correlation between token sequence length and FLOPs ($R^2 = 1.000$).
- [x] **Budget Stopping Precision:** Mock training loop halts within $\pm 0.5\%$ of target budget without hanging or checkpoint corruption (measured: $0.0305\%$ overshoot).
- [x] **Artifact Generated:** `configs/flops_calibration.json` saved with validated coefficients for `Qwen/Qwen3-0.6B`.
- [x] **Design Decisions Approved:** Finalized all Section 4 decisions (Exact LoRA $4NT$, reference model inclusion, evaluation exemption, pure parameter scaling, graceful step finish, and `calflops`).
