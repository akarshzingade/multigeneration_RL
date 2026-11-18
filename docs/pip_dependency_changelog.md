# Pip Package & Dependency Changelog

This document tracks all modifications to Python pip dependencies, version pins, package additions, and environment constraints across [`requirements.txt`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/requirements.txt), [`requirements-gpu.txt`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/requirements-gpu.txt), [`pyproject.toml`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/pyproject.toml), and cloud container images ([`scripts/modal_mixed_runner.py`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/scripts/modal_mixed_runner.py), [`scripts/modal_grpo_runner.py`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/scripts/modal_grpo_runner.py), [`scripts/modal_distill_runner.py`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/scripts/modal_distill_runner.py)).

Any future package upgrade, version pin, or removal **must** be logged here along with the technical rationale and compatibility constraints.

---

## Dependency Invariants & Version Matrix

| Package | Minimum Version | Files Enforcing Constraint | Technical Purpose & Invariant |
| :--- | :--- | :--- | :--- |
| **`trl`** | **`>=0.14.0`** | `requirements.txt`<br>`pyproject.toml`<br>`scripts/modal_mixed_runner.py`<br>`scripts/modal_grpo_runner.py` | Required for `GRPOTrainer` and `GRPOConfig`. Versions `<0.14.0` lack GRPO support and crash at import. |
| **`torch`** | **`>=2.4.0`** (GPU)<br>`>=2.1.0` (Core) | `requirements.txt`<br>`pyproject.toml`<br>`scripts/modal_*.py` | PyTorch 2.4+ includes native `torch.distributed.tensor.DTensor` needed for vLLM generation in TRL and FlashAttention-2 support on Ada Lovelace/Ampere. |
| **`vllm`** | **`>=0.4.0`** | `requirements-gpu.txt`<br>`pyproject.toml` (`sys_platform == 'linux'`)<br>`scripts/modal_*.py` | High-throughput rollout engine. Linux + CUDA only; must remain platform-conditioned so macOS local dev does not fail. |
| **`transformers`** | **`>=4.40.0`** | `requirements.txt`<br>`pyproject.toml`<br>`scripts/modal_*.py` | Compatible with Qwen2.5 and Qwen3 architectures (`AutoModelForCausalLM`). |
| **`peft`** | **`>=0.10.0`** | `requirements.txt`<br>`pyproject.toml`<br>`scripts/modal_*.py` | LoRA parameter-efficient fine-tuning with activation checkpointing support. |
| **`math-verify`** | **`>=0.1.0`** | `requirements.txt`<br>`pyproject.toml`<br>`scripts/modal_*.py` | Hugging Face official rule-based math verification library for grading GSM8K responses without brittle regexes. |
| **`datasets`** | **`>=2.18.0`** | `requirements.txt`<br>`pyproject.toml`<br>`scripts/modal_*.py` | Supports HF Hub streaming protocol and automatic dataset caching for GSM8K. |
| **`calflops`** | **`>=0.3.0`** | `requirements-gpu.txt`<br>`pyproject.toml` (`dev`) | Static FLOP analysis used to verify analytical compute formulas ($N_{\text{base}}, N_{\text{lora}}$). |

---

## Pip Package Change Records

### [2026-11-16] Bump `trl` to `>=0.14.0`
- **Files Modified:**
  - [`scripts/modal_mixed_runner.py`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/scripts/modal_mixed_runner.py#L46)
  - [`requirements.txt`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/requirements.txt#L9)
  - [`pyproject.toml`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/pyproject.toml#L20)
- **Old Constraint:** `trl>=0.8.0,<0.14.0` (in Modal runner) / `trl>=0.8.0` (in requirements)
- **New Constraint:** `trl>=0.14.0`
- **Root Cause & Technical Rationale:**
  - During Experiment 1.3 Stage 2 (GRPO Exploration), `from trl import GRPOConfig, GRPOTrainer` raised:
    ```text
    ImportError: trl is required to run GRPORunner.
    ```
  - `GRPOTrainer` and `GRPOConfig` were added to TRL in release **0.14.0**.
  - The upper bound `<0.14.0` previously present in the Modal runner image forced pip to resolve to `trl==0.13.0`, where the entire `trl.trainer.grpo_trainer` module does not exist.
- **Guideline for Future Changes:**
  - **Do NOT reintroduce an upper bound like `<0.14.0` or `<0.15.0` on `trl`.** Any runner or environment executing GRPO requires `trl >= 0.14.0`.

---

### [2026-11-14] Standardize `torch>=2.4.0` on Modal GPU Images
- **Files Modified:**
  - [`scripts/modal_mixed_runner.py`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/scripts/modal_mixed_runner.py#L44)
  - [`scripts/modal_distill_runner.py`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/scripts/modal_distill_runner.py)
- **Old Constraint:** `torch>=2.2.0`
- **New Constraint:** `torch>=2.4.0`
- **Root Cause & Technical Rationale:**
  - `trl >= 0.14.0` imports `DTensor` from `torch.distributed.tensor` inside `trl/trainer/utils.py`. In PyTorch versions `<2.4.0`, this import fails with:
    ```text
    ImportError: cannot import name 'DTensor' from 'torch.distributed.tensor'
    ```
  - Standardizing GPU runners on `torch>=2.4.0` ensures compatibility with `DTensor`, CUDA 12.4, and modern vLLM kernels.
- **Guideline for Future Changes:**
  - Cloud container images with GPU access must always specify `torch>=2.4.0`. Core requirements (`requirements.txt`) may keep `torch>=2.1.0` for CPU development on macOS.

---

### [2026-11-13] Conditional `vllm` Dependency for Multi-Platform Support
- **Files Modified:**
  - [`requirements.txt`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/requirements.txt)
  - [`requirements-gpu.txt`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/requirements-gpu.txt)
  - [`pyproject.toml`](file:///Users/akarshzingade/Desktop/coding/multigen%20rl/pyproject.toml#L23)
- **Old Constraint:** Unconditional `vllm>=0.4.0`
- **New Constraint:** Split into `requirements-gpu.txt` and `pyproject.toml` with `sys_platform == 'linux'` marker.
- **Root Cause & Technical Rationale:**
  - `vllm` does not provide macOS (Darwin) wheels and requires CUDA runtime libraries. Unconditional inclusion caused `pip install` to fail on Apple Silicon (macOS) developer machines.
  - Adding the environment marker allows local unit testing on macOS (falling back to Hugging Face standard generation) while ensuring vLLM installs seamlessly on Linux GPU workers.
- **Guideline for Future Changes:**
  - Keep `vllm` platform-conditioned. Never make `vllm` an unconditional requirement in `requirements.txt`.
