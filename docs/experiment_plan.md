# Multigen-RL: Sequential Experiment Plan

**Population-Based Continual Model Development & Adaptive Experimentation**  
*Document Version: 1.0 | Status: Active Specification*

---

## 1. Overview & Experimental Philosophy

This document outlines the end-to-end sequential experimental execution plan for the **Multigen-RL** framework. The core research objective is to investigate:

> **Core Research Question:** *Can population-based model development, combined with adaptive experiment selection, discover better models—or more useful collections of models—with less compute than conventional, fixed training pipelines?*

To validate this across our five core research hypotheses (**H1** through **H5**), all experiments are arranged in strict sequential order. Each experiment builds directly upon the validated artifacts, baselines, and compute baselines of its predecessors.

### 1.1 Experiment Flow & Dependency Graph

```mermaid
flowchart TD
    subgraph Phase 0: Calibration & Tooling
        E0_1[Exp 0.1: Reward & Verifier Calibration] --> E0_2[Exp 0.2: Compute & FLOPs Normalization]
    end

    subgraph Phase 1: Foundational Baselines & Mixed Learning (H1)
        E0_2 --> E1_1[Exp 1.1: Pure RL / GRPO Baseline]
        E0_2 --> E1_2[Exp 1.2: Pure Distillation Baseline]
        E1_1 --> E1_3[Exp 1.3: Sequential Mixed Learning H1]
        E1_2 --> E1_3
        E1_3 --> E1_4[Exp 1.4: Trace-Injected & Length-Penalized GRPO]
    end

    subgraph Phase 2: Behavioral Diversity & Solution Space (H2)
        E1_4 --> E2_1[Exp 2.1: Multi-Objective Reward Branching]
        E2_1 --> E2_2[Exp 2.2: Error Orthogonality & Oracle Pass@k H2]
    end

    subgraph Phase 3: Cross-Model Peer Learning & Merging (H3)
        E2_2 --> E3_0[Exp 3.0: Weight-Space Merging: Linear, TIES, DARE]
        E3_0 --> E3_1[Exp 3.1: Plateaued Student Peer-Distillation H3]
        E3_1 --> E3_2[Exp 3.2: Bidirectional Co-Evolution & Reciprocal Transfer]
    end

    subgraph Phase 4: Adaptive Experiment Controller (H4)
        E3_2 --> E4_1[Exp 4.1: Static & Random Search Baselines]
        E4_1 --> E4_2[Exp 4.2: Adaptive Multi-Objective Search H4]
    end

    subgraph Phase 5: Ecosystem-Level Ensembling & Deployment (H5)
        E4_2 --> E5_1[Exp 5.1: Consensus & Diverse Majority Voting H5]
        E5_1 --> E5_2[Exp 5.2: Dynamic Multi-LoRA Routing vs Monolithic Model]
    end

    subgraph Phase 6: Generalization & Robustness
        E5_2 --> E6_1[Exp 6.1: Out-of-Domain Transfer on SVAMP & MATH-500]
    end
```

### 1.2 Master Experiment Roadmap

| Seq # | Exp ID | Focus / Title | Primary Hypothesis | Base Budget (FLOPs / Model) | Key Decision Gate |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **0** | **EXP-0.1** | Verifier & Baseline Zero-Shot Evaluation | Tooling Sanity | N/A (Inference) | Reward accuracy = 100% on test suite |
| **1** | **EXP-0.2** | Compute Accounting & FLOPs Calibration | Tooling Sanity | N/A (Profiling) | FLOP estimation variance < 1% |
| **2** | **EXP-1.1** | Standalone RL (GRPO) Baseline | Baseline (H1) | $1.0 \times 10^{16}$ | Stable policy gradients, >10% $\Delta$Acc (Achieved: 64.52%) |
| **3** | **EXP-1.2** | Standalone Distillation (KD) Baseline | Baseline (H1) | $1.0 \times 10^{16}$ | Teacher trace extraction & SFT convergence (Achieved: 62.32%) |
| **4** | **EXP-1.3** | Sequential Mixed Learning (25% KD $\to$ 75% RL) | **H1 (Mode A)** | $1.0 \times 10^{16}$ | Uncovered verbosity bloat & KL lock (Achieved: 54.21%) |
| **5** | **EXP-1.4** | Trace-Injected & Length-Penalized GRPO | **H1 (Mode B)** | $1.0 \times 10^{16}$ | Eliminates token bloat; surpasses sequential EXP-1.3 |
| **6** | **EXP-2.1** | Reward Shaping & Objective Branching | H2 Prep | $4 \times 5.0 \times 10^{15}$ | Divergent completion styles produced |
| **7** | **EXP-2.2** | Error Orthogonality & Oracle Pass@$k$ | **H2** | N/A (Eval) | Pairwise $D_{\text{err}} \ge 0.35$, Oracle > Best |
| **8** | **EXP-3.0** | Weight-Space LoRA Merging (Linear, TIES, DARE) | **H3 Baseline** | $0.0$ (Weight Merge) | Quantifies $0$-FLOP weight synthesis vs token distillation |
| **9** | **EXP-3.1** | Peer Distillation on Plateaued Students | **H3** | $5.0 \times 10^{15}$ | $\Delta\text{Acc}_{\text{distill}} > \Delta\text{Acc}_{\text{self-RL}}$ & $\Delta\text{Acc}_{\text{merge}}$ |
| **10** | **EXP-3.2** | Bidirectional Co-Evolution & Reciprocal Transfer | **H3** | $1.0 \times 10^{16}$ | Mutual accuracy gain without forgetting |
| **11** | **EXP-4.1** | Baseline Exploration (Random & Greedy) | H4 Baseline | $1.0 \times 10^{17}$ (Total) | Baseline Pareto frontier logged |
| **12** | **EXP-4.2** | Adaptive Controller (`OptunaNSGAII` / Defect-Driven) | **H4** | $1.0 \times 10^{17}$ (Total) | Hypervolume $\text{HV}_{\text{adaptive}} > \text{HV}_{\text{baseline}}$ |
| **13** | **EXP-5.1** | Diverse Majority Voting vs Self-Consistency | **H5** | N/A (Inference) | Population voting > Single-model voting |
| **14** | **EXP-5.2** | Multi-LoRA Gated Routing & Best-of-$N$ Utility | **H5** | N/A (Inference) | Routing matches teacher at <20% cost |
| **15** | **EXP-6.1** | Out-of-Domain Generalization & Robustness | Robustness | N/A (Inference) | Diverse ensemble retains transfer edge |

---

## 2. Phase 0: Verification & Compute Baseline Calibration

### Experiment 0.1: Verifier & Baseline Zero-Shot Evaluation (EXP-0.1)
- **Sequential Position:** Step 0 (Pre-Flight Sanity).
- **Prerequisites:** `math-verify` installed; `openai/gsm8k` dataset downloaded; base model (`Qwen/Qwen2.5-Math-0.5B` or `Qwen/Qwen2.5-0.5B`) cached.
- **Hypothesis Link:** Sanity check for all experimental metrics.
- **Reason for Experiment:**
  To guarantee that the deterministic verifier callback (`math-verify`), formatting regexes, and answer parsers handle mathematical equivalence accurately without false positives/negatives, and to establish the zero-shot/few-shot baseline accuracy of the frozen base model.
- **Methodology & Setup:**
  - Evaluate base model zero-shot and 4-shot greedy decoding ($T=0$) on GSM8K test split (819 problems).
  - Run the reward suite (`multigen/eval/rewards.py`) over synthetic positive/negative math traces (fractions, negative values, LaTeX formats, units).
- **Expectation:**
  - Base model (`Qwen2.5-Math-0.5B`) achieves baseline accuracy of approximately 35% to 45% (or ~15% to 22% for `Qwen2.5-0.5B`).
  - Verifier awards exact $1.0$ for valid answers and $0.0$ for invalid answers across 100% of synthetic test cases.
- **Success Metrics:**
  - **Verifier Accuracy:** 100% accuracy on the reward unit test suite (`tests/test_rewards.py`).
  - **Parsing Robustness:** Zero unhandled parser exceptions or crashes on GSM8K test split completions.
  - **Zero-Shot Baseline:** Documented deterministic baseline accuracy for the chosen base model checkpoint.
- **Failure Metrics & Fallback Action:**
  - *Failure:* Verifier produces false positives (e.g., matching partial numbers) or false negatives on formatting differences (LaTeX `\frac` vs `/`).
  - *Fallback Action:* Enhance answer extraction regex and canonicalization filters in `multigen/eval/rewards.py` before initiating any RL rollouts.
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - `multigen/eval/rewards.py`: Implements regex answer extraction for GSM8K patterns (`#### <answer>`), reasoning tag parser (`<think>...</think><answer>...</answer>`), and verification wrappers around `math-verify` (`math_verify.parse`, `math_verify.verify`). Exposes `correctness_reward_fn(prompts, completions, answer)` and `format_reward_fn(prompts, completions)`.
    - `multigen/eval/evaluator.py`: Implements `GSM8KEvaluator` wrapping `vLLM` engine for greedy ($T=0$) and temperature-sampled zero-shot/few-shot evaluations, returning question IDs, binary correctness vectors, and latency metrics.
    - `tests/test_rewards.py`: Unit test suite validating edge cases (fractions, negative values, LaTeX formats like `\frac`, dollar signs, commas in numbers) against `math-verify`.
    - `scripts/evaluate_baseline.py`: Standalone CLI script to benchmark base models.
  - *Modifications & Integrations:*
    - Project configuration setup: Initialize `pyproject.toml` with pinned dependencies (`torch`, `transformers`, `datasets`, `math-verify`, `vllm`, `pytest`).
  - *Execution Command / CLI:*
    ```bash
    pytest tests/test_rewards.py -v
    python scripts/evaluate_baseline.py --model Qwen/Qwen2.5-Math-0.5B --dataset gsm8k --split test --output evals/exp0_1_baseline.json
    ```
  - *Artifacts & Outputs:*
    - `evals/exp0_1_baseline.json`: Zero-shot baseline accuracy, error set indices $\mathcal{E}_{\text{base}}$, and reward parsing validation log.

---

### Experiment 0.2: Compute Accounting & FLOPs Normalization Calibration (EXP-0.2)
- **Sequential Position:** Step 1.
- **Prerequisites:** EXP-0.1 passed; `multigen/accounting/flops.py` implemented.
- **Hypothesis Link:** Foundation for fair compute matching across H1, H3, and H4.
- **Reason for Experiment:**
  To calibrate theoretical FLOP calculations ($2NT$ rollout generation, $6NT$ training backpropagation, $2N_{\text{teacher}}T$ teacher inference) against actual runtime measurements, preventing bias where one training method secretly consumes more compute than another.
- **Methodology & Setup:**
  - Profile 50 steps of GRPO rollouts, 50 steps of SFT backpropagation, and 50 teacher rollout calls with `calflops` and wall-clock profilers.
  - Compute normalized tokens and theoretical FLOP counters.
- **Expectation:**
  Calculated FLOPs scale strictly linearly with token length and active parameters. Standard error across duplicate runs is negligible.
- **Success Metrics:**
  - **FLOP Linearity:** $R^2 \ge 0.999$ correlation between token counts and calculated FLOPs.
  - **Budget Enforceability:** The budget tracker terminates training within $\pm 0.5\%$ of target FLOP threshold.
- **Failure Metrics & Fallback Action:**
  - *Failure:* Divergence between parameter counts or missing KV-cache FLOP accounting in teacher rollouts.
  - *Fallback Action:* Recalibrate formula constants in `multigen/accounting/flops.py` using explicit forward/backward profiling hooks.
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - `multigen/accounting/flops.py`: Implements parameterized theoretical FLOP accounting formulas based on Chinchilla / Kaplan scaling:
      - `rollout_flops(num_params, prompt_tokens, gen_tokens)`: $2 \times N \times T_{\text{total}}$.
      - `train_step_flops(num_params, seq_len, is_lora=True)`: $6 \times N_{\text{active}} \times T_{\text{train}}$ (or $2 \times N_{\text{frozen}} \times T + 4 \times N_{\text{lora}} \times T$ for adapter backpropagation).
      - `teacher_flops(teacher_params, prompt_tokens, gen_tokens)`: $2 \times N_{\text{teacher}} \times T_{\text{teacher}}$.
    - `multigen/accounting/tracker.py`: Implements `FLOPBudgetCallback` (a Hugging Face / TRL `TrainerCallback`) that monitors cumulative FLOPs and raises `BudgetExhaustedException` when the allotted experiment budget is hit.
    - `tests/test_flops.py`: Calibration tests comparing theoretical calculations against empirical `calflops` forward/backward profiling hooks.
    - `scripts/calibrate_flops.py`: Profiling script running forward/backward passes and verifying linear scaling.
  - *Modifications & Integrations:*
    - Extend `multigen/eval/evaluator.py` to record exact inference FLOPs using `multigen/accounting/flops.py`.
  - *Execution Command / CLI:*
    ```bash
    pytest tests/test_flops.py -v
    python scripts/calibrate_flops.py --model Qwen/Qwen2.5-Math-0.5B --steps 50 --output configs/flops_calibration.json
    ```
  - *Artifacts & Outputs:*
    - `configs/flops_calibration.json`: Validated linear coefficients, FLOP multiplier constants, and runtime profiling reports.

---

## 3. Phase 1: Foundational Baselines & Mixed Learning (Hypothesis H1)

### Experiment 1.1: Standalone Reinforcement Learning (GRPO) Baseline (EXP-1.1)
- **Sequential Position:** Step 2.
- **Prerequisites:** EXP-0.1, EXP-0.2 completed.
- **Hypothesis Link:** Baseline for **H1** (Mixed Learning).
- **Reason for Experiment:**
  To establish the single-agent RL performance ceiling and learning dynamics under fixed compute ($1.0 \times 10^{16}$ FLOPs) using `trl.GRPOTrainer` with rule-based correctness rewards.
- **Methodology & Setup:**
  - Base Model: `Qwen2.5-Math-0.5B` (LoRA rank 16, alpha 32).
  - Algorithm: GRPO, group size $G=8$, learning rate $2 \times 10^{-5}$, KL penalty $\beta=0.04$.
  - Reward: $R = R_{\text{correct}} + 0.1 \times R_{\text{format}}$.
  - Compute Budget: Capped strictly at $1.0 \times 10^{16}$ FLOPs (~2,500 training steps).
  - Checkpoint Frequency: Every $2.0 \times 10^{15}$ FLOPs.
- **Expectation:**
  - Accuracy increases monotonically during the initial phases as the model learns formatting and basic arithmetic verification, then exhibits diminishing returns/plateauing.
- **Success Metrics:**
  - **Convergence:** Policy loss stabilizes; gradient norms remain bounded (< 1.0).
  - **Accuracy Improvement:** Dev accuracy improves by $\ge 12.0$ percentage points over base model zero-shot.
  - **Format Compliance:** Greater than 95% of generated completions adhere to `<think>...</think><answer>...</answer>` tags.
- **Failure Metrics & Fallback Action:**
  - *Failure:* Reward hacking (repetitive formatting tokens without correct math) or policy collapse (entropy dropping to 0 with degraded accuracy).
  - *Fallback Action:* Increase KL coefficient $\beta$ from 0.04 to 0.08 and apply max-length truncation penalties.
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - `multigen/core/schema.py`: Pydantic data models establishing structured contracts: `TrainingRecipe` (algorithm, hyperparameters, teacher ID, FLOP budget), `CapabilityProfile` (accuracy, pass@k, failure indices), and `ModelNode` (node ID, parent IDs, adapter path, cumulative FLOPs).
    - `multigen/core/population_dag.py`: NetworkX `DiGraph` wrapper managing population lineage, tracking parent-child mutations and compute ancestry. Registers root base model node `node_0`.
    - `multigen/training/grpo_runner.py`: Orchestrates RL training via `trl.GRPOTrainer` with `vLLM` rollout acceleration (`use_vllm=True`). Attaches PEFT/LoRA adapter (rank 16, alpha 32) to frozen base model, injects `correctness_reward_fn` and `format_reward_fn`, and attaches `FLOPBudgetCallback` to cap run at $1.0 \times 10^{16}$ FLOPs.
    - `configs/experiments/exp1_1_grpo_baseline.yaml`: Hyperparameters ($G=8$, lr $2 \times 10^{-5}$, $\beta=0.04$, budget $1.0 \times 10^{16}$ FLOPs).
    - `scripts/run_grpo_experiment.py`: CLI driver executing GRPO training and recording child checkpoint nodes into the lineage DAG.
  - *Modifications & Integrations:*
    - `multigen/eval/evaluator.py`: Updated to load PEFT LoRA adapters on top of the frozen base model for intermediate and post-training checkpoint evaluations.
  - *Execution Command / CLI:*
    ```bash
    python scripts/run_grpo_experiment.py --config configs/experiments/exp1_1_grpo_baseline.yaml --seed 42
    ```
  - *Artifacts & Outputs:*
    - Trained LoRA adapters: `checkpoints/exp1_1_grpo_seed{1,2,3}/adapter_model.bin`.
    - Lineage update: `population_dag.json` updated with nodes `exp1_1_grpo_seed{1,2,3}` and evaluation profiles.

---

### Experiment 1.2: Standalone Knowledge Distillation (KD) Baseline (EXP-1.2)
- **Sequential Position:** Step 3.
- **Prerequisites:** EXP-0.1, EXP-0.2 completed.
- **Hypothesis Link:** Baseline for **H1** (Mixed Learning).
- **Reason for Experiment:**
  To establish the performance ceiling and sample efficiency of pure imitation learning from a static teacher (`Qwen2.5-Math-7B-Instruct` or verified teacher rollouts) under an identical budget ($1.0 \times 10^{16}$ FLOPs, including teacher inference FLOPs).
- **Methodology & Setup:**
  - Teacher: `Qwen2.5-Math-7B-Instruct` generates $K=4$ rationales per GSM8K training problem; filtered by `math-verify` to retain only verified correct reasoning paths.
  - Student: `Qwen2.5-Math-0.5B` trained using `trl.SFTTrainer` on the verified teacher dataset.
  - Accounting: Teacher rollout FLOPs ($2 \times N_{\text{teacher}} \times T_{\text{teacher}}$) + Student SFT FLOPs ($6 \times N_{\text{student}} \times T_{\text{student}}$) $= 1.0 \times 10^{16}$ FLOPs.
- **Expectation:**
  - Rapid early accuracy gains due to high token efficiency of supervised learning on high-quality rationales, but plateaus earlier than RL due to absence of exploratory trial-and-error.
- **Success Metrics:**
  - **SFT Loss Convergence:** Cross-entropy loss converges smoothly below 0.65.
  - **Accuracy Improvement:** Final Dev accuracy exceeds base model by $\ge 14.0$ percentage points.
  - **Reasoning Coherence:** High qualitative rationale cleanliness with standard mathematical derivation steps.
- **Failure Metrics & Fallback Action:**
  - *Failure:* Student memorizes teacher stylistic artifacts without generalization, failing on held-out test splits.
  - *Fallback Action:* Filter training data for minimal token length and apply prompt diversity augmentations.
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - `multigen/training/sft_runner.py`: Wrapper around `trl.SFTTrainer` for LoRA supervised fine-tuning on question-rationale pairs with integrated FLOP accounting.
    - `multigen/training/distill_runner.py`: Two-stage distillation pipeline:
      1. Rationale Generation: Calls `vLLM` to sample $K=4$ rationales per GSM8K training problem using `Qwen2.5-Math-7B-Instruct`.
      2. Verifier Filtering: Filters traces using `math-verify` through `multigen.eval.rewards` to retain strictly correct derivations in `data/teacher_traces_gsm8k.jsonl`.
      3. Student Distillation: Invokes `sft_runner.py` on `Qwen2.5-Math-0.5B` until cumulative teacher inference + student SFT FLOPs reach $1.0 \times 10^{16}$.
    - `configs/experiments/exp1_2_distill_baseline.yaml`: Recipe configuration specifying teacher model, student LoRA hyperparameters, and unified FLOP cap.
    - `scripts/run_distill_experiment.py`: Standalone CLI running teacher trace generation, filtering, student SFT, and post-training evaluation.
  - *Modifications & Integrations:*
    - `multigen/accounting/flops.py`: Adds `track_distillation_budget()` combining teacher rollout FLOPs ($2 N_{\text{teacher}} T$) and student training FLOPs ($6 N_{\text{student}} T$).
    - `multigen/core/population_dag.py`: Supports recording external teacher metadata and teacher $\to$ student distillation edges.
  - *Execution Command / CLI:*
    ```bash
    python scripts/run_distill_experiment.py --config configs/experiments/exp1_2_distill_baseline.yaml --seed 42
    ```
  - *Artifacts & Outputs:*
    - Curated teacher dataset: `data/teacher_traces_gsm8k.jsonl`.
    - Distilled student LoRA adapters: `checkpoints/exp1_2_distill_seed{1,2,3}/adapter_model.bin`.
    - Lineage nodes logged in `population_dag.json`.

---

### Experiment 1.3: Mixed Learning (RL + Distillation) vs Baselines (EXP-1.3)
- **Sequential Position:** Step 4.
- **Prerequisites:** EXP-1.1 and EXP-1.2 completed under identical compute accounting.
- **Hypothesis Link:** **H1 (Mixed Learning)** — *Does combining RL with teacher distillation outperform RL-only training at matched compute?*
- **Reason for Experiment:**
  Directly evaluate whether hybrid training regimes outperform isolated RL and isolated KD when total active compute ($1.0 \times 10^{16}$ FLOPs) is strictly equalized.
- **Methodology & Setup:**
  - Compare two mixed variants against EXP-1.1 and EXP-1.2:
    1. **Interleaved Cycles:** Alternate between a 25% budget distillation burst and a 75% budget GRPO exploration phase.
    2. **Teacher-Regularized GRPO:** GRPO with dynamic teacher reference distributions / verified trace demonstrations injected into candidate pools.
  - Budget: Total FLOPs across teacher generation, SFT, and GRPO = $1.0 \times 10^{16}$ FLOPs.
  - Evaluation: Measure GSM8K Dev and Test accuracy every $2.0 \times 10^{15}$ FLOPs.
- **Expectation:**
  - Mixed learning leverages KD to establish a strong reasoning prior rapidly, followed by RL to explore beyond the teacher distribution, achieving superior final accuracy compared to pure RL or pure KD.
- **Success Metrics (Validating H1):**
  - **Accuracy Dominance:** Mixed learning achieves higher final test accuracy than both pure RL (EXP-1.1) and pure KD (EXP-1.2) by at least $\ge 3.5$ percentage points ($p < 0.05$ over 3 seeds).
  - **Compute Efficiency:** Reaches EXP-1.1's peak accuracy in $\le 60\%$ of EXP-1.1's compute.
  - **Pareto Superiority:** The FLOPs-vs-Accuracy curve of mixed learning strictly dominates single-method curves.
- **Failure Metrics & Fallback Action:**
  - *Failure:* Mixed learning performs worse than or equal to pure KD (indicating RL degrades teacher priors) or pure RL (indicating distillation restricts exploration).
  - *Fallback Action:* Adjust interleaving ratio (e.g., 15% KD / 85% RL) or apply soft KL regularization against the teacher rather than hard SFT steps.
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - `multigen/training/mixed_runner.py`: Coordinates hybrid post-training under equalized $1.0 \times 10^{16}$ FLOP budgets:
      - *Interleaved Mode:* Chains a 25% compute budget SFT stage using `distill_runner.py` directly into a 75% compute budget RL stage using `grpo_runner.py`, transferring adapter state seamlessly.
      - *Regularized GRPO Mode:* Integrates verified teacher traces into GRPO rollout batches as positive reference anchors with KL penalty constraint against the teacher distribution.
    - `configs/experiments/exp1_3_mixed_learning.yaml`: Config defining cycle split ratios, learning rates, and target FLOP boundaries.
    - `scripts/run_mixed_experiment.py`: Master benchmarking harness that runs EXP-1.1, EXP-1.2, and EXP-1.3 across 3 seeds under identical compute accounting and plots the comparative learning curves.
  - *Modifications & Integrations:*
    - `multigen/accounting/tracker.py`: Extended to handle phased multi-stage runs, ensuring continuous FLOP accumulation without resetting state across runner handoffs.
    - `multigen/eval/evaluator.py`: Adds intermediate periodic evaluation triggers at every $2.0 \times 10^{15}$ FLOPs checkpoint.
  - *Execution Command / CLI:*
    ```bash
    python scripts/run_mixed_experiment.py --config configs/experiments/exp1_3_mixed_learning.yaml
    ```
  - *Artifacts & Outputs:*
    - Hybrid-trained LoRA checkpoints: `checkpoints/exp1_3_mixed_{interleaved,regularized}_seed{1,2,3}/`.
    - Milestone 1 Validation Report: `evals/exp1_3_h1_benchmark.json`.
    - Comparative Pareto curve plot: `plots/h1_flops_vs_accuracy.png`.

---

### Experiment 1.4: Trace-Injected & Length-Penalized GRPO (EXP-1.4)
- **Sequential Position:** Step 5.
- **Prerequisites:** EXP-1.3 completed; root-cause analysis from Phase 1 Report (`result_reports/phase1_report.md`).
- **Hypothesis Link:** **H1 (Mode B Remediation)** — *Does dynamic in-group teacher trace injection combined with conditioned length penalties resolve the verbosity bloat and KL-locking failure modes of sequential SFT-then-RL under matched compute?*
- **Reason for Experiment:**
  In EXP-1.3, sequential chaining (25% SFT $\to$ 75% GRPO) degraded accuracy to 54.21% because imitating the 7B teacher inflated average output length by +47% (403.9 vs 274.3 tokens), consuming rollout FLOPs and cutting total RL exploration steps by 32%, while anchoring the policy KL penalty ($\beta = 0.04$) to a verbose checkpoint. EXP-1.4 tests whether dynamically injecting a single verified teacher trace into the GRPO rollout group ($1$ teacher trace + $G-1$ self-rollouts) with an explicit length penalty preserves RL exploration steps while transferring teacher guidance on hard problems.
- **Methodology & Setup:**
  - Initialize directly from the frozen base model (`Qwen3-0.6B`) to avoid SFT KL-locking.
  - During each GRPO group rollout ($G=8$), replace $1$ candidate rollout with a pre-verified concise trace from `data/teacher_traces_gsm8k.jsonl` whenever the group's empirical pass rate falls below $25\%$.
  - Apply a conditioned length penalty: $R_{\text{total}} = R_{\text{correct}} + 0.1 R_{\text{format}} - \gamma_{\text{len}} \max(0, |o| - L_{\text{target}}) \cdot R_{\text{correct}}$ with $L_{\text{target}} = 280$.
  - Budget: Strictly equalized at $1.0 \times 10^{16}$ FLOPs.
- **Expectation:**
  - Eliminates rationale verbosity bloat (mean completion length remains $<300$ tokens), restores $>550$ global RL steps, and outperforms sequential EXP-1.3 (+10% accuracy) while rivaling or exceeding standalone GRPO (64.52%).
- **Success Metrics:**
  - **Token Efficiency:** Mean output length $\le 300$ tokens on the GSM8K test split.
  - **Accuracy Recovery:** Surpasses EXP-1.3 (54.21%) by $\ge +10.0$ percentage points and exceeds standalone EXP-1.1 (64.52%).
- **Implementation & Deliverables:**
  - *Modifications & Integrations:*
    - `multigen/training/mixed_runner.py`: Implement `TeacherInjectedGRPORunner` supporting conditional in-group trace replacement and conditioned length penalty rewards.
    - `configs/experiments/exp1_4_trace_injected_grpo.yaml`: Recipe configuration for Mode B mixed learning.
  - *Artifacts & Outputs:*
    - Checkpoint adapter: `checkpoints/exp1_4_trace_injected_seed42/`.
    - Registered lineage node in `population_dag.json`.

---

## 4. Phase 2: Behavioral Diversity & Solution Space (Hypothesis H2)

### Experiment 2.1: Multi-Objective Reward Branching (EXP-2.1)
- **Sequential Position:** Step 6.
- **Prerequisites:** EXP-1.3 completed.
- **Hypothesis Link:** Precursor to **H2 (Useful Diversity)**.
- **Reason for Experiment:**
  To generate structurally and behavioral divergent model branches from the base model by varying reward functions and training hyperparameters, rather than training a single monotonic checkpoint.
- **Methodology & Setup:**
  - Train 4 distinct population branches from the base checkpoint ($5.0 \times 10^{15}$ FLOPs each):
    - **Branch A (Concise Reasoner):** $R_{\text{correct}} + \text{Length Penalty } R_{\text{len}}$ (penalizing solutions over 200 tokens).
    - **Branch B (Exploratory / CoT Reasoner):** $R_{\text{correct}} + \text{Bonus for Step Verification}$ (rewarding detailed scratchpad proofs).
    - **Branch C (Format-Strict Reasoner):** High format reward weight ($0.4 \times R_{\text{format}}$) + standard correctness.
    - **Branch D (Teacher-Biased Reasoner):** High initial KD ratio followed by temperature-scaled GRPO.
- **Expectation:**
  - All four models achieve reasonable standalone accuracy, but exhibit substantially different completion lengths, reasoning structures, and failure patterns.
- **Success Metrics:**
  - **Standalone Competence:** Every branch achieves within $5\%$ relative accuracy of the standard EXP-1.1 checkpoint.
  - **Solution Length Variance:** Mean completion length of Branch A is $\ge 40\%$ shorter than Branch B.
- **Failure Metrics & Fallback Action:**
  - *Failure:* Length penalty causes Branch A to collapse into incomplete answers without solving problems.
  - *Fallback Action:* Condition the length penalty only on correct answers ($R_{\text{len}} \times R_{\text{correct}}$) to avoid rewarding premature termination.
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - In `multigen/eval/rewards.py`:
      - `length_penalty_reward_fn`: Conditioned length penalty penalizing completions $>200$ tokens only when correct ($R_{\text{len}} = -\alpha \max(0, \text{len} - 200) \times R_{\text{correct}}$).
      - `step_verification_reward_fn`: Step-verification bonus awarding credit for explicit reasoning steps and intermediate assertions.
      - `strict_format_reward_fn`: Enhanced format verifier checking strict compliance with `<think>...</think><answer>...</answer>`.
    - Dynamic composite reward dispatcher in `multigen/training/grpo_runner.py`: Supports arbitrary linear combinations $R = w_{\text{acc}} R_{\text{correct}} + w_{\text{fmt}} R_{\text{format}} + w_{\text{len}} R_{\text{len}} + w_{\text{step}} R_{\text{step}}$.
    - `configs/experiments/exp2_1_branching.yaml`: Defines configurations for 4 distinct branches (Branch A: Concise Reasoner, Branch B: Exploratory Reasoner, Branch C: Format-Strict Reasoner, Branch D: Teacher-Biased Reasoner).
    - `scripts/run_branching_experiments.py`: Dispatcher script launching parallel or sequential branch runs under $5.0 \times 10^{15}$ FLOPs each.
  - *Modifications & Integrations:*
    - `multigen/core/schema.py`: Extend `TrainingRecipe` with `reward_weights: Dict[str, float]` and `target_behavior: str`.
  - *Execution Command / CLI:*
    ```bash
    python scripts/run_branching_experiments.py --config configs/experiments/exp2_1_branching.yaml
    ```
  - *Artifacts & Outputs:*
    - 4 distinct LoRA branch adapters: `checkpoints/exp2_1_branch_{A,B,C,D}/`.
    - Registered branch nodes in `population_dag.json`.

---

### Experiment 2.2: Error Orthogonality & Complementarity Profiling (EXP-2.2)
- **Sequential Position:** Step 6.
- **Prerequisites:** EXP-2.1 completed; evaluation outputs collected on GSM8K Dev/Test.
- **Hypothesis Link:** **H2 (Useful Diversity)** — *Do varied rewards and training objectives yield models with complementary failure modes rather than simple monotonic rankings?*
- **Reason for Experiment:**
  To test whether models trained with different objectives fail on different subsets of problems, creating an orthogonal, complementary population.
- **Methodology & Setup:**
  - Evaluate Branches A, B, C, and D on GSM8K Test Split (819 problems).
  - Extract failure problem index sets $\mathcal{E}_A, \mathcal{E}_B, \mathcal{E}_C, \mathcal{E}_D$.
  - Compute pairwise Jaccard error distance:
    $$D_{\text{err}}(M_i, M_j) = 1 - \frac{|\mathcal{E}_i \cap \mathcal{E}_j|}{|\mathcal{E}_i \cup \mathcal{E}_j|}$$
  - Calculate **Oracle Ensemble Pass@1** (Union accuracy):
    $$\text{OraclePass@1} = 1 - \frac{|\mathcal{E}_A \cap \mathcal{E}_B \cap \mathcal{E}_C \cap \mathcal{E}_D|}{N_{\text{total}}}$$
- **Expectation:**
  - Pairwise error orthogonality is high ($D_{\text{err}} \ge 0.35$), and the Oracle Ensemble Pass@1 significantly exceeds the accuracy of any individual branch.
- **Success Metrics (Validating H2):**
  - **Error Orthogonality:** Average pairwise error distance $\bar{D}_{\text{err}} \ge 0.35$ across population branches.
  - **Oracle Superiority:** Oracle Ensemble Pass@1 is at least $\ge 8.0$ percentage points higher than the single best individual branch.
  - **Complementary Clusters:** Qualitative inspection reveals distinct error types (e.g., Branch A fails on complex multi-step problems; Branch B fails on simple direct calculations due to over-reasoning).
- **Failure Metrics & Fallback Action:**
  - *Failure:* High error overlap ($D_{\text{err}} < 0.20$), where weak models simply fail on a superset of the problems solved by the strong model (monotonic failure ranking).
  - *Fallback Action:* Increase diversity pressure by introducing negative data filtering (rewarding models that solve problems other population members failed).
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - `multigen/eval/diversity.py`: Implements population error profiling and diversity metrics:
      - `extract_error_sets(eval_results)`: Extracts binary pass/fail vectors and failed problem index sets $\mathcal{E}_i \subseteq \{1, \dots, N_{\text{gsm8k}}\}$.
      - `jaccard_error_distance(err_a, err_b)`: Computes pairwise error orthogonality $D_{\text{err}}(M_i, M_j) = 1 - \frac{|\mathcal{E}_i \cap \mathcal{E}_j|}{|\mathcal{E}_i \cup \mathcal{E}_j|}$.
      - `compute_oracle_pass_at_1(error_sets_dict)`: Calculates union accuracy coverage across all combinations of population branches.
      - `compute_diversity_matrix(model_nodes)`: Constructs symmetric $N \times N$ error distance matrix.
    - `scripts/evaluate_diversity.py`: CLI tool taking arbitrary checkpoint IDs from the DAG, evaluating them on the GSM8K test split, and outputting pairwise distance heatmaps and oracle pass@1 reports.
  - *Modifications & Integrations:*
    - `multigen/core/schema.py`: Extend `CapabilityProfile` to store `error_indices: List[int]`, `mean_token_length: float`, and `pairwise_diversity_scores: Dict[str, float]`.
    - `multigen/core/population_dag.py`: Add `get_population_diversity_matrix()` querying orthogonality across active DAG nodes.
  - *Execution Command / CLI:*
    ```bash
    python scripts/evaluate_diversity.py --nodes branch_A branch_B branch_C branch_D --dataset gsm8k --split test --output evals/exp2_2_diversity.json
    ```
  - *Artifacts & Outputs:*
    - Orthogonality matrix & Oracle pass@1 report: `evals/exp2_2_diversity.json`.
    - Heatmap plot: `plots/exp2_2_error_orthogonality_heatmap.png`.
    - Empirical validation data for Hypothesis H2.

---

## 5. Phase 3: Cross-Model & Peer Learning (Hypothesis H3)

### Experiment 3.0: Weight-Space LoRA Merging vs Token-Space Synthesis Baseline (EXP-3.0)
- **Sequential Position:** Step 8.
- **Prerequisites:** EXP-1.1 (RL adapter), EXP-1.2 (KD adapter), and EXP-2.1/2.2 (diverse branch adapters) sharing the same frozen `Qwen3-0.6B` base and LoRA rank ($r=16$).
- **Hypothesis Link:** **H3 Baseline (Weight-Space vs Token-Space Transfer)** — *Can training-free parameter-space merging ($0$ FLOPs) combine orthogonal reasoning capabilities (algebraic RL + linguistic KD) into a single adapter, or does representation/format divergence require token-space cross-distillation?*
- **Reason for Experiment:**
  In Phase 1, `exp1_1_grpo_seed42` (64.52%) and `exp1_2_distill_seed42` (62.32%) demonstrated an Oracle Ensemble union of **76.42%** (+11.90% above the single best model). Because both checkpoints are parameter-efficient LoRA adapters trained from the exact same base model initialization, we can test whether their orthogonal skills can be unified directly in weight space at **$0$ training FLOPs** before spending compute on token-space distillation in EXP-3.1.
- **Methodology & Setup:**
  - Evaluate three training-free LoRA merging algorithms using `peft` (`add_weighted_adapter`):
    1. **Linear Model Soup (`linear`):** $\Delta W_{\text{merged}} = \alpha \Delta W_{\text{RL}} + (1 - \alpha) \Delta W_{\text{KD}}$ across $\alpha \in \{0.2, 0.35, 0.5, 0.65, 0.8\}$.
    2. **TIES-Merging (`ties`):** Prune low-magnitude delta parameters (density $d \in \{0.5, 0.7\}$), elect dominant parameter signs to resolve conflicts ($\text{sgn}(\Delta W_{\text{RL}}) \neq \text{sgn}(\Delta W_{\text{KD}})$), and average aligned weights.
    3. **DARE-TIES (`dare_ties`):** Randomly drop delta parameters with rate $p$ and rescale by $1/(1-p)$ prior to TIES sign consensus.
  - Evaluate each merged adapter on the GSM8K test split ($N=1,319$) and profile how many asymmetric defect problems ($\mathcal{E}_{\text{RL}} \setminus \mathcal{E}_{\text{KD}}$ and $\mathcal{E}_{\text{KD}} \setminus \mathcal{E}_{\text{RL}}$) are retained vs degraded.
- **Expectation:**
  - Naive linear averaging suffers from policy and formatting interference (`\boxed{}` vs `<answer>`), while TIES/DARE recovers a partial gain over individual parents at $0$ FLOPs, establishing a rigorous zero-compute baseline that token-space peer distillation (EXP-3.1) and co-evolution (EXP-3.2) must surpass.
- **Success Metrics:**
  - **Zero-Compute Benchmark Established:** Complete characterization of accuracy and error retention across $\alpha$ and merging methods (`linear`, `ties`, `dare_ties`).
  - **Interference Quantification:** Measure exact retention rate on symmetric successes ($\overline{\mathcal{E}_{\text{RL}}} \cap \overline{\mathcal{E}_{\text{KD}}}$) vs asymmetric successes ($\mathcal{E}_{\text{RL}} \oplus \mathcal{E}_{\text{KD}}$).
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - `multigen/training/merge_runner.py`: Implements LoRA parameter merging (`linear`, `ties`, `dare_ties`, `dare_linear`) wrapping `PeftModel.add_weighted_adapter`, with automated grid evaluation and DAG lineage registration (`algorithm="lora_merge"`, `flops=0.0`).
    - `scripts/run_lora_merge.py`: CLI script executing weight-space merge sweeps between any two or more DAG nodes and logging comparative accuracy against the Oracle union.
  - *Execution Command / CLI:*
    ```bash
    python scripts/run_lora_merge.py --adapters exp1_1_grpo_seed42 exp1_2_distill_seed42 --methods linear ties dare_ties --output evals/exp3_0_lora_merge.json
    ```
  - *Artifacts & Outputs:*
    - Merged LoRA checkpoints: `checkpoints/exp3_0_merge_{linear,ties,dare}/`.
    - Weight-space vs Oracle evaluation report: `evals/exp3_0_lora_merge.json`.

---

### Experiment 3.1: Plateaued Student Peer-Distillation Unblocking (EXP-3.1)
- **Sequential Position:** Step 9.
- **Prerequisites:** EXP-1.1 (RL plateaued checkpoint) and EXP-2.1/2.2 (diverse population checkpoints).
- **Hypothesis Link:** **H3 (Cross-Model Learning)** — *Can distillation from a peer population member unblock or accelerate a student beyond its isolated RL trajectory?*
- **Reason for Experiment:**
  To test whether peer-to-peer knowledge distillation between models of equivalent parameter scale can revive learning in an RL student that has hit an asymptotic plateau.
- **Methodology & Setup:**
  - Take the plateaued student checkpoint from EXP-1.1 ($S_0$).
  - Identify peer teacher $P$ from EXP-2.1 that exhibits highest error orthogonality relative to $S_0$.
  - Condition 1 (Control): Continue self-RL on $S_0$ for an additional $5.0 \times 10^{15}$ FLOPs.
  - Condition 2 (Treatment): Distill verified traces from peer $P$ into $S_0$ using `multigen.training.distill_runner`, followed by RL exploration, totaling $5.0 \times 10^{15}$ FLOPs.
- **Expectation:**
  - The student model absorbs solution pathways from peer $P$ that it could not discover via self-exploration, resulting in a statistically significant jump in accuracy.
- **Success Metrics (Validating H3):**
  - **Delta Accuracy Advantage:** $\Delta\text{Acc}_{\text{peer-distill}} - \Delta\text{Acc}_{\text{self-RL}} \ge +3.0$ percentage points on the held-out test split under matched compute.
  - **Defect Resolution:** $S_0$ successfully resolves at least $\ge 25\%$ of the problems that peer $P$ solved but $S_0$ previously failed.
- **Failure Metrics & Fallback Action:**
  - *Failure:* Peer distillation yields no significant gain over continued self-RL ($\Delta\text{Acc} \le 0.5\%$) or degrades the student's unique correct answers (destructive interference).
  - *Fallback Action:* Distill only on the subset of problems where the student failed and the peer succeeded, avoiding generic dataset retraining.
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - Peer-to-peer defect filtering in `multigen/training/distill_runner.py`:
      - `extract_defect_subset(student_profile, peer_profile)`: Computes asymmetric defect set $\mathcal{D}_{\text{transfer}} = \mathcal{E}_{\text{student}} \setminus \mathcal{E}_{\text{peer}}$ (problems failed by student $S_0$ but solved by peer $P$).
      - Targeted peer distillation: Generates verified traces on $\mathcal{D}_{\text{transfer}}$ using the peer LoRA checkpoint via `vLLM`, then applies `sft_runner.py` on the plateaued student adapter.
    - `scripts/run_peer_distill.py`: Execution harness running control (continued self-RL on $S_0$) vs treatment (targeted peer distillation on $\mathcal{D}_{\text{transfer}}$ + RL exploration) under matched $5.0 \times 10^{15}$ FLOPs.
  - *Modifications & Integrations:*
    - `multigen/core/population_dag.py`: Add `add_peer_distillation_edge(peer_node_id, student_node_id, recipe_metadata)` to represent peer-to-peer horizontal transfer in the NetworkX graph.
  - *Execution Command / CLI:*
    ```bash
    python scripts/run_peer_distill.py --student exp1_1_grpo_seed1 --peer exp2_1_branch_B --budget 5e15 --output evals/exp3_1_peer_distill.json
    ```
  - *Artifacts & Outputs:*
    - Checkpoint adapters: `checkpoints/exp3_1_control_self_rl/` and `checkpoints/exp3_1_treatment_peer_distill/`.
    - DAG edge linking peer to student.
    - Statistical comparison report validating Hypothesis H3 ($\Delta\text{Acc}_{\text{peer}} > \Delta\text{Acc}_{\text{self-RL}}$).

---

### Experiment 3.2: Bidirectional Co-Evolution & Reciprocal Transfer (EXP-3.2)
- **Sequential Position:** Step 8.
- **Prerequisites:** EXP-3.1 completed.
- **Hypothesis Link:** Extension of **H3**.
- **Reason for Experiment:**
  To evaluate whether two peer models can reciprocally improve each other through iterative rounds of cross-distillation without external teacher models or larger scale scaffolds.
- **Methodology & Setup:**
  - Alternate cross-distillation between Branch A and Branch B:
    - Step 1: Branch A generates verified traces for problems Branch B failed; Branch B fine-tunes on them.
    - Step 2: Branch B generates verified traces for problems Branch A failed; Branch A fine-tunes on them.
    - Step 3: Both undergo brief GRPO exploration.
  - Total Budget: $1.0 \times 10^{16}$ FLOPs distributed evenly.
- **Expectation:**
  - Both branches expand their capability frontiers, demonstrating mutual elevation without collapsing into identical models.
- **Success Metrics:**
  - **Mutual Improvement:** Both Branch A and Branch B achieve net positive accuracy deltas ($\Delta\text{Acc}_A \ge +2.0\%$, $\Delta\text{Acc}_B \ge +2.0\%$).
  - **Preserved Diversity:** Error orthogonality between A and B remains $D_{\text{err}} \ge 0.25$.
- **Failure Metrics & Fallback Action:**
  - *Failure:* Homogenization (both models converge to identical weights and identical error sets, losing diversity).
  - *Fallback Action:* Add a novelty reward bonus during the GRPO exploration phase to actively penalize generating identical token sequences.
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - `multigen/training/coevolution_runner.py`: Coordinates iterative bidirectional peer co-evolution between Branch A and Branch B:
      - Step 1: Branch A generates verified traces for $\mathcal{E}_B \setminus \mathcal{E}_A$; Branch B fine-tunes via SFT.
      - Step 2: Branch B generates verified traces for $\mathcal{E}_A \setminus \mathcal{E}_B$; Branch A fine-tunes via SFT.
      - Step 3: Joint GRPO exploration with anti-collapse KL penalty.
      - Budget allocator dividing $1.0 \times 10^{16}$ FLOPs evenly across all co-evolution stages.
    - `scripts/run_coevolution.py`: CLI driver running multi-turn reciprocal transfer and evaluating mutual improvement.
  - *Modifications & Integrations:*
    - `multigen/eval/diversity.py`: Add `diversity_drift_monitor()` callback during co-evolution to verify $D_{\text{err}}(A, B)$ remains $\ge 0.25$, guarding against model homogenization.
  - *Execution Command / CLI:*
    ```bash
    python scripts/run_coevolution.py --model_a exp2_1_branch_A --model_b exp2_1_branch_B --rounds 2 --budget 1e16
    ```
  - *Artifacts & Outputs:*
    - Reciprocally trained adapters: `checkpoints/exp3_2_coevo_round2_{A,B}/`.
    - Bidirectional lineage edges registered in `population_dag.json`.
    - Co-evolution evaluation log `evals/exp3_2_coevolution.json` confirming mutual gain without loss of orthogonality.

---

## 6. Phase 4: Adaptive Experiment Controller (Hypothesis H4)

### Experiment 4.1: Static & Random Exploration Baselines (EXP-4.1)
- **Sequential Position:** Step 9.
- **Prerequisites:** Phases 1–3 completed; controller baselines implemented in `multigen/controller/baselines.py`.
- **Hypothesis Link:** Baseline for **H4 (Adaptive Selection)**.
- **Reason for Experiment:**
  To establish rigorous reference benchmarks for population evolution under a fixed total compute budget ($B_{\text{search}} = 1.0 \times 10^{17}$ FLOPs) using non-adaptive strategies:
  1. **`RandomController`:** Selects candidate parents, recipes (`grpo`, `distill`, `mixed`), and teachers uniformly at random.
  2. **`GreedyController`:** Always selects the current highest-accuracy checkpoint and applies a fixed, standard training recipe.
- **Methodology & Setup:**
  - Run `RandomController` and `GreedyController` for 20 population trials each, logging population DAGs in NetworkX and tracking the Pareto frontier (Accuracy vs Cumulative FLOPs).
- **Expectation:**
  - GreedyController suffers premature convergence/mode collapse; RandomController wastes significant compute exploring dead-end branches.
- **Success Metrics:**
  - **Baseline Completion:** Both controller runs execute 20 valid trials without crash or state corruption.
  - **Lineage Tracking:** Accurate NetworkX DAG records with complete metadata for all 40 generated checkpoints.
- **Failure Metrics & Fallback Action:**
  - *Failure:* Checkpoint disk storage exhaustion or memory leaks during sequential trial dispatch.
  - *Fallback Action:* Activate aggressive checkpoint pruning (keeping only LoRA adapters) and garbage-collecting intermediary optimizer states.
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - `multigen/controller/baselines.py`: Implements reference controllers:
      - `RandomController`: Uniformly samples parent student node from active DAG models, recipe type (`grpo`, `distill`, `mixed`), teacher node (if applicable), and reward weights.
      - `GreedyController`: Selects the model with $\max(\text{Dev Accuracy})$ from the DAG and executes a fixed, standard GRPO recipe.
    - `multigen/controller/dispatcher.py`: General trial dispatcher accepting any proposed `TrainingRecipe`, invoking the appropriate training runner (`grpo_runner`, `distill_runner`, `mixed_runner`), running post-trial evaluation via `evaluator.py`, recording FLOPs, and registering the resulting child node in `population_dag.py`.
    - Checkpoint pruner in `multigen/core/population_dag.py`: Automatic pruning/archiving (`prune_dominated_checkpoints`) to prevent disk exhaustion over 20-trial runs by retaining only LoRA weights on the current Pareto front.
    - `scripts/run_controller_baseline.py`: CLI driver running 20 sequential trials under budget $B_{\text{search}} = 1.0 \times 10^{17}$ FLOPs for random or greedy baseline.
  - *Modifications & Integrations:*
    - `multigen/accounting/tracker.py`: Support global cumulative budget tracking across multi-trial controller sessions.
  - *Execution Command / CLI:*
    ```bash
    python scripts/run_controller_baseline.py --controller random --budget 1e17 --max-trials 20 --output runs/exp4_1_random/
    python scripts/run_controller_baseline.py --controller greedy --budget 1e17 --max-trials 20 --output runs/exp4_1_greedy/
    ```
  - *Artifacts & Outputs:*
    - Baseline trial histories: `runs/exp4_1_random/history.json`, `runs/exp4_1_greedy/history.json`.
    - Lineage DAG exports: `runs/exp4_1_{random,greedy}/lineage_dag.graphml`.
    - Baseline Pareto frontier logs for H4 comparison.

---

### Experiment 4.2: Adaptive Multi-Objective Controller vs Baselines (EXP-4.2)
- **Sequential Position:** Step 10.
- **Prerequisites:** EXP-4.1 completed; `multigen/controller/controller_study.py` implemented.
- **Hypothesis Link:** **H4 (Adaptive Selection)** — *Can a controller utilizing historical lineage and evaluation outcomes choose more productive experiments than static or random schedules?*
- **Reason for Experiment:**
  To validate whether an Optuna multi-objective controller (`NSGAIISampler` or `TPESampler`) utilizing historical lineage and capability profiles discovers superior models faster and navigates the Pareto frontier more efficiently than random or greedy selection.
- **Methodology & Setup:**
  - Execute `scripts/run_controller.py` with multi-objective Optuna controller under identical total search budget ($B_{\text{search}} = 1.0 \times 10^{17}$ FLOPs, ~20 trials).
  - Objectives:
    1. Maximize GSM8K Dev Accuracy.
    2. Maximize Diversity (mean error distance to existing Pareto front).
    3. Minimize Step FLOPs.
  - Compare Pareto Hypervolume, Area Under the Pareto Curve (AUPC), and wall-clock FLOPs required to discover a model exceeding 60% accuracy.
- **Expectation:**
  - The adaptive controller intelligently balances exploration (spawning diverse branches) and exploitation (refining promising students via mixed learning), discovering higher-performing models with lower cumulative compute.
- **Success Metrics (Validating H4):**
  - **Hypervolume Advantage:** Adaptive controller Pareto Hypervolume exceeds `RandomController` by $\ge 25\%$ and `GreedyController` by $\ge 15\%$.
  - **Discovery Acceleration:** Discovers a top-tier checkpoint ($\ge 60\%$ accuracy) in $\le 50\%$ of the compute required by the random baseline.
  - **Autonomous Recipe Selection:** Controller dynamically increases the ratio of mixed/distill recipes over pure RL as the population matures.
- **Failure Metrics & Fallback Action:**
  - *Failure:* Adaptive controller performs on par with RandomController (AUPC difference not statistically significant).
  - *Fallback Action:* Enrich state representation passed to Optuna (include ancestral branch depth and per-problem category error profiles).
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - `multigen/controller/controller_study.py`:
      - Integrates `Optuna` with `NSGAIISampler` (multi-objective genetic search) and `TPESampler`.
      - Formulates 3 explicit multi-objective fitness targets:
        1. $f_1$: Maximize GSM8K Dev Accuracy.
        2. $f_2$: Maximize Orthogonality Distance to existing Pareto frontier models ($\min_{m \in \mathcal{P}} D_{\text{err}}(\text{candidate}, m)$).
        3. $f_3$: Minimize Step FLOP consumption.
      - Search space sampler dynamically parameterized by DAG state: proposes `parent_id`, `recipe_type` (`grpo`, `distill`, `mixed`), `teacher_id`, reward weights, and token budget.
    - `scripts/run_controller.py`: Production controller CLI connecting Optuna study, `dispatcher.py`, `population_dag.py`, and real-time `wandb` / MLflow dashboard.
    - `multigen/eval/pareto.py`: Computes 2D/3D Pareto Hypervolume, Area Under the Pareto Curve (AUPC), and discovery step analysis comparing Adaptive vs Random vs Greedy.
  - *Modifications & Integrations:*
    - Extend `multigen/controller/dispatcher.py` to feedback evaluation metrics and error vectors directly to the Optuna study trial.
  - *Execution Command / CLI:*
    ```bash
    python scripts/run_controller.py --config configs/experiments/h4_adaptive_search.yaml --budget 1e17 --output runs/exp4_2_adaptive/
    ```
  - *Artifacts & Outputs:*
    - Optuna study database: `runs/exp4_2_adaptive/study.db`.
    - Pareto Hypervolume comparison plot: `plots/exp4_2_pareto_hypervolume.png`.
    - Comprehensive empirical validation report for Hypothesis H4.

---

## 7. Phase 5: Ecosystem-Level Ensembling & Deployment (Hypothesis H5)

### Experiment 5.1: Diverse Majority Voting vs Self-Consistency (EXP-5.1)
- **Sequential Position:** Step 11.
- **Prerequisites:** Population generated from EXP-4.2.
- **Hypothesis Link:** **H5 (Ecosystem-Level Value)** — *Does preserving a complementary ensemble/pool of models outperform a single highest-scoring checkpoint on real-world test distributions?*
- **Reason for Experiment:**
  To test whether an ensemble of diverse, complementary population members using majority voting outperforms self-consistency sampling from the single best checkpoint at matched inference compute ($N=5, 7, 11$ samples).
- **Methodology & Setup:**
  - **Condition A (Single Best Self-Consistency):** Sample $N$ completions ($T=0.7$) from the single highest-accuracy model in the population; select final answer via majority vote.
  - **Condition B (Diverse Population Majority Vote):** Sample 1 completion from each of the $N$ most orthogonal models on the Pareto frontier; select final answer via majority vote.
  - Test on GSM8K Test split across $N \in \{3, 5, 7, 11\}$.
- **Expectation:**
  - Because diverse population models have uncorrelated error distributions (as proven in H2), their consensus filters out idiosyncratic errors much more effectively than multiple samples from a single model.
- **Success Metrics (Validating H5):**
  - **Consensus Outperformance:** Diverse Population Majority Voting exceeds Single-Model Self-Consistency by $\ge 3.0$ percentage points at $N=5$ and $\ge 4.5$ percentage points at $N=11$.
  - **Error Uncorrelation:** The probability of simultaneous errors on identical problems is significantly lower in Condition B than Condition A.
- **Failure Metrics & Fallback Action:**
  - *Failure:* Condition B underperforms Condition A due to weaker individual models dragging down the majority vote.
  - *Fallback Action:* Apply weighted majority voting where each model's vote is weighted by its historical validation accuracy, or restrict voting to models on the Pareto frontier.
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - `multigen/eval/ensemble.py`: Implements population ensembling strategies:
      - `SingleModelSelfConsistency`: Samples $N$ stochastic completions ($T=0.7$) from the single highest-accuracy model from EXP-4.2, parses answers with `math-verify`, and takes the majority vote.
      - `DiversePopulationMajorityVoting`: Queries the top $N$ most orthogonal models from the Pareto front of the DAG, samples 1 completion from each, and aggregates votes.
      - `WeightedPopulationVoting`: Weights each model's vote by its historical validation accuracy.
    - `scripts/evaluate_ensembles.py`: Evaluates GSM8K test split across sample counts $N \in \{3, 5, 7, 11\}$ for both conditions.
  - *Modifications & Integrations:*
    - `multigen/eval/evaluator.py`: Add batch concurrent sampling across multiple LoRA adapters via `vLLM` multi-LoRA serving.
  - *Execution Command / CLI:*
    ```bash
    python scripts/evaluate_ensembles.py --dag runs/exp4_2_adaptive/lineage_dag.json --sample_sizes 3 5 7 11 --output evals/exp5_1_ensembles.json
    ```
  - *Artifacts & Outputs:*
    - Comparative accuracy curves: `plots/exp5_1_majority_vs_self_consistency.png`.
    - Ensemble evaluation report validating Hypothesis H5.

---

### Experiment 5.2: Gated Dynamic Routing vs Monolithic Checkpoint (EXP-5.2)
- **Sequential Position:** Step 12.
- **Prerequisites:** EXP-5.1 completed.
- **Hypothesis Link:** Extension of **H5**.
- **Reason for Experiment:**
  To demonstrate that a routing mechanism (e.g., lightweight classifier or Best-of-$N$ reward reranker) over the evolved model pool delivers teacher-grade performance at a fraction of the inference compute cost of running a monolithic massive model.
- **Methodology & Setup:**
  - Implement a simple router that dispatches short, simple problems to concise reasoners (Branch A) and complex multi-step problems to exploratory reasoners (Branch B).
  - Evaluate total inference tokens, end-to-end latency, and GSM8K accuracy against a monolithic `Qwen2.5-Math-7B-Instruct` model.
- **Expectation:**
  - The routed population pool achieves competitive accuracy with the 7B teacher model while reducing total inference FLOPs by $>60\%$.
- **Success Metrics:**
  - **Competitive Accuracy:** Routed ensemble attains $\ge 92\%$ of the monolithic 7B teacher model's accuracy.
  - **Efficiency Gain:** Total inference FLOPs and latency are at least $3\times$ lower than the 7B baseline.
- **Failure Metrics & Fallback Action:**
  - *Failure:* Router misclassifies problem difficulty, routing hard problems to concise models.
  - *Fallback Action:* Replace heuristic router with a Best-of-$N$ verifier reranker using `math-verify` on candidate answers.
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - `multigen/routing/router.py`:
      - `ComplexityGatedRouter`: Lightweight classifier/heuristic based on input prompt length and lexical complexity features that routes short arithmetic queries to concise reasoners (e.g. Branch A) and complex multi-step questions to deep exploratory reasoners (e.g. Branch B).
      - `VerifierReranker`: Best-of-$N$ consensus selector that ranks candidate answers across dispatched specialist models using `math-verify` cross-agreement.
    - `multigen/routing/benchmark.py`: Benchmarking harness comparing total tokens generated, end-to-end latency, and inference FLOPs of the routed population against the monolithic `Qwen2.5-Math-7B-Instruct` model.
    - `scripts/run_routing_benchmark.py`: CLI driver executing the comparative deployment benchmark on GSM8K test split.
  - *Modifications & Integrations:*
    - `multigen/accounting/flops.py`: Add inference FLOP profiler for routed requests across heterogeneous adapter models.
  - *Execution Command / CLI:*
    ```bash
    python scripts/run_routing_benchmark.py --dag runs/exp4_2_adaptive/lineage_dag.json --teacher Qwen/Qwen2.5-Math-7B-Instruct --output evals/exp5_2_routing.json
    ```
  - *Artifacts & Outputs:*
    - Efficiency & latency benchmarking report: `evals/exp5_2_routing.json`.
    - Latency/Cost-vs-Accuracy Pareto plot: `plots/exp5_2_routing_efficiency.png`.

---

## 8. Phase 6: Generalization & Out-of-Domain Robustness

### Experiment 6.1: Out-of-Domain Math Transfer (SVAMP, MATH-500) (EXP-6.1)
- **Sequential Position:** Step 13 (Final Capstone).
- **Prerequisites:** All prior experiments completed.
- **Hypothesis Link:** Generalization & Robustness validation for H1–H5.
- **Reason for Experiment:**
  To confirm that the discovered models and population ensembles have learned generalized mathematical problem-solving skills rather than overfitting to GSM8K prompt structures and numerical distributions.
- **Methodology & Setup:**
  - Evaluate base model, pure RL baseline (EXP-1.1), mixed champion (EXP-1.3), adaptive controller champion (EXP-4.2), and diverse ensemble (EXP-5.1) zero-shot on:
    - **SVAMP** (challenging word problems with adversarial perturbations).
    - **MATH-500** (diverse competition-level mathematics).
- **Expectation:**
  - The population-based models and ensembles retain their relative performance advantage over the pure RL baseline on out-of-domain benchmarks, showing superior resistance to distribution shift.
- **Success Metrics:**
  - **Positive Transfer:** Adaptive controller champion and diverse ensemble achieve statistically significant accuracy gains over the base model on SVAMP ($\ge +10\%$) and MATH-500 ($\ge +5\%$).
  - **Ensemble Robustness:** Diverse ensemble exhibits $\le 50\%$ of the performance degradation suffered by the pure RL baseline on SVAMP adversarial questions.
- **Failure Metrics & Fallback Action:**
  - *Failure:* Models show negative transfer on MATH-500, indicating GSM8K template memorization.
  - *Fallback Action:* Document domain boundaries and add cross-domain training prompts in future iterations.
- **Implementation & Deliverables:**
  - *New Modules & Classes:*
    - `multigen/eval/ood_loaders.py`: Ingestion and standardization for out-of-domain benchmarks:
      - **SVAMP**: Word problem benchmark with adversarial phrasing and variations.
      - **MATH-500**: Subsampled competition math benchmark spanning geometry, algebra, and number theory.
    - Out-of-domain verifier adaptations in `multigen/eval/rewards.py` to handle broader mathematical notation (algebraic variables, matrices, coordinate tuples).
    - `scripts/evaluate_ood_transfer.py`: Unified evaluation script testing key lineage checkpoints (Base, EXP-1.1 RL, EXP-1.3 Mixed, EXP-4.2 Champion, EXP-5.1 Ensemble) across GSM8K, SVAMP, and MATH-500.
  - *Modifications & Integrations:*
    - `multigen/eval/evaluator.py`: Add multi-dataset evaluation runner with consolidated cross-domain reporting.
  - *Execution Command / CLI:*
    ```bash
    python scripts/evaluate_ood_transfer.py --checkpoints configs/final_evaluation_checkpoints.yaml --benchmarks gsm8k svamp math500 --output evals/exp6_1_ood_transfer.json
    ```
  - *Artifacts & Outputs:*
    - Cross-domain radar chart: `plots/exp6_1_ood_transfer_radar.png`.
    - Final project synthesis table: `evals/exp6_1_synthesis.json` summarizing validation results across all hypotheses H1 through H5.

---

## 9. Comprehensive Compute Budget Allocation Matrix

All experiments are conducted within strict FLOP and token constraints to guarantee reproducibility on modest GPU infrastructure (e.g., 1–2 NVIDIA RTX 4090 or A100 GPUs):

| Phase | Experiment ID | FLOP Budget per Run | Replicates / Seeds | Cumulative FLOPs | Est. GPU Hours (A100) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 0** | EXP-0.1 (Verifier/Eval) | Inference Only | 1 | Negligible | 0.5 |
| | EXP-0.2 (FLOPs Calib) | Profiling Only | 1 | Negligible | 0.5 |
| **Phase 1** | EXP-1.1 (Pure RL Baseline) | $1.0 \times 10^{16}$ | 3 | $3.0 \times 10^{16}$ | 6.0 |
| | EXP-1.2 (Pure KD Baseline) | $1.0 \times 10^{16}$ | 3 | $3.0 \times 10^{16}$ | 4.5 |
| | EXP-1.3 (Mixed Learning - H1) | $1.0 \times 10^{16}$ | 3 | $3.0 \times 10^{16}$ | 6.0 |
| **Phase 2** | EXP-2.1 (Objective Branching) | $4 \times 5.0 \times 10^{15}$ | 1 | $2.0 \times 10^{16}$ | 4.0 |
| | EXP-2.2 (Orthogonality - H2) | Inference Only | 1 | Negligible | 1.0 |
| **Phase 3** | EXP-3.1 (Peer Distillation - H3) | $2 \times 5.0 \times 10^{15}$ | 2 | $2.0 \times 10^{16}$ | 4.0 |
| | EXP-3.2 (Co-Evolution - H3) | $1.0 \times 10^{16}$ | 1 | $1.0 \times 10^{16}$ | 2.5 |
| **Phase 4** | EXP-4.1 (Static/Random - H4) | $2 \times 1.0 \times 10^{17}$ | 1 | $2.0 \times 10^{17}$ | 36.0 |
| | EXP-4.2 (Adaptive Optuna - H4) | $1.0 \times 10^{17}$ | 1 | $1.0 \times 10^{17}$ | 18.0 |
| **Phase 5** | EXP-5.1 (Ensemble Voting - H5) | Inference Only | 3 | Negligible | 2.5 |
| | EXP-5.2 (Gated Routing - H5) | Inference Only | 1 | Negligible | 1.5 |
| **Phase 6** | EXP-6.1 (OOD Transfer) | Inference Only | 1 | Negligible | 2.0 |
| **Total** | **All Experimental Phases** | — | — | **$\sim 4.4 \times 10^{17}$** | **$\sim 89.0$ hrs** |

---

## 10. Execution Protocol & Phase-Gate Checklist

Before advancing between sequential phases, verify the corresponding gate conditions:

- [ ] **Phase 0 Gate:** `math-verify` unit tests pass with 0 false positives/negatives; theoretical FLOP calculator verified against runtime.
- [ ] **Phase 1 Gate:** EXP-1.3 shows statistically significant advantage for Mixed Learning over pure RL and KD baselines at matched compute ($1.0 \times 10^{16}$ FLOPs).
- [ ] **Phase 2 Gate:** Error orthogonality between objective-differentiated branches exceeds $D_{\text{err}} \ge 0.35$; Oracle Pass@1 exceeds single best model by $\ge 8\%$.
- [ ] **Phase 3 Gate:** Peer distillation unblocks plateaued student with $\Delta\text{Acc} \ge +3.0\%$ over continued self-RL.
- [ ] **Phase 4 Gate:** Optuna multi-objective controller demonstrates $\ge 25\%$ higher Pareto Hypervolume than random exploration baseline.
- [ ] **Phase 5 Gate:** Population majority voting beats single-model self-consistency at equal sample counts ($N=5, 11$).
- [ ] **Phase 6 Gate:** Discovered models demonstrate positive transfer and error resistance on SVAMP and MATH-500.
