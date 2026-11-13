# Experiment 0.1: Verifier Calibration & Baseline Zero-Shot Evaluation (`EXP-0.1`)

- **Sequential Position:** Step 0 (Pre-Flight Sanity)
- **Status:** Implemented (Unit Tests Passing)
- **Parent Plans:** [Sequential Experiment Plan](../experiment_plan.md) | [Implementation Plan](../implementaiton_plan.md)

---

## 1. Objective & Scope

`EXP-0.1` is the foundational gate before any FLOP accounting (`EXP-0.2`) or RL/distillation training (`EXP-1.x`). It has three concrete goals:

1. **Bootstrap the repository skeleton:** Initialize `pyproject.toml`, the `multigen` package layout, `tests/`, `scripts/`, and `evals/`.
2. **Calibrate and verify the deterministic reward/parser pipeline:** Ensure `multigen/eval/rewards.py` (wrapping `math-verify`) extracts answers and evaluates mathematical equivalence with **0 unhandled exceptions** and **100% accuracy** on a curated edge-case unit test suite (`tests/test_rewards.py`).
3. **Establish the frozen base-model baseline:** Evaluate the chosen base model on GSM8K (0-shot and 4-shot, greedy decoding $T=0$) to record baseline accuracy, format compliance rate, and the initial failure index set $\mathcal{E}_{\text{base}}$ in `evals/exp0_1_baseline.json`.

---

## 2. Implementation Details

### 2.1 Repository Bootstrap (`pyproject.toml` & Layout)

Create the minimal package structure required for `EXP-0.1`:

```text
multigen-rl/
├── pyproject.toml
├── multigen/
│   ├── __init__.py
│   └── eval/
│       ├── __init__.py
│       ├── rewards.py
│       └── evaluator.py
├── scripts/
│   └── evaluate_baseline.py
├── tests/
│   └── test_rewards.py
└── evals/                      # Created at runtime for JSON outputs
```

**`pyproject.toml` requirements:**
- Build system: `setuptools>=68.0` or `hatchling`, Python `>=3.10`.
- Core dependencies for `EXP-0.1`:
  - `torch`
  - `transformers`
  - `datasets`
  - `math-verify`
  - `pydantic>=2.0`
  - `pyyaml`
- Inference backend:
  - `vllm` (GPU/Linux primary backend; see [Section 4.3](#43-compute-environment--inference-backend)).
- Dev/test dependencies:
  - `pytest`

---

### 2.2 Reward & Verifier Module (`multigen/eval/rewards.py`)

This module will be reused directly by `trl.GRPOTrainer` in `EXP-1.1+` and by `GSM8KEvaluator` in `EXP-0.1`. Design its function signatures to be compatible with both standalone evaluation and `trl` reward callbacks.

#### A. Answer Extraction (`extract_answer(text: str) -> Optional[str]`)
Implement an ordered extraction pipeline with clear priority so structured tags take precedence over legacy patterns:

1. **`<answer>...</answer>` tag:** Match the *last* `<answer>(.*?)</answer>` block (case-insensitive, `re.DOTALL`).
2. **`\boxed{...}` LaTeX:** Use a balanced-brace parser (not a naive regex, which breaks on nested braces like `\boxed{\frac{1}{2}}`) to extract the *last* `\boxed{...}` content.
3. **GSM8K `####` marker:** Match `####\s*(.+?)(?:\n|$)` and take the last occurrence.
4. **Optional fallback (baseline diagnostics only):** If `allow_last_number_fallback=True`, extract the final numeric token; otherwise return `None`. Keep this `False` by default for `correctness_reward_fn` (RL-safe), and use it only to compute the secondary `permissive_accuracy` diagnostic in `EXP-0.1`.

#### B. Canonicalization & `math-verify` Wrapper (`verify_math_answer(pred: Optional[str], gold: str) -> float`)
1. If `pred is None` or `pred.strip() == ""`, return `0.0`.
2. Pre-clean common noise without altering mathematical meaning:
   - Strip surrounding whitespace, trailing periods, and currency symbols (`$`), and normalize thousands commas (e.g., `1,200` $\to$ `1200`).
   - Do **not** strip arbitrary trailing unit words (e.g., `12 apples` remains strict `0.0` unless `math-verify` handles it) so RL learns to emit pure numeric answers.
   - Extract gold answer from raw GSM8K `answer` field if it still contains `#### <gold>`.
3. Call `math_verify.parse` and `math_verify.verify`:
   ```python
   from math_verify import parse, verify

   try:
       gold_parsed = parse(gold_clean)
       pred_parsed = parse(pred_clean)
       return 1.0 if verify(gold_parsed, pred_parsed) else 0.0
   except Exception:
       # Never crash training or evaluation on malformed LaTeX/text
       return 0.0
   ```
4. **Guard against partial-number false positives:** Ensure `parse` does not match arbitrary numbers embedded in prose when `extract_answer` is used first.

#### C. `trl`-Compatible Reward Callbacks
- **`correctness_reward_fn(prompts, completions, answer, **kwargs) -> list[float]`**
  - Normalize `completions` (handle both `list[str]` and `trl` conversational format `list[list[dict]]` where assistant text is `comp[-1]["content"]`).
  - For each `(completion, gold)` pair, extract the predicted answer and return `verify_math_answer(pred, gold)`.
- **`format_reward_fn(prompts, completions, **kwargs) -> list[float]`**
  - Check strict compliance with `^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$` (`re.DOTALL`).
  - Return `1.0` if matched (with non-empty `<think>` and `<answer>`), else `0.0`.

---

### 2.3 Dataset & Prompting Utilities (inside `multigen/eval/evaluator.py`)

1. **Dataset Loading:**
   - Load `openai/gsm8k` (`name="main"`) via Hugging Face `datasets`.
   - Parse each example's `answer` field to isolate `gold_answer` (the substring after `####`).
   - **Deterministic Split:** Carve `dev` (500 examples) deterministically from the `train` split using a fixed seed (e.g., `seed=42`), leaving:
     - `train`: **6,973** examples
     - `dev`: **500** examples
     - `test`: **1,319** examples (full official GSM8K test set)
2. **Prompt Formatting:**
   - **Zero-shot prompt:** Instruct the model to reason inside `<think>...</think>` and output the final answer inside `<answer>...</answer>`.
   - **4-shot prompt:** Prepend 4 fixed GSM8K training examples formatted with the same reasoning/answer structure so the frozen base model has an in-context demonstration of the target output format.

---

### 2.4 Baseline Evaluator (`multigen/eval/evaluator.py`)

Implement `GSM8KEvaluator` and a Pydantic `BaselineEvalResult` schema:

- **Inputs:**
  - `model_name_or_path: str`
  - `adapter_path: Optional[str] = None` (unused in `EXP-0.1`, included for `EXP-1.1+` reuse)
  - `backend: Literal["vllm", "transformers"] = "vllm"`
  - `temperature: float = 0.0` (greedy)
  - `max_tokens: int = 512`
- **Execution:**
  - Batch-generate completions for the target split.
  - Score every completion with `correctness_reward_fn` and `format_reward_fn`.
  - Record per-example results: `question_id`, `gold_answer`, `extracted_answer`, `correct` (`bool`), `format_valid` (`bool`), `completion_tokens` (`int`), and raw `completion` text (optional/truncated for debugging).
- **Outputs (`BaselineEvalResult`):**
  - `model_name`, `split`, `num_shots`, `total_examples`
  - `accuracy`: `float` (`correct_count / total_examples`, strict tag/marker extraction)
  - `permissive_accuracy`: `float` (diagnostic accuracy with last-number fallback enabled)
  - `format_compliance_rate`: `float`
  - `error_indices`: `list[int]` ($\mathcal{E}_{\text{base}}$, indices of failed problems under strict extraction)
  - `mean_completion_tokens`: `float`
  - `wall_time_seconds`: `float`
  - `parser_exceptions`: `int` (must be `0`)

---

### 2.5 Unit Test Suite (`tests/test_rewards.py`)

Write deterministic `pytest` unit tests covering:

| Category | Example Input / Scenario | Expected Result |
| :--- | :--- | :--- |
| **Exact integer** | `<answer>42</answer>` vs `42` | `1.0` |
| **Negative values** | `<answer>-17</answer>` vs `-17` | `1.0` |
| **Fractions vs decimals** | `<answer>\frac{1}{2}</answer>` vs `0.5` | `1.0` |
| **LaTeX `\boxed{}`** | `Therefore \boxed{\frac{3}{4}}.` vs `0.75` | `1.0` |
| **Commas & currency** | `<answer>$1,250</answer>` vs `1250` | `1.0` |
| **Units / trailing text** | `<answer>12 apples</answer>` vs `12` | Verify behavior & document |
| **Partial-number trap** | `Step 1: 10. <answer>12</answer>` vs `10` | `0.0` (must not match `10`) |
| **Empty / malformed** | `""`, `<answer></answer>`, `\boxed{\frac{1}{` | `0.0` (no exception) |
| **Format reward (valid)** | `<think>2+2=4</think><answer>4</answer>` | `1.0` |
| **Format reward (invalid)** | `<think></think><answer>4</answer>` or missing tags | `0.0` |
| **TRL batch format** | Conversational `list[list[dict]]` + `list[str]` | Identical float lists |

---

### 2.6 CLI Driver (`scripts/evaluate_baseline.py`)

- **Arguments:**
  - `--model` (default: `Qwen/Qwen3-0.6B`)
  - `--dataset` (default: `gsm8k`)
  - `--split` (`dev` or `test`, default: `test`)
  - `--num-shots` (`0` or `4`, default: `0`)
  - `--max-tokens` (default: `512`)
  - `--backend` (`vllm` or `transformers`, default: `vllm`; use `transformers` for local macOS CPU/MPS runs)
  - `--output` (default: `evals/exp0_1_baseline.json`)
- **Behavior:** Runs `GSM8KEvaluator`, prints summary metrics (`accuracy`, `format_compliance_rate`, `error_count`, `wall_time_seconds`), and writes the full JSON artifact to `--output`.

---

## 3. Execution & Acceptance Gate

### Commands
```bash
# 1. Run verifier unit tests (must be 100% pass)
pytest tests/test_rewards.py -v

# 2. Run 0-shot and 4-shot baseline evaluations
python scripts/evaluate_baseline.py --model Qwen/Qwen3-0.6B --split test --num-shots 0 --output evals/exp0_1_baseline_0shot.json
python scripts/evaluate_baseline.py --model Qwen/Qwen3-0.6B --split test --num-shots 4 --output evals/exp0_1_baseline_4shot.json
```

### Decision Gate (Pass Criteria)
- [ ] `pytest tests/test_rewards.py` passes **100%** of test cases.
- [ ] **0 unhandled parser exceptions** across the entire GSM8K evaluation split.
- [ ] Baseline accuracy and $\mathcal{E}_{\text{base}}$ are saved to `evals/exp0_1_baseline_*.json`.

---

## 4. Finalized Design Decisions

| # | Area | Decision |
| :---: | :--- | :--- |
| **1** | **Base Model** | `Qwen/Qwen3-0.6B` |
| **2** | **GSM8K Split** | Carve `dev (500)` from `train` (`seed=42`) $\to$ `6,973` train / `500` dev / `1,319` full official test |
| **3** | **Inference Backend** | `vllm` default (GPU) + `transformers` fallback (CPU/MPS for local macOS) |
| **4** | **Prompting Format** | `<think>...</think><answer>...</answer>` for both 0-shot and 4-shot |
| **5** | **Extraction Policy** | Strict (`<answer>`, `\boxed{}`, `####` only) for `correctness_reward_fn`; log `permissive_accuracy` as a secondary diagnostic in `EXP-0.1` |
| **6** | **Unit-Bearing Answers** | *(Recommended)* Strict (`0.0` on trailing unit words; only strip `$` and `,` separators) so RL trains clean numeric outputs |
