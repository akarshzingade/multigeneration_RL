"""Reward functions and verifier logic for mathematical reasoning (GSM8K).

Integrates math-verify with robust regex and balanced-brace parsers, providing
TRL-compatible reward callbacks for GRPOTrainer and standardized evaluation.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, List, Optional, Union

# ==============================================================================
# Verification Backend Import
# ==============================================================================
# We attempt to import math_verify (official DeepSeek/HuggingFace math evaluation library).
# If math_verify is not present, the module falls back gracefully to a robust built-in
# symbolic/fraction/float evaluator, ensuring all tests and pipelines run reliably.
try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


def _extract_boxed_content(text: str) -> Optional[str]:
    """Extract content of the last \\boxed{...} LaTeX expression using balanced brace parsing.

    Why balanced brace parsing instead of naive regex?
      A naive regular expression such as r'\\boxed{([^}]+)}' fails on nested braces
      commonly found in LaTeX math, such as:
          \\boxed{\\frac{1}{2}}   -> matches only '\\frac{1' and breaks
          \\boxed{\\textbf{42}}  -> matches only '\\textbf{42'
      This algorithm scans character by character, tracking depth of opening '{' and
      closing '}' to extract the exact matching substring regardless of nested braces.

    Args:
        text: Input text containing LaTeX expressions.

    Returns:
        Stripped inner content of the last \\boxed{...}, or None if no valid box is found.
    """
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None

    start = idx + len(marker)
    depth = 1
    i = start
    n = len(text)

    # Walk characters forward until all opened braces are matched
    while i < n and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1

    # depth == 0 confirms balanced closing brace
    if depth == 0:
        return text[start : i - 1].strip()
    return None


def extract_answer(
    text: str,
    allow_last_number_fallback: bool = False,
) -> Optional[str]:
    """Extract candidate mathematical answer with strict delimiter priority.

    Delimiter Priority Cascade:
      1. Last <answer>...</answer> block (explicit reasoning structure)
      2. Last \\boxed{...} LaTeX block using balanced braces (standard math model format)
      3. Last GSM8K #### <answer> pattern (standard dataset exemplar format)
      4. Optional: Last numeric token (only enabled when allow_last_number_fallback=True)

    Design Rationale for RL Training:
      - During Reinforcement Learning (GRPO, EXP-1.1+), allow_last_number_fallback MUST
        remain False. If an RL agent could receive rewards for arbitrary numbers in its
        scratchpad, it would engage in 'reward hacking' (emitting dozens of numbers
        in prose until hitting the gold value by chance).
      - During frozen zero-shot baseline evaluation (EXP-0.1), allow_last_number_fallback=True
        is used to record 'permissive_accuracy', diagnosing raw mathematical skill
        independently of formatting compliance.

    Args:
        text: Generated completion string.
        allow_last_number_fallback: Whether to fall back to the final number if no tags exist.

    Returns:
        Extracted answer string, or None if no valid answer was found.
    """
    if not text or not isinstance(text, str):
        return None

    # Priority 1: Match <answer>(.*?)</answer> tags (case-insensitive)
    answer_tag_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    matches = list(answer_tag_pattern.finditer(text))
    if matches:
        extracted = matches[-1].group(1).strip()
        if extracted:
            return extracted

    # Priority 2: Match \boxed{...} with balanced brace depth
    boxed = _extract_boxed_content(text)
    if boxed is not None and boxed != "":
        return boxed

    # Priority 3: Match GSM8K legacy #### marker
    hash_matches = list(re.finditer(r"####\s*([^\n]+)", text))
    if hash_matches:
        extracted = hash_matches[-1].group(1).strip()
        if extracted:
            return extracted

    # Priority 4: Optional permissive fallback to the last numeric token
    if allow_last_number_fallback:
        # Regex matches integers, decimals, fractions, and values with thousands separators (1,200)
        num_pattern = re.compile(
            r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?"
        )
        nums = list(num_pattern.finditer(text))
        if nums:
            return nums[-1].group(0).strip()

    return None


def _normalize_math_string(s: str) -> str:
    """Pre-clean common formatting noise without altering mathematical meaning.

    Normalizations applied:
      - Strips '####' prefix if present from raw dataset entries.
      - Strips leading and trailing whitespace.
      - Removes trailing sentence periods (e.g. "42." -> "42").
      - Strips currency dollar signs (e.g. "$1250" -> "1250").
      - Normalizes thousands commas between digits (e.g. "1,250" -> "1250").
      - Retains negative signs, decimals, and fractional slashes.
    """
    if not s:
        return ""

    # Strip raw GSM8K prefix if passed
    if "####" in s:
        s = s.split("####")[-1]

    s = s.strip()

    # Remove sentence-ending punctuation while preserving decimals or ellipsis
    if s.endswith(".") and not s.endswith(".."):
        s = s[:-1].strip()

    # Strip currency symbols
    s = s.replace("$", "").strip()

    # Remove commas between digits (e.g., 1,200 -> 1200)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)

    return s


def _builtin_verify_fallback(pred_str: str, gold_str: str) -> bool:
    """Builtin verification fallback using Python standard library Fraction and float parsing.

    Used when math-verify is unavailable or encounters unparseable syntax.
    Evaluates:
      1. Exact string equality
      2. Exact integer equivalence ('42' == '42')
      3. Floating-point equivalence within 1e-6 tolerance ('0.5' == '0.5000')
      4. Fraction equivalence: '\\frac{1}{2}' == '1/2' == '0.5'
    """
    if pred_str == gold_str:
        return True

    # 1. Integer comparison
    try:
        return int(pred_str) == int(gold_str)
    except ValueError:
        pass

    # 2. Floating point comparison
    try:
        p_val = float(pred_str)
        g_val = float(gold_str)
        return abs(p_val - g_val) < 1e-6
    except ValueError:
        pass

    # 3. Fraction comparison (parses both LaTeX \frac{a}{b} and standard a/b)
    def to_fraction(expr: str) -> Optional[Fraction]:
        expr = expr.strip()
        # Parse LaTeX \frac{numerator}{denominator}
        frac_match = re.fullmatch(r"\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", expr)
        if frac_match:
            denom = int(frac_match.group(2))
            if denom != 0:
                return Fraction(int(frac_match.group(1)), denom)

        # Parse standard a/b
        frac_match2 = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", expr)
        if frac_match2:
            denom = int(frac_match2.group(2))
            if denom != 0:
                return Fraction(int(frac_match2.group(1)), denom)

        # Parse standard integer or decimal string as Fraction
        try:
            return Fraction(expr)
        except (ValueError, ZeroDivisionError):
            return None

    p_frac = to_fraction(pred_str)
    g_frac = to_fraction(gold_str)
    if p_frac is not None and g_frac is not None:
        return p_frac == g_frac

    return False


def verify_math_answer(pred: Optional[str], gold: str) -> float:
    """Verify extracted candidate answer against gold answer for mathematical equivalence.

    Verification Flow:
      1. Returns 0.0 immediately if pred is None or empty.
      2. Cleans whitespace, currency ($), and thousands commas via _normalize_math_string.
      3. Passes to math_verify.verify (DeepSeek/HuggingFace library) if installed.
      4. Falls back to _builtin_verify_fallback (standard library Fraction/float).
      5. Guaranteed exception-safe: wraps all external parser calls in try/except and returns 0.0.

    Args:
        pred: Extracted model prediction string (or None).
        gold: Ground-truth target answer string.

    Returns:
        1.0 if mathematically equivalent, 0.0 otherwise.
    """
    if pred is None:
        return 0.0

    pred_clean = _normalize_math_string(pred)
    gold_clean = _normalize_math_string(gold)

    if not pred_clean or not gold_clean:
        return 0.0

    # Step 1: Use official math-verify library if available
    if HAS_MATH_VERIFY:
        try:
            p_parsed = mv_parse(pred_clean)
            g_parsed = mv_parse(gold_clean)
            if mv_verify(g_parsed, p_parsed):
                return 1.0
        except Exception:
            # Ignore library parse errors and try builtin fallback
            pass

    # Step 2: Fallback to builtin fraction/float equivalence parser
    try:
        if _builtin_verify_fallback(pred_clean, gold_clean):
            return 1.0
    except Exception:
        pass

    return 0.0


def _extract_completion_text(completion: Union[str, List[dict], Any]) -> str:
    """Extract completion string handling both plain text and TRL conversational message lists.

    In Hugging Face TRL (GRPOTrainer), completions may be passed as:
      - Plain strings: "completion text"
      - Conversational lists: [{"role": "assistant", "content": "completion text"}]
    This helper normalizes both formats into a single plain text string.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) > 0:
        last_turn = completion[-1]
        if isinstance(last_turn, dict) and "content" in last_turn:
            return str(last_turn["content"])
    return str(completion)


def correctness_reward_fn(
    prompts: Any,
    completions: List[Union[str, List[dict], Any]],
    answer: Union[str, List[str]],
    **kwargs: Any,
) -> List[float]:
    """TRL-compatible correctness reward callback for GRPOTrainer.

    Interface:
      Matches Hugging Face TRL GRPOTrainer reward callback signature:
        reward_fn(prompts, completions, answer, **kwargs) -> list[float]

    Behavior:
      - Strictly extracts answers via <answer>, \\boxed{}, or #### (no numeric fallback).
      - Computes binary reward (1.0 for mathematically correct, 0.0 otherwise).
      - Handles both scalar target answers and per-sample target answer lists.
    """
    rewards: List[float] = []

    # Broadcast scalar answer across completions if needed
    if isinstance(answer, (list, tuple)):
        gold_answers = list(answer)
    else:
        gold_answers = [str(answer)] * len(completions)

    for comp, gold in zip(completions, gold_answers):
        comp_text = _extract_completion_text(comp)
        pred = extract_answer(comp_text, allow_last_number_fallback=False)
        reward = verify_math_answer(pred, str(gold))
        rewards.append(reward)

    return rewards


def format_reward_fn(
    prompts: Any,
    completions: List[Union[str, List[dict], Any]],
    **kwargs: Any,
) -> List[float]:
    """TRL-compatible format reward callback for GRPOTrainer.

    Enforces strict adherence to the structured output convention:
      ^<think>non-empty reasoning</think><answer>non-empty answer</answer>$

    Returns:
      1.0 if the completion strictly satisfies the tag pattern with non-empty contents,
      0.0 otherwise.
    """
    # Regex enforces optional outer whitespace, non-empty <think>, and non-empty <answer>
    pattern = re.compile(
        r"^\s*<think>(?P<think>.+?)</think>\s*<answer>(?P<ans>.+?)</answer>\s*$",
        re.DOTALL | re.IGNORECASE,
    )

    rewards: List[float] = []
    for comp in completions:
        comp_text = _extract_completion_text(comp)
        m = pattern.match(comp_text)
        if m and m.group("think").strip() and m.group("ans").strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards
