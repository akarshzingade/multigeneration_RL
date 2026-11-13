"""Standardized GSM8K baseline evaluator supporting vLLM and Transformers backends.

Provides deterministic train/dev/test dataset partitioning, structured few-shot prompt
formatting with <think>...</think><answer>...</answer> tags, and greedy zero-shot/few-shot
evaluation harnesses measuring strict accuracy, permissive diagnostic accuracy, and format compliance.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from multigen.eval.rewards import (
    extract_answer,
    format_reward_fn,
    verify_math_answer,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Fixed 4-Shot Exemplars for In-Context Learning
# ==============================================================================
# These exemplars are drawn from the GSM8K training set and explicitly rewritten
# into the standardized <think>...</think><answer>...</answer> format.
# They serve two key purposes:
#   1. Teach frozen base models (e.g. Qwen3-0.6B) the target structural output schema.
#   2. Demonstrate step-by-step mathematical reasoning prior to emitting the final number.
GSM8K_4SHOT_EXEMPLARS = [
    {
        "question": (
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
            "clips in May. How many clips did Natalia sell altogether in April and May?"
        ),
        "think": (
            "Natalia sold 48 clips in April. In May, she sold half as many clips as in April, "
            "so she sold 48 / 2 = 24 clips. Altogether in April and May, she sold 48 + 24 = 72 clips."
        ),
        "answer": "72",
    },
    {
        "question": (
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
            "babysitting. How much did she earn?"
        ),
        "think": (
            "Weng earns $12 for 60 minutes of babysitting. This means she earns $12 / 60 = $0.20 per minute. "
            "For 50 minutes of work, she earns 50 * $0.20 = $10."
        ),
        "answer": "10",
    },
    {
        "question": (
            "Betty is saving money for a new wallet which costs $100. Betty has only half of the money "
            "she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as "
            "much as her parents. How much more money does Betty need to buy the wallet?"
        ),
        "think": (
            "The wallet costs $100. Betty currently has 100 / 2 = $50. Her parents give her $15. "
            "Her grandparents give her twice as much as her parents, which is 15 * 2 = $30. "
            "Total money Betty has now is 50 + 15 + 30 = $95. She still needs 100 - 95 = $5 to buy the wallet."
        ),
        "answer": "5",
    },
    {
        "question": (
            "A deep-sea monster rises from the waters once every 100 years to feast on a ship and then "
            "returns to the deep. The monster has feasted on 13 ships so far. How many years has the monster lived?"
        ),
        "think": (
            "The monster feasted on 13 ships. Since it rises once every 100 years for each ship, "
            "13 feasting cycles take 13 * 100 = 1300 years."
        ),
        "answer": "1300",
    },
]


@dataclass
class BaselineEvalResult:
    """Structured evaluation report containing summary metrics and per-problem diagnostics.

    Attributes:
        model_name: HuggingFace model ID or local checkpoint path.
        split: Evaluated dataset split ('dev' or 'test').
        num_shots: In-context few-shot count (0 for zero-shot, 4 for 4-shot).
        total_examples: Total number of evaluated problems.
        accuracy: Primary metric. Strict evaluation requiring valid <answer>, \\boxed{},
            or #### delimiters. Answers outside delimiters are scored 0.0.
        permissive_accuracy: Diagnostic metric. Falls back to the last numeric token in the
            completion if no delimiters are present. Used to decouple math capability from formatting.
        format_compliance_rate: Percentage of completions that strictly satisfy
            ^<think>...</think><answer>...</answer>$ with non-empty content.
        error_indices: Zero-based indices of failed problems (E_base). Used by Phase 2 (EXP-2.2)
            to compute pairwise Jaccard error distance and error orthogonality across models.
        mean_completion_tokens: Average generated tokens per solution.
        wall_time_seconds: Total wall-clock time for batch inference and verification.
        parser_exceptions: Count of unhandled exceptions during parsing (must be 0 for pass gate).
        per_example_details: Optional granular log of inputs, completions, extracted answers, and scores.
    """

    model_name: str
    split: str
    num_shots: int
    total_examples: int
    accuracy: float
    permissive_accuracy: float
    format_compliance_rate: float
    error_indices: List[int] = field(default_factory=list)
    mean_completion_tokens: float = 0.0
    wall_time_seconds: float = 0.0
    parser_exceptions: int = 0
    per_example_details: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result dataclass to a plain dictionary for serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize evaluation result to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def format_gsm8k_prompt(question: str, num_shots: int = 0) -> str:
    """Build standardized evaluation prompt with explicit tag formatting instructions.

    Args:
        question: The GSM8K math question string.
        num_shots: Number of few-shot demonstrations to prepend (0 to 4).

    Returns:
        Formatted prompt string instructing the model to output reasoning within
        <think>...</think> and the final numerical answer within <answer>...</answer>.
    """
    # System instruction establishing reasoning guidelines and formatting tags
    system_instruction = (
        "Solve the following math problem step by step.\n"
        "Write your chain of thought reasoning inside <think>...</think> tags.\n"
        "Write your final numeric answer inside <answer>...</answer> tags.\n\n"
    )

    # Prepend up to 4 structured in-context exemplars
    exemplars_text = ""
    if num_shots > 0:
        selected_shots = GSM8K_4SHOT_EXEMPLARS[:num_shots]
        for shot in selected_shots:
            exemplars_text += (
                f"Problem: {shot['question']}\n"
                f"<think>{shot['think']}</think>\n"
                f"<answer>{shot['answer']}</answer>\n\n"
            )

    return f"{system_instruction}{exemplars_text}Problem: {question}\n"


def split_gsm8k_dataset(
    dataset_dict: Any,
    dev_size: int = 500,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    """Partition official GSM8K into train (6,973), dev (500), and test (1,319).

    Design Rationale:
      - The official 'openai/gsm8k' dataset has 7,473 train examples and 1,319 test examples.
      - To evaluate on the full, official 1,319 test set (ensuring external comparability with
        published literature), we carve the 500-example validation/dev split deterministically
        from the training split using a fixed random seed.
      - Final split sizes:
          * train: 6,973 problems (for RL rollouts and distillation SFT in Phase 1+)
          * dev:     500 problems (for Optuna controller objective optimization in Phase 4)
          * test:  1,319 problems (held-out final benchmark for H1-H5 validation)

    Args:
        dataset_dict: HuggingFace DatasetDict with 'train' and 'test' keys.
        dev_size: Number of examples to allocate to dev split (default: 500).
        seed: Random seed for deterministic shuffling of the training set.

    Returns:
        Dictionary mapping 'train', 'dev', and 'test' to lists of dicts containing
        'question', 'gold_answer', and 'raw_answer'.
    """
    raw_train = list(dataset_dict["train"])
    raw_test = list(dataset_dict["test"])

    # Perform deterministic pseudo-random shuffling of train indices
    import random

    rng = random.Random(seed)
    train_indices = list(range(len(raw_train)))
    rng.shuffle(train_indices)

    # Carve the first `dev_size` indices for validation, remaining for training
    dev_idx_set = set(train_indices[:dev_size])
    final_dev = [raw_train[i] for i in dev_idx_set]
    final_train = [raw_train[i] for i in range(len(raw_train)) if i not in dev_idx_set]

    def extract_gold(ans_field: str) -> str:
        """Extract the final numerical ground-truth ("gold") answer from raw GSM8K answer text.

        In the official GSM8K dataset, each answer entry contains the full human solution
        rationale followed by a '####' marker and the final numerical result.
        Example raw answer:
            "Natalia sold 48 clips in April... 48 + 24 = 72.\n#### 72"

        This helper:
            1. Splits on the '####' delimiter.
            2. Takes the final segment (the pure target answer).
            3. Strips surrounding whitespace and newlines.
            4. Falls back to stripping the full string if '####' is absent.

        In NLP literature, 'gold' refers to the verified ground-truth reference target.
        """
        if "####" in ans_field:
            return ans_field.split("####")[-1].strip()
        return ans_field.strip()

    def process_split(data_list: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract question and gold answer pairs for each problem in the split."""
        processed = []
        for item in data_list:
            processed.append({
                "question": item["question"].strip(),
                "gold_answer": extract_gold(item["answer"]),
                "raw_answer": item["answer"],
            })
        return processed

    return {
        "train": process_split(final_train),
        "dev": process_split(final_dev),
        "test": process_split(raw_test),
    }


class GSM8KEvaluator:
    """Inference and scoring engine executing greedy zero-shot/few-shot GSM8K evaluation.

    Supports two backends:
      1. 'vllm': High-throughput batched inference for Linux + CUDA GPU environments.
      2. 'transformers': Fallback backend for local testing on macOS (MPS / CPU).
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        backend: Literal["vllm", "transformers"] = "vllm",
        max_tokens: int = 512,
        temperature: float = 0.0,
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        """Initialize the evaluator with model specifications and inference configurations.

        Args:
            model_name_or_path: HuggingFace model ID (e.g. 'Qwen/Qwen3-0.6B') or local weights dir.
            adapter_path: Optional path to a PEFT LoRA adapter checkpoint (used in EXP-1.1+).
            backend: Inference backend ('vllm' or 'transformers').
            max_tokens: Generation token ceiling per problem.
            temperature: Sampling temperature (0.0 for deterministic greedy decoding).
            batch_size: Batch size for HuggingFace transformers backend.
            device: Explicit device string ('cuda', 'mps', or 'cpu'). Auto-detected if None.
        """
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.backend = backend
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device or self._detect_device()

        # Initialize the selected inference engine
        self._init_backend()

    @staticmethod
    def _detect_device() -> str:
        """Detect best available compute hardware: CUDA GPU -> Apple Silicon MPS -> CPU."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _init_backend(self) -> None:
        """Initialize inference engine (vLLM engine or HuggingFace Transformers pipeline)."""
        # Primary backend: vLLM (high throughput on Linux/CUDA)
        if self.backend == "vllm":
            try:
                from vllm import LLM, SamplingParams

                logger.info("Initializing vLLM engine for %s", self.model_name_or_path)
                self.llm = LLM(
                    model=self.model_name_or_path,
                    trust_remote_code=True,
                    max_model_len=2048,
                )
                self.sampling_params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                # Graceful fallback: switch to transformers if vLLM fails or is on macOS
                logger.warning(
                    "Failed to initialize vLLM (%s). Falling back to transformers backend.", e
                )
                self.backend = "transformers"

        # Fallback backend: HuggingFace Transformers (portable across Linux, macOS MPS, CPU)
        if self.backend == "transformers":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(
                "Initializing Transformers engine for %s on device=%s",
                self.model_name_or_path,
                self.device,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            # Ensure pad token is defined for batch generation
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            import torch

            # Use half-precision float16 on accelerators, full precision on CPU
            dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            )

            # Optionally attach trained LoRA adapter (used in Phase 1+ checkpoints)
            if self.adapter_path:
                from peft import PeftModel

                logger.info("Loading LoRA adapter from %s", self.adapter_path)
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

            self.model.to(self.device)
            self.model.eval()

    def generate_batch(self, prompts: List[str]) -> Tuple[List[str], List[int]]:
        """Generate model completions and record generated token lengths for a list of prompts.

        Args:
            prompts: List of prompt strings.

        Returns:
            Tuple of (completions, token_counts) where each completion is the generated text
            and each token count is the number of newly generated tokens.
        """
        # Generation via vLLM
        if self.backend == "vllm":
            outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
            completions = [out.outputs[0].text for out in outputs]
            token_counts = [len(out.outputs[0].token_ids) for out in outputs]
            return completions, token_counts

        # Generation via HuggingFace Transformers
        import torch

        completions = []
        token_counts = []

        # Process in batches to manage accelerator memory
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]

            # Tokenize batch with left/right padding
            encoded = self.tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=1536,
            ).to(self.device)

            # Measure prompt length per example so we can slice out newly generated tokens
            input_lens = [
                len(self.tokenizer.encode(p, truncation=True, max_length=1536))
                for p in batch_prompts
            ]

            # Autoregressive generation without gradient tracking
            with torch.no_grad():
                gen_tokens = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_tokens,
                    do_sample=(self.temperature > 0.0),
                    temperature=self.temperature if self.temperature > 0.0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the newly generated continuation tokens (excluding prompt tokens)
            for b_idx, full_gen in enumerate(gen_tokens):
                prompt_len = input_lens[b_idx]
                generated_ids = full_gen[prompt_len:]
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                completions.append(text)
                token_counts.append(len(generated_ids))

        return completions, token_counts

    def evaluate(
        self,
        dataset: List[Dict[str, str]],
        split_name: str = "test",
        num_shots: int = 0,
        record_details: bool = True,
    ) -> BaselineEvalResult:
        """Run complete evaluation over dataset problems and compute verification metrics.

        Args:
            dataset: List of problem dicts with 'question' and 'gold_answer'.
            split_name: Name of evaluated split (e.g. 'dev' or 'test').
            num_shots: Few-shot count (0 or 4).
            record_details: If True, records per-problem inputs, outputs, and scores.

        Returns:
            Populated BaselineEvalResult dataclass containing aggregate scores and diagnostics.
        """
        # Format prompts and collect gold answers
        prompts = [format_gsm8k_prompt(item["question"], num_shots=num_shots) for item in dataset]
        gold_answers = [item["gold_answer"] for item in dataset]

        # Time the generation phase
        start_time = time.time()
        completions, token_counts = self.generate_batch(prompts)
        wall_time = time.time() - start_time

        # Metric accumulators
        strict_correct = 0
        permissive_correct = 0
        format_valid_count = 0
        parser_exceptions = 0
        error_indices: List[int] = []
        details: List[Dict[str, Any]] = []

        # Batch-evaluate format compliance
        format_rewards = format_reward_fn(prompts, completions)

        # Evaluate each problem outcome
        for idx, (comp, gold, n_tok, fmt_rew) in enumerate(
            zip(completions, gold_answers, token_counts, format_rewards)
        ):
            try:
                # 1. Strict extraction: requires <answer>, \boxed{}, or #### delimiters (RL-safe)
                strict_pred = extract_answer(comp, allow_last_number_fallback=False)
                strict_ok = verify_math_answer(strict_pred, gold) == 1.0

                # 2. Permissive extraction: falls back to final numeric token (diagnostic)
                permissive_pred = extract_answer(comp, allow_last_number_fallback=True)
                permissive_ok = verify_math_answer(permissive_pred, gold) == 1.0

                if strict_ok:
                    strict_correct += 1
                else:
                    # Record failed index for Phase 2 error orthogonality matrix
                    error_indices.append(idx)

                if permissive_ok:
                    permissive_correct += 1

                if fmt_rew == 1.0:
                    format_valid_count += 1

                # Record granular debug record if requested
                if record_details:
                    details.append({
                        "index": idx,
                        "question": dataset[idx]["question"],
                        "gold": gold,
                        "strict_pred": strict_pred,
                        "strict_correct": strict_ok,
                        "permissive_pred": permissive_pred,
                        "permissive_correct": permissive_ok,
                        "format_valid": (fmt_rew == 1.0),
                        "tokens": n_tok,
                        "completion": comp,
                    })

            except Exception as e:
                # Any parser exception is tracked to ensure verifier robustness
                parser_exceptions += 1
                error_indices.append(idx)
                logger.error("Unhandled parser exception on index %d: %s", idx, e)

        # Assemble summary report
        n_total = len(dataset)
        return BaselineEvalResult(
            model_name=self.model_name_or_path,
            split=split_name,
            num_shots=num_shots,
            total_examples=n_total,
            accuracy=(strict_correct / n_total) if n_total > 0 else 0.0,
            permissive_accuracy=(permissive_correct / n_total) if n_total > 0 else 0.0,
            format_compliance_rate=(format_valid_count / n_total) if n_total > 0 else 0.0,
            error_indices=error_indices,
            mean_completion_tokens=(sum(token_counts) / n_total) if n_total > 0 else 0.0,
            wall_time_seconds=wall_time,
            parser_exceptions=parser_exceptions,
            per_example_details=details if record_details else None,
        )
