"""Distillation pipeline runner orchestrating teacher generation, verification, and student SFT.

Implements Experiment 1.2 two-stage distillation architecture:
  - Stage 1: Batch sampling from teacher (Qwen2.5-Math-7B-Instruct) via vLLM (or HF fallback)
  - Filtering: Deterministic Best-of-N verifier filtering via math-verify
  - Stage 2: Supervised fine-tuning of student (Qwen3-0.6B) with LoRA and FLOP budgeting
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from multigen.accounting.flops import compute_teacher_flops
from multigen.accounting.tracker import FLOPAccumulator
from multigen.core.schema import TrainingRecipe
from multigen.eval.rewards import extract_answer, format_reward_fn, verify_math_answer
from multigen.training.sft_runner import SFTRunner

logger = logging.getLogger(__name__)

TEACHER_PROMPT_TEMPLATE = """Solve the following math word problem step by step. Put your chain-of-thought reasoning inside <think>...</think> tags and the final numerical answer inside <answer>...</answer> tags.

Question: {question}
"""


def normalize_teacher_trace(cand: str, gold: str) -> Optional[str]:
    """Normalize a verified teacher completion into standard XML format.

    Extracts step-by-step mathematical reasoning and ensures strict
    encapsulation inside <think>...</think> and <answer>...</answer> tags.
    """
    if not cand or not cand.strip():
        return None

    think_match = re.search(r"<think>(.*?)</think>", cand, re.DOTALL | re.IGNORECASE)
    if think_match and think_match.group(1).strip():
        think_text = think_match.group(1).strip()
    else:
        clean_text = cand
        # Remove trailing boxed answers, hash markers, and answer tags
        clean_text = re.sub(r"\\boxed\{[^}]*\}", "", clean_text)
        clean_text = re.sub(r"####.*$", "", clean_text)
        clean_text = re.sub(r"<answer>.*?</answer>", "", clean_text, flags=re.DOTALL | re.IGNORECASE)
        # Remove common trailing "Therefore/The answer is..." phrases
        clean_text = re.sub(
            r"(?:Therefore|Thus|Hence|So)?,?\s*(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*$",
            "",
            clean_text.strip(),
            flags=re.IGNORECASE,
        )
        think_text = clean_text.strip()

    if not think_text:
        return None

    return f"<think>\n{think_text}\n</think>\n<answer>{gold.strip()}</answer>"


class DistillRunner:
    """Orchestrates two-stage Knowledge Distillation baseline (EXP-1.2)."""

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

        self.teacher_params = 7_610_000_000  # Default ~7.61B for Qwen2.5-Math-7B

    def _determine_vllm_usage(self) -> bool:
        """Check if vLLM can be used for teacher inference."""
        if self.recipe.use_vllm is False:
            return False
        if not torch.cuda.is_available():
            return False
        try:
            import vllm  # noqa: F401
            return True
        except ImportError:
            return False

    def generate_teacher_candidates_hf(
        self,
        questions: List[str],
        teacher_model_name: str,
        k_samples: int = 4,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> List[List[str]]:
        """Generate candidate traces using Hugging Face transformers (fallback/CPU mode)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading teacher model %s via Hugging Face...", teacher_model_name)
        device_map = "auto" if torch.cuda.is_available() else None
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        all_candidates: List[List[str]] = []
        for q in questions:
            prompt = TEACHER_PROMPT_TEMPLATE.format(question=q)
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            q_candidates: List[str] = []
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=k_samples,
                    do_sample=temperature > 0.0,
                    temperature=temperature if temperature > 0.0 else 1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
                for out_seq in outputs:
                    gen_tokens = out_seq[inputs["input_ids"].shape[1] :]
                    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    q_candidates.append(text)
            all_candidates.append(q_candidates)

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return all_candidates

    def generate_teacher_candidates_vllm(
        self,
        questions: List[str],
        teacher_model_name: str,
        k_samples: int = 4,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> List[List[str]]:
        """Generate candidate traces using vLLM for high-throughput batched sampling."""
        import multiprocessing
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["VLLM_USE_V1"] = "0"
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        from vllm import LLM, SamplingParams

        logger.info("Initializing vLLM engine for teacher %s...", teacher_model_name)
        sampling_params = SamplingParams(
            n=k_samples,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        llm = LLM(
            model=teacher_model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
        )

        prompts = [TEACHER_PROMPT_TEMPLATE.format(question=q) for q in questions]
        outputs = llm.generate(prompts, sampling_params)

        all_candidates: List[List[str]] = []
        for out in outputs:
            q_candidates = [cand.text for cand in out.outputs]
            all_candidates.append(q_candidates)

        # Explicitly tear down vLLM to free all GPU memory before student training
        del llm
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return all_candidates

    def filter_and_format_traces(
        self,
        questions: List[str],
        gold_answers: List[str],
        candidate_lists: List[List[str]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Filter candidate completions with math-verify and select shortest correct path."""
        verified_traces: List[Dict[str, Any]] = []
        total_candidates = 0
        correct_candidates = 0
        problems_with_verified = 0

        total_teacher_tokens = 0

        for q, gold, candidates in zip(questions, gold_answers, candidate_lists):
            total_candidates += len(candidates)
            verified_for_q: List[Dict[str, Any]] = []

            for cand in candidates:
                # Approximate token length (~4 chars/token heuristic or split)
                cand_tokens = len(cand.split()) * 4 // 3 + 1
                total_teacher_tokens += cand_tokens

                # 1. Answer correctness verification
                pred_answer = extract_answer(cand, allow_last_number_fallback=False)
                is_correct = verify_math_answer(pred_answer, str(gold))

                if is_correct:
                    # 2. Normalize into strict XML tags (<think>...</think><answer>...</answer>)
                    normalized_cand = normalize_teacher_trace(cand, str(gold))
                    if normalized_cand is not None:
                        fmt_scores = format_reward_fn(None, [normalized_cand])
                        if fmt_scores and fmt_scores[0] > 0.5:
                            correct_candidates += 1
                            verified_for_q.append({
                                "question": q,
                                "response": normalized_cand.strip(),
                                "answer": str(gold),
                                "token_len": cand_tokens,
                            })

            if verified_for_q:
                problems_with_verified += 1
                # Best-of-N selection: shortest verified correct path to encourage conciseness
                best_trace = min(verified_for_q, key=lambda x: x["token_len"])
                verified_traces.append(best_trace)

        stats = {
            "total_problems": len(questions),
            "problems_with_verified": problems_with_verified,
            "verification_coverage": problems_with_verified / max(1, len(questions)),
            "total_candidates": total_candidates,
            "correct_candidates": correct_candidates,
            "total_teacher_tokens": total_teacher_tokens,
        }
        return verified_traces, stats

    def generate_and_filter_teacher_traces(
        self,
        questions: List[str],
        gold_answers: List[str],
        output_path: Optional[str] = None,
        force_regenerate: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Stage 1: Generate teacher rollouts, filter via verifier, and persist to JSONL."""
        save_path = output_path or self.recipe.teacher_trace_dataset_path or "data/teacher_traces_gsm8k.jsonl"
        path_obj = Path(save_path)

        # Check if verified traces already exist on disk
        if path_obj.exists() and not force_regenerate and path_obj.stat().st_size > 0:
            logger.info("Found cached teacher traces at %s. Loading from disk.", save_path)
            traces: List[Dict[str, Any]] = []
            with open(path_obj, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        traces.append(json.loads(line))
            logger.info("Loaded %d verified teacher traces from cache.", len(traces))
            return traces, {"cached": True, "count": len(traces)}

        teacher_model = self.recipe.teacher_model_name_or_path or "Qwen/Qwen2.5-Math-7B-Instruct"
        k_samples = self.recipe.teacher_k_samples
        temp = self.recipe.teacher_temperature
        max_tokens = self.recipe.teacher_max_tokens

        use_vllm = self._determine_vllm_usage()
        candidates = None
        if use_vllm:
            try:
                candidates = self.generate_teacher_candidates_vllm(
                    questions=questions,
                    teacher_model_name=teacher_model,
                    k_samples=k_samples,
                    temperature=temp,
                    max_new_tokens=max_tokens,
                )
            except Exception as e:
                logger.warning(
                    "vLLM candidate generation failed (%s). Falling back to Hugging Face transformers...",
                    e,
                )
                candidates = None

        if candidates is None:
            candidates = self.generate_teacher_candidates_hf(
                questions=questions,
                teacher_model_name=teacher_model,
                k_samples=k_samples,
                temperature=temp,
                max_new_tokens=max_tokens,
            )

        verified_traces, stats = self.filter_and_format_traces(
            questions=questions,
            gold_answers=gold_answers,
            candidate_lists=candidates,
        )

        # Compute teacher inference FLOPs: 2 * N_teacher * tokens
        total_tokens = stats["total_teacher_tokens"]
        teacher_flops = compute_teacher_flops(
            teacher_params=self.teacher_params,
            prompt_tokens=len(questions) * 60,
            gen_tokens=total_tokens,
        )
        stats["teacher_flops"] = teacher_flops

        # Debit compute based on accounting mode
        if self.recipe.accounting_mode == "strict_split":
            self.accumulator.debit(teacher_flops, category="teacher", tokens=total_tokens)
            logger.info(
                "[Strict Split Mode] Debited %.2e teacher FLOPs against total budget.",
                teacher_flops,
            )
        else:
            # Amortized mode: record in teacher ledger without deducting from student training
            self.accumulator.teacher_flops += teacher_flops
            self.accumulator.total_tokens_processed += total_tokens
            logger.info(
                "[Amortized Mode] Logged %.2e teacher FLOPs as offline data curation compute.",
                teacher_flops,
            )

        # Save to JSONL
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as f:
            for item in verified_traces:
                f.write(json.dumps(item) + "\n")

        logger.info(
            "Saved %d verified teacher traces to %s (Coverage: %.1f%%).",
            len(verified_traces),
            save_path,
            stats["verification_coverage"] * 100,
        )
        return verified_traces, stats

    def run_student_distillation(
        self,
        verified_traces: List[Dict[str, Any]],
        eval_dataset: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Stage 2: Train student model on verified traces via SFTRunner."""
        sft_runner = SFTRunner(
            recipe=self.recipe,
            output_dir=self.output_dir,
            accumulator=self.accumulator,
        )
        training_result = sft_runner.train(
            train_dataset=verified_traces,
            eval_dataset=eval_dataset,
        )
        return training_result

    def run(
        self,
        questions: List[str],
        gold_answers: List[str],
        eval_dataset: Optional[Any] = None,
        output_path: Optional[str] = None,
        force_regenerate: bool = False,
    ) -> Dict[str, Any]:
        """Execute full end-to-end two-stage distillation baseline."""
        logger.info("=== Phase 1: Generating & Verifying Teacher Traces ===")
        verified_traces, teacher_stats = self.generate_and_filter_teacher_traces(
            questions=questions,
            gold_answers=gold_answers,
            output_path=output_path,
            force_regenerate=force_regenerate,
        )

        # Force garbage collection and CUDA cache release between Phase 1 and Phase 2
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        logger.info("=== Phase 2: Student Supervised Fine-Tuning ===")
        sft_result = self.run_student_distillation(
            verified_traces=verified_traces,
            eval_dataset=eval_dataset,
        )

        return {
            "teacher_stats": teacher_stats,
            "sft_result": sft_result,
            "final_adapter_path": sft_result["final_adapter_path"],
            "intermediate_checkpoints": sft_result["intermediate_checkpoints"],
            "budget_summary": self.accumulator.get_summary(),
        }
