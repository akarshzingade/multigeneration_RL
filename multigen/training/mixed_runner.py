"""Mixed Learning Runner combining Knowledge Distillation and RL under strict compute budgets.

Implements Experiment 1.3 specification (Hypothesis H1):
  - Mode A: Sequential Phased Interleaved Transfer (SFT Warmup -> GRPO Exploration) [DEFAULT]
  - Mode B: Dynamic Teacher-Regularized GRPO
  - Mode C: Multi-Cycle Micro-Interleaving
  - Robust crash recovery, atomic checkpointing, and independent cadence toggles
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch

from multigen.accounting.tracker import FLOPAccumulator
from multigen.core.population_dag import PopulationDAG
from multigen.core.schema import CapabilityProfile, ModelNode, TrainingRecipe
from multigen.training.distill_runner import DistillRunner
from multigen.training.grpo_runner import GRPORunner
from multigen.training.recovery import ExperimentStateJournal, SignalTrap
from multigen.training.sft_runner import SFTRunner

logger = logging.getLogger(__name__)


class MixedRunner:
    """Orchestrates hybrid distillation and RL with crash resilience and compute parity."""

    def __init__(
        self,
        recipe: TrainingRecipe,
        output_dir: Union[str, Path],
        resume: bool = True,
        accumulator: Optional[FLOPAccumulator] = None,
    ) -> None:
        self.recipe = recipe
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume

        # Journal filename from recipe
        journal_filename = getattr(recipe, "journal_filename", "recovery_state.json")
        self.journal_path = self.output_dir / journal_filename
        self.journal = ExperimentStateJournal(str(self.journal_path))
        if not self.resume:
            logger.info("Resumption disabled (--no-resume). Resetting state journal.")
            self.journal.reset()

        # Initialize or reuse FLOP Accumulator
        if accumulator is not None:
            self.accumulator = accumulator
        else:
            self.accumulator = FLOPAccumulator(target_budget=recipe.target_flop_budget)

        # Register signal handlers for emergency exit saving if enabled
        self.signal_trap: Optional[SignalTrap] = None
        if getattr(recipe, "enable_signal_trapping", True):
            signals = getattr(recipe, "save_on_signals", ["SIGINT", "SIGTERM"])
            self.signal_trap = SignalTrap(signals=signals, on_signal_fn=self.save_emergency_checkpoint)
            self.signal_trap.register()

        # Rehydrate from journal if resuming
        if getattr(recipe, "resumption_enabled", True) and self.resume and self.journal.exists():
            self._rehydrate_from_journal()

    def _rehydrate_from_journal(self) -> None:
        """Restore FLOP ledger and random states from journal."""
        ledger_state = self.journal.state.get("ledger", {})
        if ledger_state:
            self.accumulator.cumulative_flops = float(ledger_state.get("cumulative_flops", 0.0))
            self.accumulator.train_flops = float(ledger_state.get("train_flops", 0.0))
            self.accumulator.rollout_flops = float(ledger_state.get("rollout_flops", 0.0))
            self.accumulator.teacher_flops = float(ledger_state.get("teacher_flops", 0.0))
            self.accumulator.ref_flops = float(ledger_state.get("ref_flops", 0.0))
            self.accumulator.eval_flops = float(ledger_state.get("eval_flops", 0.0))
            self.accumulator.total_tokens_processed = int(ledger_state.get("total_tokens_processed", 0))
            self.accumulator.step_history = list(ledger_state.get("step_history", []))
            logger.info(
                "Rehydrated FLOP Accumulator: %.2e / %.2e FLOPs (%.1f%% of budget)",
                self.accumulator.cumulative_flops,
                self.accumulator.target_budget or 0.0,
                self.accumulator.budget_utilization * 100,
            )
        self.journal.restore_rng_states()

    def save_checkpoint(self, tag: str = "milestone", step: Optional[int] = None) -> str:
        """Persist current execution progress, random states, and FLOP ledger."""
        current_flops = int(self.accumulator.cumulative_flops)
        ckpt_name = f"checkpoint_{tag}_flops_{current_flops}"
        if step is not None:
            ckpt_name += f"_step_{step}"
        ckpt_dir = self.output_dir / ckpt_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Update ledger in journal
        self.journal.update_ledger(self.accumulator)
        self.journal.capture_rng_states()
        self.journal.save()

        # Save standalone FLOP ledger copy inside checkpoint directory
        self.accumulator.save(ckpt_dir / "flops_ledger.json")

        logger.info("Saved state journal and checkpoint to %s", ckpt_dir)

        # Commit cloud storage volume if available
        self._commit_cloud_volume()
        return str(ckpt_dir)

    def save_emergency_checkpoint(self, signum: int = 0) -> str:
        """Triggered upon catching SIGINT or SIGTERM."""
        logger.warning("Saving emergency checkpoint on signal %d...", signum)
        active_phase = self.journal.state.get("active_phase", "unknown")
        self.journal.state["status"] = "interrupted"
        self.journal.state["interrupted_signal"] = signum
        return self.save_checkpoint(tag="emergency")

    def _commit_cloud_volume(self) -> None:
        """Attempt to commit Modal volume if running in Modal container."""
        try:
            import modal  # noqa: F401
            vol = modal.Volume.from_name("multigen-storage", create_if_missing=False)
            vol.commit()
            logger.debug("Committed checkpoint to Modal storage volume.")
        except Exception:
            pass

    def run(
        self,
        questions: Optional[List[str]] = None,
        gold_answers: Optional[List[str]] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute mixed learning according to configured mode."""
        mode = getattr(self.recipe, "mode", "interleaved_sequential")
        budget_disp = (
            f"{self.recipe.target_flop_budget:.2e}"
            if self.recipe.target_flop_budget
            else "None (unconstrained)"
        )
        logger.info(
            "Starting MixedRunner execution (mode=%s, target_flops=%s)",
            mode,
            budget_disp,
        )

        if mode == "interleaved_sequential":
            return self._run_interleaved_sequential(
                questions=questions,
                gold_answers=gold_answers,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
        elif mode == "teacher_regularized":
            return self._run_teacher_regularized(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
        elif mode == "micro_interleaved":
            return self._run_micro_interleaved(
                questions=questions,
                gold_answers=gold_answers,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
        else:
            raise ValueError(f"Unknown mixed learning mode: {mode}")

    def _run_interleaved_sequential(
        self,
        questions: Optional[List[str]] = None,
        gold_answers: Optional[List[str]] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Mode A: Phased Interleaved SFT Warmup -> GRPO Exploration."""
        target_budget = self.recipe.target_flop_budget
        split_ratio = getattr(self.recipe, "distill_budget_ratio", 0.25)
        sft_budget = (target_budget * split_ratio) if target_budget else None
        stage_criterion = getattr(
            self.recipe,
            "stage_transition_criterion",
            "plateau" if target_budget is None else "flops",
        )

        sft_state = self.journal.get_phase("sft")
        grpo_state = self.journal.get_phase("grpo")

        sft_adapter_path: Optional[str] = sft_state.get("checkpoint_path")
        sft_result: Dict[str, Any] = {}

        # =========================================================================
        # Stage 1: Supervised Knowledge Distillation (Warmup)
        # =========================================================================
        if target_budget and stage_criterion == "flops":
            sft_done = self.journal.is_phase_completed("sft") and (self.accumulator.cumulative_flops >= sft_budget * 0.95)
        else:
            sft_done = self.journal.is_phase_completed("sft")

        if not sft_done:
            budget_msg = (
                f"{sft_budget:.2e} FLOPs, {split_ratio * 100:.1f}%"
                if sft_budget
                else f"Unconstrained ({stage_criterion})"
            )
            logger.info(
                "=== [Mixed EXP-1.3 Stage 1] SFT Distillation Warmup (Budget: %s) ===",
                budget_msg,
            )
            self.journal.update_phase("sft", status="in_progress")

            sft_recipe = self.recipe.model_copy(deep=True)
            sft_recipe.algorithm = "distill"
            if target_budget and stage_criterion == "flops":
                sft_recipe.target_flop_budget = sft_budget
                self.accumulator.target_budget = sft_budget
            else:
                sft_recipe.target_flop_budget = None
                self.accumulator.target_budget = None
                sft_recipe.termination_criterion = stage_criterion
                if self.recipe.sft_epochs is not None:
                    sft_recipe.num_train_epochs = self.recipe.sft_epochs
                if self.recipe.sft_max_steps is not None:
                    sft_recipe.max_steps = self.recipe.sft_max_steps
                sft_recipe.early_stopping_patience = getattr(self.recipe, "sft_patience", 3)

            sft_out = self.output_dir / "sft_stage"
            sft_out.mkdir(parents=True, exist_ok=True)

            # Check if verified teacher trace dataset exists
            traces_path = getattr(self.recipe, "teacher_trace_dataset_path", "data/teacher_traces_gsm8k.jsonl")
            if traces_path and os.path.exists(traces_path):
                logger.info("Found pre-existing teacher traces at %s. Running SFTRunner directly.", traces_path)
                from datasets import load_dataset
                dataset = load_dataset("json", data_files=traces_path, split="train")

                sft_runner = SFTRunner(
                    recipe=sft_recipe,
                    output_dir=str(sft_out),
                    accumulator=self.accumulator,
                )
                sft_result = sft_runner.train(train_dataset=dataset, eval_dataset=eval_dataset)
                sft_adapter_path = sft_result["final_adapter_path"]
            else:
                logger.info("Teacher traces not pre-cached. Running full DistillRunner pipeline.")
                distill_runner = DistillRunner(
                    recipe=sft_recipe,
                    output_dir=str(sft_out),
                    accumulator=self.accumulator,
                )
                # If questions/gold_answers not provided, load from GSM8K train
                if not questions or not gold_answers:
                    from datasets import load_dataset
                    ds = load_dataset(self.recipe.dataset_name, self.recipe.dataset_subset, split="train")
                    questions = [ex["question"] for ex in ds]
                    gold_answers = [ex["answer"] for ex in ds]

                distill_result = distill_runner.run(
                    questions=questions,
                    gold_answers=gold_answers,
                    eval_dataset=eval_dataset,
                    output_path=traces_path,
                )
                sft_result = distill_result["sft_result"]
                sft_adapter_path = distill_result["final_adapter_path"]

            self.journal.update_phase(
                "sft",
                status="completed",
                flops=self.accumulator.cumulative_flops,
                checkpoint_path=sft_adapter_path,
            )
            self.save_checkpoint(tag="sft_complete")
            logger.info("Stage 1 (SFT) Complete! Warmup adapter persisted to: %s", sft_adapter_path)

            # Memory cleanup between phases
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        else:
            logger.info("Stage 1 (SFT) already completed in prior run. Loaded adapter: %s", sft_adapter_path)

        # =========================================================================
        # Stage 2: Online GRPO Exploration
        # =========================================================================
        grpo_adapter_path: Optional[str] = grpo_state.get("checkpoint_path")
        grpo_result: Dict[str, Any] = {}

        termination_criterion = getattr(
            self.recipe,
            "termination_criterion",
            "plateau" if target_budget is None else "flops",
        )
        if target_budget and termination_criterion == "flops":
            grpo_done = self.journal.is_phase_completed("grpo") and (self.accumulator.cumulative_flops >= target_budget * 0.95)
        else:
            grpo_done = self.journal.is_phase_completed("grpo")

        if not grpo_done:
            budget_msg = (
                f"Ceiling: {target_budget:.2e} FLOPs"
                if target_budget
                else f"Unconstrained ({termination_criterion})"
            )
            logger.info(
                "=== [Mixed EXP-1.3 Stage 2] Online GRPO Exploration (%s) ===",
                budget_msg,
            )
            self.journal.update_phase("grpo", status="in_progress")

            grpo_recipe = self.recipe.model_copy(deep=True)
            grpo_recipe.algorithm = "grpo"
            if target_budget and termination_criterion == "flops":
                grpo_recipe.target_flop_budget = target_budget  # Continues until total ceiling
                self.accumulator.target_budget = target_budget
            else:
                grpo_recipe.target_flop_budget = None
                self.accumulator.target_budget = None
                grpo_recipe.termination_criterion = termination_criterion
                if self.recipe.grpo_epochs is not None:
                    grpo_recipe.num_train_epochs = self.recipe.grpo_epochs
                if self.recipe.grpo_max_steps is not None:
                    grpo_recipe.max_steps = self.recipe.grpo_max_steps
                grpo_recipe.early_stopping_patience = getattr(self.recipe, "grpo_patience", 3)

            # Reference policy strategy (Decision 4.3)
            ref_policy_source = getattr(self.recipe, "ref_policy_source", "sft_warmup")
            if ref_policy_source == "sft_warmup":
                grpo_recipe.parent_adapter_path = sft_adapter_path
            elif ref_policy_source == "base_model":
                # Start policy from sft adapter but reference from base model
                grpo_recipe.parent_adapter_path = sft_adapter_path
            elif ref_policy_source == "ema":
                grpo_recipe.parent_adapter_path = sft_adapter_path

            grpo_out = self.output_dir / "grpo_stage"
            grpo_out.mkdir(parents=True, exist_ok=True)

            grpo_runner = GRPORunner(
                recipe=grpo_recipe,
                output_dir=str(grpo_out),
                accumulator=self.accumulator,
            )

            grpo_result = grpo_runner.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            grpo_adapter_path = grpo_result["adapter_path"]

            self.journal.update_phase(
                "grpo",
                status="completed",
                flops=self.accumulator.cumulative_flops,
                checkpoint_path=grpo_adapter_path,
            )
            self.journal.state["status"] = "completed"
            self.journal.save()
            self.save_checkpoint(tag="grpo_complete")
            logger.info("Stage 2 (GRPO) Complete! Final hybrid adapter saved to: %s", grpo_adapter_path)
        else:
            logger.info("Stage 2 (GRPO) already marked completed. Loaded adapter: %s", grpo_adapter_path)

        # Cleanup signal trap on successful finish
        if self.signal_trap is not None:
            self.signal_trap.unregister()

        return {
            "mode": "interleaved_sequential",
            "sft_adapter_path": sft_adapter_path,
            "final_adapter_path": grpo_adapter_path,
            "sft_result": sft_result,
            "grpo_result": grpo_result,
            "budget_summary": self.accumulator.get_summary(),
        }

    def _run_teacher_regularized(
        self,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Mode B: Teacher-regularized online GRPO."""
        phase_name = "teacher_regularized_grpo"
        phase_state = self.journal.get_phase(phase_name)
        final_adapter_path: Optional[str] = phase_state.get("checkpoint_path")

        if not self.journal.is_phase_completed(phase_name):
            self.journal.update_phase(phase_name, status="in_progress")

            grpo_recipe = self.recipe.model_copy(deep=True)
            grpo_recipe.algorithm = "grpo"
            grpo_out = self.output_dir / "regularized_grpo_stage"

            runner = GRPORunner(
                recipe=grpo_recipe,
                output_dir=str(grpo_out),
                accumulator=self.accumulator,
            )
            res = runner.train(train_dataset=train_dataset, eval_dataset=eval_dataset)
            final_adapter_path = res["adapter_path"]

            self.journal.update_phase(
                phase_name,
                status="completed",
                flops=self.accumulator.cumulative_flops,
                checkpoint_path=final_adapter_path,
            )
            self.journal.state["status"] = "completed"
            self.journal.save()
        else:
            logger.info("Phase %s already completed: %s", phase_name, final_adapter_path)

        return {
            "mode": "teacher_regularized",
            "final_adapter_path": final_adapter_path,
            "budget_summary": self.accumulator.get_summary(),
        }

    def _run_micro_interleaved(
        self,
        questions: Optional[List[str]] = None,
        gold_answers: Optional[List[str]] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Mode C: Multi-cycle micro-interleaving."""
        num_cycles = getattr(self.recipe, "num_cycles", 4)
        total_budget = self.recipe.target_flop_budget
        cycle_budget = (total_budget / num_cycles) if total_budget else None
        sft_per_cycle = (cycle_budget * getattr(self.recipe, "distill_budget_ratio", 0.25)) if cycle_budget else None

        current_adapter = self.recipe.parent_adapter_path

        for c in range(1, num_cycles + 1):
            sft_phase_name = f"cycle_{c}_sft"
            grpo_phase_name = f"cycle_{c}_grpo"

            target_cycle_sft_cap = (c - 1) * cycle_budget + sft_per_cycle
            target_cycle_grpo_cap = c * cycle_budget

            # 1. Micro SFT
            if not self.journal.is_phase_completed(sft_phase_name):
                self.journal.update_phase(sft_phase_name, status="in_progress")
                sft_recipe = self.recipe.model_copy(deep=True)
                sft_recipe.algorithm = "distill"
                sft_recipe.parent_adapter_path = current_adapter
                sft_recipe.target_flop_budget = target_cycle_sft_cap

                sft_out = self.output_dir / f"cycle_{c}_sft"
                sft_runner = SFTRunner(
                    recipe=sft_recipe,
                    output_dir=str(sft_out),
                    accumulator=self.accumulator,
                )
                traces_path = getattr(self.recipe, "teacher_trace_dataset_path", "data/teacher_traces_gsm8k.jsonl")
                from datasets import load_dataset
                ds = load_dataset("json", data_files=traces_path, split="train") if os.path.exists(traces_path) else None
                res = sft_runner.train(train_dataset=ds, eval_dataset=eval_dataset)
                current_adapter = res["final_adapter_path"]

                self.journal.update_phase(
                    sft_phase_name,
                    status="completed",
                    flops=self.accumulator.cumulative_flops,
                    checkpoint_path=current_adapter,
                )

            # 2. Micro GRPO
            if not self.journal.is_phase_completed(grpo_phase_name):
                self.journal.update_phase(grpo_phase_name, status="in_progress")
                grpo_recipe = self.recipe.model_copy(deep=True)
                grpo_recipe.algorithm = "grpo"
                grpo_recipe.parent_adapter_path = current_adapter
                grpo_recipe.target_flop_budget = target_cycle_grpo_cap

                grpo_out = self.output_dir / f"cycle_{c}_grpo"
                grpo_runner = GRPORunner(
                    recipe=grpo_recipe,
                    output_dir=str(grpo_out),
                    accumulator=self.accumulator,
                )
                res = grpo_runner.train(train_dataset=train_dataset, eval_dataset=eval_dataset)
                current_adapter = res["adapter_path"]

                self.journal.update_phase(
                    grpo_phase_name,
                    status="completed",
                    flops=self.accumulator.cumulative_flops,
                    checkpoint_path=current_adapter,
                )

        self.journal.state["status"] = "completed"
        self.journal.save()
        return {
            "mode": "micro_interleaved",
            "num_cycles": num_cycles,
            "final_adapter_path": current_adapter,
            "budget_summary": self.accumulator.get_summary(),
        }

    def register_in_population_dag(
        self,
        dag_path: str,
        adapter_path: str,
        capability_profile: Optional[CapabilityProfile] = None,
        node_id: Optional[str] = None,
    ) -> ModelNode:
        """Register the mixed learning result in the Population Lineage DAG with dual parent provenance."""
        dag = PopulationDAG.load_or_create(dag_path)

        # Ensure teacher node is registered
        teacher_node_id = "teacher_qwen25_math_7b"
        if not dag.has_node(teacher_node_id):
            teacher_node = ModelNode(
                node_id=teacher_node_id,
                parent_ids=[],
                depth=0,
                recipe=TrainingRecipe(
                    algorithm="distill",
                    base_model_name_or_path=self.recipe.teacher_model_name_or_path or "Qwen/Qwen2.5-Math-7B-Instruct",
                ),
                cumulative_flops=0.0,
                metadata={
                    "is_external_teacher": True,
                    "teacher_model": self.recipe.teacher_model_name_or_path,
                },
            )
            dag.add_node(teacher_node)

        # Create hybrid child node with dual ancestry (Decision 4.7)
        if node_id is None:
            node_id = f"exp1_3_mixed_seed{self.recipe.seed}"

        child_node = ModelNode(
            node_id=node_id,
            parent_ids=["node_0", teacher_node_id],
            depth=1,
            recipe=self.recipe,
            capability_profile=capability_profile,
            adapter_path=adapter_path,
            cumulative_flops=self.accumulator.cumulative_flops,
            metadata={
                "hybrid_mode": getattr(self.recipe, "mode", "interleaved_sequential"),
                "distill_budget_ratio": getattr(self.recipe, "distill_budget_ratio", 0.25),
                "breakdown": self.accumulator.get_summary().get("breakdown", {}),
            },
        )
        dag.add_node(child_node)
        dag.save(dag_path)
        logger.info("Registered hybrid node %s in Population DAG at %s", node_id, dag_path)
        return child_node
