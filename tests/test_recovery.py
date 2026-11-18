"""Unit tests for ExperimentStateJournal and SignalTrap (multigen/training/recovery.py)."""

from __future__ import annotations

import json
import os
import random
import signal
import tempfile
import pytest
import torch

from multigen.accounting.tracker import FLOPAccumulator
from multigen.training.recovery import ExperimentStateJournal, PhaseProgress, SignalTrap


def test_phase_progress_to_and_from_dict():
    p = PhaseProgress(
        status="completed",
        global_step=150,
        dataset_sample_offset=1200,
        flops_incurred=2.5e15,
        checkpoint_path="checkpoints/sft_final",
        extra_metadata={"val_loss": 0.45},
    )
    d = p.to_dict()
    assert d["status"] == "completed"
    assert d["global_step"] == 150
    assert d["flops_incurred"] == 2.5e15

    p_restored = PhaseProgress.from_dict(d)
    assert p_restored.status == "completed"
    assert p_restored.global_step == 150
    assert p_restored.checkpoint_path == "checkpoints/sft_final"
    assert p_restored.extra_metadata["val_loss"] == 0.45


def test_experiment_state_journal_atomic_save_and_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = os.path.join(tmp_dir, "recovery_state.json")
        journal = ExperimentStateJournal(journal_path)

        assert not journal.exists()
        assert journal.state["status"] == "pending"

        # Update Phase 1
        journal.update_phase(
            phase_name="sft",
            status="completed",
            step=320,
            sample_offset=2560,
            flops=2.5e15,
            checkpoint_path="/tmp/sft_ckpt",
            extra={"loss": 0.42},
        )

        assert journal.exists()
        assert journal.is_phase_completed("sft")
        assert not journal.is_phase_completed("grpo")

        # Reload from another instance
        reloaded = ExperimentStateJournal(journal_path)
        assert reloaded.is_phase_completed("sft")
        sft_data = reloaded.get_phase("sft")
        assert sft_data["global_step"] == 320
        assert sft_data["flops_incurred"] == 2.5e15
        assert sft_data["checkpoint_path"] == "/tmp/sft_ckpt"
        assert sft_data["extra_metadata"]["loss"] == 0.42


def test_experiment_state_journal_ledger_snapshot():
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = os.path.join(tmp_dir, "recovery_state.json")
        journal = ExperimentStateJournal(journal_path)

        acc = FLOPAccumulator(target_budget=1.0e16)
        acc.debit(1.0e15, category="train", tokens=200000, step=100)
        acc.debit(1.5e15, category="rollout", tokens=300000, step=101)

        journal.update_ledger(acc)

        # Verify saved file
        with open(journal_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "ledger" in data
        assert data["ledger"]["cumulative_flops"] == 2.5e15
        assert data["ledger"]["total_tokens_processed"] == 500000

        # Test accumulator deserialization from ledger
        rehydrated = FLOPAccumulator.from_dict(data["ledger"])
        assert rehydrated.cumulative_flops == 2.5e15
        assert rehydrated.train_flops == 1.0e15
        assert rehydrated.rollout_flops == 1.5e15
        assert rehydrated.target_budget == 1.0e16


def test_experiment_state_journal_rng_capture_and_restoration():
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = os.path.join(tmp_dir, "recovery_state.json")
        journal = ExperimentStateJournal(journal_path)

        # Set specific random seeds
        random.seed(12345)
        torch.manual_seed(12345)

        # Capture RNG
        journal.capture_rng_states()
        journal.save()

        # Advance RNG
        r1_orig = random.random()
        t1_orig = torch.rand(5)

        # Change RNG states arbitrarily
        random.seed(99999)
        torch.manual_seed(99999)
        assert random.random() != r1_orig

        # Reload journal and restore RNG
        reloaded = ExperimentStateJournal(journal_path)
        reloaded.restore_rng_states()

        # Verify restored sequence matches original
        r1_restored = random.random()
        t1_restored = torch.rand(5)

        assert r1_restored == r1_orig
        assert torch.allclose(t1_restored, t1_orig)


def test_signal_trap_registration():
    handled = []

    def mock_handler(signum: int) -> None:
        handled.append(signum)

    trap = SignalTrap(signals=["SIGINT"], on_signal_fn=mock_handler)
    assert not trap.is_registered

    trap.register()
    assert trap.is_registered
    assert signal.SIGINT in trap.original_handlers

    trap.unregister()
    assert not trap.is_registered
    assert len(trap.original_handlers) == 0
