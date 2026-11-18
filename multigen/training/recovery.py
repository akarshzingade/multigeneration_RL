"""Fault-tolerant state journaling, atomic checkpointing, and signal trapping for Multigen-RL.

Ensures zero compute leakage, atomic filesystem updates, and deterministic resumption
across user interruptions (SIGINT), cloud preemption (SIGTERM), and system crashes.
"""

from __future__ import annotations

import json
import logging
import os
import random
import signal
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import torch

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from multigen.accounting.tracker import FLOPAccumulator

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PhaseProgress:
    """Tracks state and progress of an individual training phase."""

    status: str = "pending"  # "pending", "in_progress", "completed", "failed"
    global_step: int = 0
    dataset_sample_offset: int = 0
    flops_incurred: float = 0.0
    checkpoint_path: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseProgress":
        return cls(
            status=data.get("status", "pending"),
            global_step=data.get("global_step", 0),
            dataset_sample_offset=data.get("dataset_sample_offset", 0),
            flops_incurred=float(data.get("flops_incurred", 0.0)),
            checkpoint_path=data.get("checkpoint_path"),
            extra_metadata=data.get("extra_metadata", {}),
        )


class ExperimentStateJournal:
    """Atomic state journal ensuring complete recovery across process interruptions."""

    def __init__(self, journal_path: Union[str, Path]) -> None:
        self.journal_path = Path(journal_path).resolve()
        self.state: Dict[str, Any] = {
            "version": 1,
            "status": "pending",  # "pending", "in_progress", "completed", "interrupted"
            "active_phase": None,
            "phases": {},
            "ledger": {},
            "random_states": {},
            "last_updated_utc": None,
        }
        if self.journal_path.exists():
            self.load()

    def exists(self) -> bool:
        """Check if journal file exists on disk."""
        return self.journal_path.exists()

    def reset(self) -> None:
        """Reset journal to a fresh initial state."""
        self.state = {
            "version": 1,
            "status": "pending",
            "active_phase": None,
            "phases": {},
            "ledger": {},
            "random_states": {},
            "last_updated_utc": None,
        }
        if self.journal_path.exists():
            try:
                self.journal_path.unlink()
            except OSError:
                pass

    def load(self) -> None:
        """Load state from existing journal file."""
        try:
            with open(self.journal_path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
            logger.info("Loaded existing recovery state from %s", self.journal_path)
        except Exception as e:
            logger.error("Failed to load journal %s: %s", self.journal_path, e)
            raise

    def save(self) -> None:
        """Atomically persist state using tempfile and atomic rename."""
        self.state["last_updated_utc"] = _utc_now_iso()
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

        temp_file = self.journal_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, self.journal_path)

    def update_phase(
        self,
        phase_name: str,
        status: str,
        step: int = 0,
        sample_offset: int = 0,
        flops: float = 0.0,
        checkpoint_path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update individual phase progress and save atomically."""
        if "phases" not in self.state:
            self.state["phases"] = {}

        if phase_name not in self.state["phases"]:
            self.state["phases"][phase_name] = {}

        p = self.state["phases"][phase_name]
        p["status"] = status
        p["global_step"] = step
        p["dataset_sample_offset"] = sample_offset
        p["flops_incurred"] = flops
        if checkpoint_path is not None:
            p["checkpoint_path"] = str(checkpoint_path)
        if extra:
            p.setdefault("extra_metadata", {}).update(extra)

        self.state["active_phase"] = phase_name
        if self.state["status"] != "completed":
            self.state["status"] = "in_progress"
        self.save()

    def get_phase(self, phase_name: str) -> Dict[str, Any]:
        """Get state dictionary for a specific phase."""
        return self.state.get("phases", {}).get(phase_name, {})

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a specific phase has completed."""
        return self.get_phase(phase_name).get("status") == "completed"

    def update_ledger(self, accumulator: FLOPAccumulator) -> None:
        """Snapshot the FLOPAccumulator ledger into state journal."""
        self.state["ledger"] = accumulator.to_dict()
        self.save()

    def capture_rng_states(self) -> None:
        """Capture Python, NumPy, and PyTorch RNG states."""
        rng: Dict[str, Any] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state().tolist(),
        }
        if torch.cuda.is_available():
            rng["cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
        if HAS_NUMPY:
            try:
                np_state = np.random.get_state()
                # np_state is tuple of (str, ndarray of uint32, int, int, float)
                state_list = [int(x) for x in np_state[1]]
                rng["numpy"] = [
                    str(np_state[0]),
                    state_list,
                    int(np_state[2]),
                    int(np_state[3]),
                    float(np_state[4]),
                ]
            except Exception as e:
                logger.debug("Could not serialize numpy RNG: %s", e)

        self.state["random_states"] = rng

    def restore_rng_states(self) -> None:
        """Restore Python, NumPy, and PyTorch RNG states."""
        rng = self.state.get("random_states", {})
        if not rng:
            return

        try:
            if "python" in rng and rng["python"]:
                py_state = rng["python"]
                if isinstance(py_state, (list, tuple)) and len(py_state) == 3:
                    version, state_list, gauss = py_state
                    if isinstance(state_list, (list, tuple)):
                        state_tuple = tuple(state_list)
                        random.setstate((version, state_tuple, gauss))
                    else:
                        random.setstate(tuple(py_state))
                elif isinstance(py_state, (list, tuple)):
                    random.setstate(tuple(py_state))
        except Exception as e:
            logger.warning("Failed to restore Python RNG: %s", e)

        try:
            if "torch" in rng and rng["torch"]:
                torch.set_rng_state(torch.ByteTensor(rng["torch"]))
        except Exception as e:
            logger.warning("Failed to restore PyTorch RNG: %s", e)

        try:
            if "cuda" in rng and rng["cuda"] and torch.cuda.is_available():
                cuda_tensors = [torch.ByteTensor(s) for s in rng["cuda"]]
                torch.cuda.set_rng_state_all(cuda_tensors)
        except Exception as e:
            logger.warning("Failed to restore CUDA RNG: %s", e)

        if HAS_NUMPY and "numpy" in rng and rng["numpy"]:
            try:
                np_state = rng["numpy"]
                # numpy state is tuple of (str, ndarray, int, int, float)
                if isinstance(np_state, list) and len(np_state) == 5:
                    np_tuple = (
                        np_state[0],
                        np.array(np_state[1], dtype=np.uint32),
                        np_state[2],
                        np_state[3],
                        np_state[4],
                    )
                    np.random.set_state(np_tuple)
            except Exception as e:
                logger.warning("Failed to restore NumPy RNG: %s", e)


class SignalTrap:
    """Registers POSIX signal traps to cleanly save emergency state upon cancellation."""

    def __init__(
        self,
        signals: Optional[List[str]] = None,
        on_signal_fn: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.signal_names = signals if signals is not None else ["SIGINT", "SIGTERM"]
        self.on_signal_fn = on_signal_fn
        self.original_handlers: Dict[int, Any] = {}
        self.is_registered = False

    def register(self) -> None:
        """Register signal handlers for configured signals."""
        if self.is_registered:
            return

        for name in self.signal_names:
            sig_num = getattr(signal, name, None)
            if sig_num is not None:
                try:
                    self.original_handlers[sig_num] = signal.getsignal(sig_num)
                    signal.signal(sig_num, self._handle_signal)
                    logger.debug("Registered trap for %s (%d)", name, sig_num)
                except (ValueError, OSError) as e:
                    logger.debug("Could not register trap for %s: %s", name, e)

        self.is_registered = True

    def unregister(self) -> None:
        """Restore original signal handlers."""
        if not self.is_registered:
            return

        for sig_num, handler in self.original_handlers.items():
            try:
                signal.signal(sig_num, handler)
            except (ValueError, OSError):
                pass

        self.original_handlers.clear()
        self.is_registered = False

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Intercept termination signal, invoke callback, and terminate cleanly."""
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.warning("Caught signal %s (%d). Triggering emergency save callback...", sig_name, signum)
        if self.on_signal_fn is not None:
            try:
                self.on_signal_fn(signum)
            except Exception as e:
                logger.error("Error during signal trap callback: %s", e)

        # Restore original handler and exit
        self.unregister()
        sys.exit(128 + signum)
