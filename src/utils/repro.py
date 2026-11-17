"""Seeding and RNG-state capture for reproducible, resumable runs."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU + CUDA).

    ``deterministic=True`` enables cuDNN determinism (slower; some ops may raise).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> bool:
    """Restore RNG state; returns False (without raising) if it cannot be restored exactly."""
    if not state:
        return False
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
        if "cuda" in state and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
        return True
    except Exception:  # pragma: no cover - best effort (e.g. different device count)
        return False


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` giving each worker a distinct, reproducible seed."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
