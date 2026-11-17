"""Checkpoints that carry enough metadata to resume training and to refuse misuse.

A checkpoint contains model / optimizer / scheduler / scaler state, epoch, best metric,
RNG state, the split-manifest hash, the model + training config, its fingerprint and the
preprocessing fingerprint. Loading checks the manifest hash (and optionally the config
fingerprint), so a checkpoint cannot be silently evaluated against a different split.
Writing refuses to overwrite a checkpoint whose fingerprint differs unless explicitly allowed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

from ..utils.repro import capture_rng_state

CHECKPOINT_FORMAT = 1


class ManifestMismatchError(RuntimeError):
    pass


class CheckpointConfigMismatchError(RuntimeError):
    pass


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def read_checkpoint_meta(path: str | os.PathLike) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return ck.get("meta", {})


def save_checkpoint(path: str | os.PathLike, *, model: torch.nn.Module, meta: dict, optimizer=None, scheduler=None,
                    scaler=None, epoch: int = 0, best_metric: float | None = None, history: list | None = None,
                    allow_overwrite_different_config: bool = False, extra: dict | None = None) -> Path:
    """``meta`` must include ``manifest_hash`` and ``config_fingerprint``."""
    path = Path(path)
    for k in ("manifest_hash", "config_fingerprint"):
        if k not in meta:
            raise ValueError(f"checkpoint meta requires {k!r}")
    if path.exists() and not allow_overwrite_different_config:
        old = read_checkpoint_meta(path) or {}
        for k in ("manifest_hash", "config_fingerprint"):
            if old.get(k) and old.get(k) != meta[k]:
                raise CheckpointConfigMismatchError(
                    f"Refusing to overwrite {path.name}: existing {k}={str(old.get(k))[:12]} != new {str(meta[k])[:12]}. "
                    "Use a new experiment name/version or pass allow_overwrite_different_config=True.")
    state = {
        "format": CHECKPOINT_FORMAT,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "meta": meta,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_metric": best_metric,
        "history": history or [],
        "rng": capture_rng_state(),
        "extra": extra or {},
    }
    _atomic_torch_save(state, path)
    return path


def load_checkpoint(path: str | os.PathLike, model: torch.nn.Module | None = None, *,
                    expected_manifest_hash: str | None = None, expected_config_fingerprint: str | None = None,
                    optimizer=None, scheduler=None, scaler=None, map_location: str | torch.device = "cpu",
                    strict: bool = True) -> dict:
    ck = torch.load(Path(path), map_location=map_location, weights_only=False)
    meta = ck.get("meta", {})
    if expected_manifest_hash is not None and meta.get("manifest_hash") != expected_manifest_hash:
        raise ManifestMismatchError(
            f"Checkpoint {Path(path).name} was trained with manifest {str(meta.get('manifest_hash'))[:12]}, "
            f"but the current manifest is {expected_manifest_hash[:12]}. Refusing to evaluate on a different split.")
    if expected_config_fingerprint is not None and meta.get("config_fingerprint") != expected_config_fingerprint:
        raise CheckpointConfigMismatchError(
            f"Config fingerprint mismatch: checkpoint {meta.get('config_fingerprint')} vs expected {expected_config_fingerprint}")
    if model is not None:
        model.load_state_dict(ck["model"], strict=strict)
    if optimizer is not None and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    return ck
