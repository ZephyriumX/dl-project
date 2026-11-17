"""Multi-label objectives with documented class-imbalance handling.

Default: ``BCEWithLogitsLoss`` with ``pos_weight = clip(sqrt(neg/pos), 1, 20)`` computed on the
*training split only*, and per-label weights ``[1,1,1,1,1,2]`` (the RSNA metric weights ``any``
double). The ``neg/pos`` weights used in the reference notebooks (epidural ~ 218-249) are
available as ``"inverse"`` but are not the default because they badly distort calibration.
Soft (fractional) targets are supported.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import LABELS

RSNA_LABEL_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 2.0)


def compute_pos_weight(train_targets: np.ndarray, mode: str = "sqrt_inverse", clip: tuple[float, float] = (1.0, 20.0)) -> np.ndarray:
    y = np.asarray(train_targets, dtype=np.float64)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    ratio = neg / np.maximum(pos, 1.0)
    if mode == "none":
        w = np.ones(y.shape[1])
    elif mode == "sqrt_inverse":
        w = np.sqrt(ratio)
    elif mode == "inverse":
        w = ratio
    else:
        raise ValueError(f"unknown pos_weight mode {mode!r}")
    return np.clip(w, *clip).astype(np.float32)


class WeightedBCE(nn.Module):
    def __init__(self, pos_weight: np.ndarray | torch.Tensor | None = None,
                 label_weights: tuple[float, ...] | None = RSNA_LABEL_WEIGHTS):
        super().__init__()
        pw = torch.as_tensor(pos_weight if pos_weight is not None else np.ones(len(LABELS)), dtype=torch.float32)
        lw = torch.as_tensor(label_weights if label_weights is not None else np.ones(len(LABELS)), dtype=torch.float32)
        self.register_buffer("pos_weight", pw)
        self.register_buffer("label_weights", lw / lw.mean())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), pos_weight=self.pos_weight,
                                                  reduction="none")
        return (loss * self.label_weights).mean()


class FocalBCE(nn.Module):
    """Binary focal loss per label (Lin et al.), supports soft targets."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None,
                 label_weights: tuple[float, ...] | None = RSNA_LABEL_WEIGHTS):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
        lw = torch.as_tensor(label_weights if label_weights is not None else np.ones(len(LABELS)), dtype=torch.float32)
        self.register_buffer("label_weights", lw / lw.mean())

    def forward(self, logits, target):
        logits, target = logits.float(), target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        loss = bce * (1 - p_t).pow(self.gamma)
        if self.alpha is not None:
            loss = loss * (self.alpha * target + (1 - self.alpha) * (1 - target))
        return (loss * self.label_weights).mean()


def build_loss(cfg: dict, train_targets: np.ndarray | None = None) -> tuple[nn.Module, dict]:
    """Build the loss from config; returns (loss_module, description dict for logging)."""
    kind = cfg.get("kind", "bce")
    lw = cfg.get("label_weights", RSNA_LABEL_WEIGHTS)
    mode = cfg.get("pos_weight", "sqrt_inverse")
    pw = None
    if kind == "bce":
        if mode != "none":
            if train_targets is None:
                raise ValueError("train_targets required to compute pos_weight")
            pw = compute_pos_weight(train_targets, mode, tuple(cfg.get("pos_weight_clip", (1.0, 20.0))))
        loss = WeightedBCE(pw, lw)
    elif kind == "focal":
        loss = FocalBCE(cfg.get("gamma", 2.0), cfg.get("alpha"), lw)
    else:
        raise ValueError(f"unknown loss kind {kind!r}")
    desc = {"kind": kind, "pos_weight_mode": mode, "label_weights": list(lw) if lw is not None else None,
            "pos_weight": dict(zip(LABELS, pw.tolist())) if pw is not None else None}
    return loss, desc
