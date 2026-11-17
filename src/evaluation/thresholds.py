"""Per-label decision thresholds selected on VALIDATION predictions only, then frozen.

``select_thresholds`` refuses ``split='test'``. The frozen threshold file records the manifest
hash, the model/experiment, the method and a hash of the validation predictions it came from.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_curve

from .. import LABELS
from ..data.csv_parser import binarize_targets
from ..utils.io import atomic_write_json, read_json


class ThresholdLeakageError(RuntimeError):
    pass


@dataclass
class ThresholdSet:
    thresholds: dict
    method: str
    selected_on: str
    manifest_hash: str
    experiment_id: str
    predictions_hash: str
    frozen: bool = True
    details: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        return atomic_write_json(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path, expected_manifest_hash: str | None = None) -> "ThresholdSet":
        d = read_json(path)
        ts = cls(**d)
        if expected_manifest_hash and ts.manifest_hash != expected_manifest_hash:
            raise ThresholdLeakageError("threshold file belongs to a different split manifest")
        return ts


def _youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    k = int(np.argmax(j))
    t = float(thr[k])
    return (min(t, 1.0) if np.isfinite(t) else 0.5), {"youden_j": float(j[k]), "tpr": float(tpr[k]), "fpr": float(fpr[k])}


def _best_f1(y, p):
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    k = int(np.argmax(f1))
    return float(ps[k]), {"f1": float(f1[k])}


def _target_sensitivity(y, p, target: float):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target)[0]
    k = int(ok[0]) if len(ok) else len(thr) - 1
    t = float(thr[k])
    return (min(t, 1.0) if np.isfinite(t) else 0.5), {"tpr": float(tpr[k]), "fpr": float(fpr[k]), "target": target}


def select_thresholds(y_true: np.ndarray, y_prob: np.ndarray, *, split: str, manifest_hash: str, experiment_id: str,
                      method: str = "youden", labels: Sequence[str] = LABELS, target_sensitivity: float = 0.9,
                      fallback: float = 0.5) -> ThresholdSet:
    if split != "val":
        raise ThresholdLeakageError(f"Thresholds must be selected on the validation split, not {split!r}.")
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    keep = np.isfinite(y_true).all(1)
    yb, _ = binarize_targets(y_true[keep])
    p = y_prob[keep]
    thr, details = {}, {}
    for j, l in enumerate(labels):
        if len(np.unique(yb[:, j])) < 2:
            thr[l], details[l] = fallback, {"note": f"single class in validation; fallback {fallback}"}
            continue
        if method == "youden":
            thr[l], details[l] = _youden(yb[:, j], p[:, j])
        elif method == "f1":
            thr[l], details[l] = _best_f1(yb[:, j], p[:, j])
        elif method == "sensitivity":
            thr[l], details[l] = _target_sensitivity(yb[:, j], p[:, j], target_sensitivity)
        elif method == "fixed":
            thr[l], details[l] = fallback, {"note": "fixed"}
        else:
            raise ValueError(method)
    ph = hashlib.sha256(np.ascontiguousarray(np.round(p, 6)).tobytes()).hexdigest()[:16]
    return ThresholdSet(thr, method, "val", manifest_hash, experiment_id, ph, True, details)
