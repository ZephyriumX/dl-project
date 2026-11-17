"""Six-label evaluation with explicit handling of undefined metrics.

Undefined values are returned as ``NaN`` together with a reason (never silently as 0):

* ROC-AUC / AP undefined when a label has only one class in the evaluated set (``single_class``);
* precision undefined when nothing is predicted positive (``no_predicted_positive``);
* recall/sensitivity undefined without positives, specificity without negatives;
* F1 undefined when precision or recall is undefined *and* there are no TP (reported as NaN).

Macro averages are ``nanmean`` over defined labels and always report ``n_defined``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import LABELS
from ..data.csv_parser import binarize_targets

RSNA_WEIGHTS = np.array([1, 1, 1, 1, 1, 2], dtype=np.float64)


def _valid_rows(y: np.ndarray) -> np.ndarray:
    return np.isfinite(y).all(axis=1) if y.ndim == 2 else np.isfinite(y)


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def auc_or_nan(y: np.ndarray, p: np.ndarray) -> tuple[float, str]:
    if len(y) == 0:
        return float("nan"), "empty"
    if len(np.unique(y)) < 2:
        return float("nan"), "single_class"
    return float(roc_auc_score(y, p)), ""


def ap_or_nan(y: np.ndarray, p: np.ndarray) -> tuple[float, str]:
    if len(y) == 0:
        return float("nan"), "empty"
    if y.sum() == 0:
        return float("nan"), "no_positives"
    if y.sum() == len(y):
        return float("nan"), "single_class"
    return float(average_precision_score(y, p)), ""


def per_label_auc(y_true: np.ndarray, y_prob: np.ndarray, labels: Sequence[str] = LABELS) -> dict[str, float]:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if len(y_true) == 0:
        return {l: float("nan") for l in labels}
    keep = _valid_rows(y_true)
    yb, _ = binarize_targets(y_true[keep])
    return {l: auc_or_nan(yb[:, j], y_prob[keep, j])[0] for j, l in enumerate(labels)}


def confusion_counts(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    y, pred = y.astype(bool), pred.astype(bool)
    return {"tp": int((y & pred).sum()), "tn": int((~y & ~pred).sum()), "fp": int((~y & pred).sum()),
            "fn": int((y & ~pred).sum())}


def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    c = confusion_counts(y, p >= thr)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    prec, rec, spec = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn), _safe_div(tn, tn + fp)
    if tp == 0 and (fp + fn) > 0:
        f1 = 0.0  # defined: there were errors and no correct positive
    elif tp + fp + fn == 0:
        f1 = float("nan")  # no positives at all, none predicted: undefined
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)
    reasons = []
    if tp + fp == 0:
        reasons.append("no_predicted_positive")
    if tp + fn == 0:
        reasons.append("no_positives")
    if tn + fp == 0:
        reasons.append("no_negatives")
    return {**c, "threshold": float(thr), "precision": prec, "recall": rec, "sensitivity": rec, "specificity": spec,
            "f1": f1, "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
            "balanced_accuracy": float(np.nanmean([rec, spec])) if not (math.isnan(rec) and math.isnan(spec)) else float("nan"),
            "undefined": ",".join(reasons)}


@dataclass
class MetricsResult:
    per_label: pd.DataFrame
    aggregate: dict
    threshold_source: str
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"aggregate": self.aggregate, "per_label": self.per_label.to_dict("records"),
                "threshold_source": self.threshold_source, "notes": self.notes}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Mapping[str, float] | Sequence[float] | None = None,
                    labels: Sequence[str] = LABELS, threshold_source: str | None = None) -> MetricsResult:
    y_true, y_prob = np.asarray(y_true, dtype=np.float64), np.asarray(y_prob, dtype=np.float64)
    if y_true.shape != y_prob.shape or y_true.ndim != 2 or y_true.shape[1] != len(labels):
        raise ValueError(f"shape mismatch: y_true {y_true.shape}, y_prob {y_prob.shape}, labels {len(labels)}")
    notes = []
    keep = _valid_rows(y_true) & _valid_rows(y_prob)
    if (~keep).any():
        notes.append(f"{int((~keep).sum())} rows with missing targets/probabilities excluded")
    y_true, y_prob = y_true[keep], y_prob[keep]
    yb, nonbin = binarize_targets(y_true)
    if nonbin:
        notes.append("non-binary targets were binarised at 0.5 for metric computation")
    if thresholds is None:
        thr = np.full(len(labels), 0.5)
        threshold_source = threshold_source or "default_0.5 (NOT validation-selected)"
    elif isinstance(thresholds, Mapping):
        thr = np.array([float(thresholds[l]) for l in labels])
        threshold_source = threshold_source or "provided"
    else:
        thr = np.asarray(thresholds, dtype=float)
        threshold_source = threshold_source or "provided"

    rows = []
    for j, l in enumerate(labels):
        auc, auc_r = auc_or_nan(yb[:, j], y_prob[:, j])
        ap, ap_r = ap_or_nan(yb[:, j], y_prob[:, j])
        tm = threshold_metrics(yb[:, j], y_prob[:, j], thr[j])
        brier = float(np.mean((y_prob[:, j] - y_true[:, j]) ** 2)) if len(yb) else float("nan")
        rows.append({"label": l, "n": int(len(yb)), "n_pos": int(yb[:, j].sum()), "prevalence": float(yb[:, j].mean()) if len(yb) else float("nan"),
                     "roc_auc": auc, "roc_auc_undefined": auc_r, "average_precision": ap, "ap_undefined": ap_r,
                     "brier": brier, **tm})
    per = pd.DataFrame(rows)

    def nanmean(col):
        v = per[col].to_numpy(dtype=float)
        d = v[~np.isnan(v)]
        return (float(d.mean()) if len(d) else float("nan")), int(len(d))

    agg = {"n_images": int(len(yb))}
    for col, name in [("roc_auc", "macro_roc_auc"), ("average_precision", "macro_ap"), ("f1", "macro_f1"),
                      ("precision", "macro_precision"), ("recall", "macro_recall"),
                      ("specificity", "macro_specificity"), ("accuracy", "macro_accuracy"), ("brier", "mean_brier")]:
        agg[name], agg[f"{name}_n_defined"] = nanmean(col)
    flat_y, flat_p = yb.ravel(), y_prob.ravel()
    agg["micro_roc_auc"] = auc_or_nan(flat_y, flat_p)[0]
    agg["micro_ap"] = ap_or_nan(flat_y, flat_p)[0]
    pred = (y_prob >= thr[None, :]).astype(np.int8)
    agg["hamming_loss"] = float(np.mean(pred != yb)) if len(yb) else float("nan")
    agg["exact_match_accuracy"] = float(np.mean((pred == yb).all(axis=1))) if len(yb) else float("nan")
    c = confusion_counts(flat_y, pred.ravel())
    agg["micro_precision"] = _safe_div(c["tp"], c["tp"] + c["fp"])
    agg["micro_recall"] = _safe_div(c["tp"], c["tp"] + c["fn"])
    agg["micro_f1"] = _safe_div(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"])
    agg["rsna_weighted_logloss"] = rsna_weighted_logloss(y_true, y_prob) if len(yb) else float("nan")
    if "any" in labels:
        j = list(labels).index("any")
        agg["any_roc_auc"] = per.loc[j, "roc_auc"]
    agg["thresholds"] = dict(zip(labels, thr.tolist()))
    return MetricsResult(per, agg, threshold_source, notes)


def rsna_weighted_logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    """Competition metric: per-label log loss weighted 2x for ``any``, averaged per image."""
    p = np.clip(p, eps, 1 - eps)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float((ll * RSNA_WEIGHTS).sum(axis=1).mean() / RSNA_WEIGHTS.sum())


def confusion_matrices(y_true: np.ndarray, y_prob: np.ndarray, thresholds, labels=LABELS) -> dict[str, np.ndarray]:
    yb, _ = binarize_targets(np.asarray(y_true))
    thr = [thresholds[l] for l in labels] if isinstance(thresholds, Mapping) else list(thresholds)
    out = {}
    for j, l in enumerate(labels):
        c = confusion_counts(yb[:, j], np.asarray(y_prob)[:, j] >= thr[j])
        out[l] = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    return out
