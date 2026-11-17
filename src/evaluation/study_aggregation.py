"""Slice-level -> study-level aggregation for ICH triage (research output, not a diagnosis).

Study label (for evaluation) = max over slices of the slice labels. Study scores are
aggregates of slice probabilities. High-risk slices are ranked by ``any`` probability and
reported in anatomical order (slice_rank) when metadata is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import LABELS

AGGREGATORS = {
    "max": lambda v: float(np.max(v)),
    "mean": lambda v: float(np.mean(v)),
    "top3_mean": lambda v: float(np.mean(np.sort(v)[-3:])),
    "noisy_or": lambda v: float(1 - np.prod(1 - np.clip(v, 0, 1 - 1e-6))),
}


def aggregate_studies(pred: pd.DataFrame, method: str = "max", study_col: str = "study_uid") -> pd.DataFrame:
    """``pred`` has image_id, study_uid, patient_id, prob_<label> and optionally target_<label>."""
    if method not in AGGREGATORS:
        raise ValueError(f"method must be one of {list(AGGREGATORS)}")
    f = AGGREGATORS[method]
    g = pred.groupby(study_col)
    out = pd.DataFrame({study_col: list(g.groups.keys())})
    out["patient_id"] = g["patient_id"].first().to_numpy() if "patient_id" in pred else ""
    out["n_slices"] = g.size().to_numpy()
    for l in LABELS:
        out[f"prob_{l}"] = g[f"prob_{l}"].apply(lambda s: f(s.to_numpy())).to_numpy()
        if f"target_{l}" in pred:
            out[f"target_{l}"] = g[f"target_{l}"].max().to_numpy()
    out["aggregation"] = method
    return out


def high_risk_slices(pred: pd.DataFrame, study_uid: str, top_k: int = 5, study_col: str = "study_uid") -> pd.DataFrame:
    s = pred[pred[study_col] == study_uid].copy()
    s = s.sort_values("prob_any", ascending=False).head(top_k)
    if "slice_rank" in s:
        s = s.sort_values("slice_rank")
    cols = [c for c in ["image_id", "slice_rank", *[f"prob_{l}" for l in LABELS]] if c in s.columns]
    return s[cols].reset_index(drop=True)


def triage_table(study_pred: pd.DataFrame, any_threshold: float) -> pd.DataFrame:
    """Rank studies by aggregated ``any`` probability and flag those above a (validation-selected) threshold."""
    t = study_pred.copy()
    t["flag_any"] = t["prob_any"] >= any_threshold
    sub = [f"prob_{l}" for l in LABELS[:-1]]
    t["most_likely_subtype"] = t[sub].idxmax(axis=1).str.replace("prob_", "", regex=False)
    t["disclaimer"] = "model output for research; not a clinical diagnosis"
    return t.sort_values("prob_any", ascending=False).reset_index(drop=True)
