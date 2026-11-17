"""Patient-level statistical inference.

Slices from the same patient are strongly correlated, so they are **not** independent
observations. All confidence intervals and paired comparisons here resample *patients*
(clusters) with replacement - the patient-cluster bootstrap - and carry every slice of a
sampled patient along. Assumptions: patients are independent and identically sampled from
the population of interest; percentile intervals are used (they can be slightly
anti-conservative for small numbers of positive patients, which is reported as ``n_valid``).

DeLong's test is provided for reference only: it assumes independent samples and is therefore
anti-conservative with slice-level data.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import LABELS
from ..data.csv_parser import binarize_targets


def make_clusters(groups: Sequence) -> list[np.ndarray]:
    codes, uniq = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def bootstrap_indices(groups: Sequence, n_boot: int, seed: int = 0):
    """Yield index arrays of patient-cluster bootstrap resamples."""
    clusters = make_clusters(groups)
    rng = np.random.default_rng(seed)
    k = len(clusters)
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        yield np.concatenate([clusters[i] for i in pick])


def _macro_auc(yb: np.ndarray, p: np.ndarray) -> float:
    vals = [roc_auc_score(yb[:, j], p[:, j]) for j in range(yb.shape[1]) if 0 < yb[:, j].sum() < len(yb)]
    return float(np.mean(vals)) if vals else float("nan")


def metric_fn_from_name(name: str, labels: Sequence[str] = LABELS) -> Callable[[np.ndarray, np.ndarray], float]:
    """``macro_auc`` | ``macro_ap`` | ``auc:<label>`` | ``ap:<label>``."""
    if name == "macro_auc":
        return _macro_auc
    if name == "macro_ap":
        def f(yb, p):
            vals = [average_precision_score(yb[:, j], p[:, j]) for j in range(yb.shape[1]) if 0 < yb[:, j].sum() < len(yb)]
            return float(np.mean(vals)) if vals else float("nan")
        return f
    kind, _, lab = name.partition(":")
    j = list(labels).index(lab)
    if kind == "auc":
        return lambda yb, p: float(roc_auc_score(yb[:, j], p[:, j])) if 0 < yb[:, j].sum() < len(yb) else float("nan")
    if kind == "ap":
        return lambda yb, p: float(average_precision_score(yb[:, j], p[:, j])) if 0 < yb[:, j].sum() < len(yb) else float("nan")
    raise ValueError(f"unknown metric {name!r}")


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, groups: Sequence, metric: str = "macro_auc",
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    yb, _ = binarize_targets(np.asarray(y_true))
    p = np.asarray(y_prob)
    f = metric_fn_from_name(metric)
    est = f(yb, p)
    vals = []
    for idx in bootstrap_indices(groups, n_boot, seed):
        v = f(yb[idx], p[idx])
        if v == v:
            vals.append(v)
    vals = np.asarray(vals)
    lo, hi = (np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(vals) else (np.nan, np.nan))
    return {"metric": metric, "estimate": est, "ci_low": float(lo), "ci_high": float(hi), "n_boot": n_boot,
            "n_valid": int(len(vals)), "n_clusters": int(len(set(groups))), "method": "patient-cluster percentile bootstrap"}


def bootstrap_table(y_true, y_prob, groups, metrics: Sequence[str] | None = None, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    metrics = metrics or ["macro_auc", "macro_ap", *[f"auc:{l}" for l in LABELS], *[f"ap:{l}" for l in LABELS]]
    yb, _ = binarize_targets(np.asarray(y_true))
    p = np.asarray(y_prob)
    fns = {m: metric_fn_from_name(m) for m in metrics}
    est = {m: f(yb, p) for m, f in fns.items()}
    acc = {m: [] for m in metrics}
    for idx in bootstrap_indices(groups, n_boot, seed):
        yy, pp = yb[idx], p[idx]
        for m, f in fns.items():
            v = f(yy, pp)
            if v == v:
                acc[m].append(v)
    rows = []
    for m in metrics:
        v = np.asarray(acc[m])
        lo, hi = (np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan))
        rows.append({"metric": m, "estimate": est[m], "ci_low": float(lo), "ci_high": float(hi), "n_valid": len(v)})
    return pd.DataFrame(rows)


def paired_bootstrap(y_true, prob_a, prob_b, groups, metric: str = "macro_auc", n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired patient-cluster bootstrap of metric(a) - metric(b) on identical images."""
    yb, _ = binarize_targets(np.asarray(y_true))
    pa, pb = np.asarray(prob_a), np.asarray(prob_b)
    if pa.shape != pb.shape or len(pa) != len(yb):
        raise ValueError("paired comparison requires predictions for the same images in the same order")
    f = metric_fn_from_name(metric)
    obs = f(yb, pa) - f(yb, pb)
    d = []
    for idx in bootstrap_indices(groups, n_boot, seed):
        a, b = f(yb[idx], pa[idx]), f(yb[idx], pb[idx])
        if a == a and b == b:
            d.append(a - b)
    d = np.asarray(d)
    if len(d) == 0:
        return {"metric": metric, "difference": obs, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan, "n_valid": 0}
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    p = float(min(1.0, max(p, 1.0 / len(d))))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"metric": metric, "difference": float(obs), "ci_low": float(lo), "ci_high": float(hi), "p_value": p,
            "n_valid": int(len(d)), "method": "paired patient-cluster bootstrap"}


def holm(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    adj = np.empty_like(pv)
    running = 0.0
    m = len(pv)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pv[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def delong_test(y: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> dict:
    """DeLong test for two correlated AUCs (assumes independent samples; reference only)."""
    y = np.asarray(y).astype(int)
    m, n = int(y.sum()), int(len(y) - y.sum())
    if m == 0 or n == 0:
        return {"difference": np.nan, "z": np.nan, "p_value": np.nan}
    order = np.argsort(-y, kind="stable")
    preds = np.vstack([pa, pb])[:, order]
    tx = np.array([stats.rankdata(r[:m]) for r in preds])
    ty = np.array([stats.rankdata(r[m:]) for r in preds])
    tz = np.array([stats.rankdata(r) for r in preds])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"difference": float(aucs[0] - aucs[1]), "z": np.nan, "p_value": 1.0}
    z = (aucs[0] - aucs[1]) / math.sqrt(var)
    return {"difference": float(aucs[0] - aucs[1]), "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z))),
            "note": "assumes independent samples; anti-conservative for correlated slices"}


def seed_summary(values: Sequence[float], alpha: float = 0.05) -> dict:
    """Mean, SD and t-interval across independent training seeds."""
    v = np.asarray([x for x in values if x == x], dtype=float)
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if len(v) == 1:
        return {"n": 1, "mean": float(v[0]), "sd": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    sd = v.std(ddof=1)
    h = stats.t.ppf(1 - alpha / 2, len(v) - 1) * sd / math.sqrt(len(v))
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(sd), "ci_low": float(v.mean() - h),
            "ci_high": float(v.mean() + h)}


def paired_seed_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired comparison across seeds (same seed = same data order/initialisation of shared parts)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = {"n": int(len(d)), "mean_difference": float(np.mean(d)) if len(d) else np.nan}
    if len(d) >= 2 and np.std(d) > 0:
        t = stats.ttest_rel(a, b)
        out.update(t_stat=float(t.statistic), p_ttest=float(t.pvalue))
        try:
            w = stats.wilcoxon(a, b)
            out.update(p_wilcoxon=float(w.pvalue))
        except ValueError:
            out.update(p_wilcoxon=np.nan)
    return out
