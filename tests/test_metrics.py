"""Metrics, thresholds, statistics, study aggregation, submission formatting."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src import LABELS
from src.evaluation import (ThresholdLeakageError, ThresholdSet, aggregate_studies, bootstrap_ci, compute_metrics,
                            delong_test, holm, paired_bootstrap, select_thresholds, seed_summary)
from src.evaluation.metrics import rsna_weighted_logloss, threshold_metrics


def _synthetic(n=400, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 6)) < [0.02, 0.05, 0.04, 0.05, 0.06, 0.0]).astype(float)
    y[:, 5] = y[:, :5].max(1)
    logits = rng.normal(size=(n, 6)) + signal * (2 * y - 1)
    p = 1 / (1 + np.exp(-logits))
    groups = np.repeat(np.arange(n // 4), 4)
    return y, p, groups


def test_metrics_match_sklearn():
    y, p, _ = _synthetic()
    m = compute_metrics(y, p)
    for j, l in enumerate(LABELS):
        row = m.per_label.set_index("label").loc[l]
        if 0 < y[:, j].sum() < len(y):
            assert row["roc_auc"] == pytest.approx(roc_auc_score(y[:, j], p[:, j]))
            assert row["average_precision"] == pytest.approx(average_precision_score(y[:, j], p[:, j]))
    assert m.aggregate["micro_roc_auc"] == pytest.approx(roc_auc_score(y.ravel(), p.ravel()))
    assert "default_0.5" in m.threshold_source
    for k in ("macro_roc_auc", "macro_ap", "hamming_loss", "exact_match_accuracy", "mean_brier", "micro_f1"):
        assert np.isfinite(m.aggregate[k])


def test_undefined_metrics_are_nan_not_zero():
    y = np.zeros((10, 6))
    y[:5, 5] = 1
    y[:5, 4] = 1
    p = np.full((10, 6), 0.1)
    m = compute_metrics(y, p)
    per = m.per_label.set_index("label")
    assert math.isnan(per.loc["epidural", "roc_auc"]) and per.loc["epidural", "roc_auc_undefined"] == "single_class"
    assert math.isnan(per.loc["epidural", "average_precision"])
    assert math.isnan(per.loc["epidural", "recall"])  # no positives
    assert math.isnan(per.loc["any", "precision"])  # nothing predicted positive
    assert m.aggregate["macro_roc_auc_n_defined"] == 2


def test_threshold_metrics_counts():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1, 0.2])
    t = threshold_metrics(y, p, 0.5)
    assert (t["tp"], t["fn"], t["fp"], t["tn"]) == (1, 1, 1, 2)
    assert t["precision"] == 0.5 and t["recall"] == 0.5 and t["specificity"] == pytest.approx(2 / 3)
    assert t["f1"] == pytest.approx(0.5) and t["accuracy"] == pytest.approx(0.6)


def test_thresholds_val_only_and_frozen(tmp_path):
    y, p, _ = _synthetic()
    with pytest.raises(ThresholdLeakageError):
        select_thresholds(y, p, split="test", manifest_hash="h", experiment_id="e")
    ts = select_thresholds(y, p, split="val", manifest_hash="h", experiment_id="e")
    assert set(ts.thresholds) == set(LABELS) and all(0 <= v <= 1 for v in ts.thresholds.values())
    ts.save(tmp_path / "t.json")
    assert ThresholdSet.load(tmp_path / "t.json", "h").thresholds == ts.thresholds
    with pytest.raises(ThresholdLeakageError):
        ThresholdSet.load(tmp_path / "t.json", "different")
    for method in ("f1", "sensitivity", "fixed"):
        assert select_thresholds(y, p, split="val", manifest_hash="h", experiment_id="e", method=method).thresholds


def test_cluster_bootstrap_ci_and_paired():
    y, p, g = _synthetic(800)
    ci = bootstrap_ci(y, p, g, "macro_auc", n_boot=100, seed=0)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"] and ci["n_clusters"] == 200
    same = paired_bootstrap(y, p, p, g, n_boot=50)
    assert same["difference"] == 0 and same["p_value"] == 1.0
    _, worse, _ = _synthetic(800, seed=0, signal=0.2)
    d = paired_bootstrap(y, p, worse, g, n_boot=100)
    assert d["difference"] > 0 and d["ci_low"] > 0 and d["p_value"] < 0.05
    with pytest.raises(ValueError):
        paired_bootstrap(y, p, p[:-1], g)


def test_holm_and_delong_and_seed_summary():
    adj = holm([0.01, 0.04, 0.03, np.nan])
    assert np.isnan(adj[3]) and adj[0] == pytest.approx(0.03) and (adj[:3] <= 1).all()
    y, p, _ = _synthetic()
    r = delong_test(y[:, 5], p[:, 5], p[:, 5])
    assert r["difference"] == pytest.approx(0) and r["p_value"] == pytest.approx(1.0)
    s = seed_summary([0.8, 0.82, 0.81])
    assert s["n"] == 3 and s["ci_low"] < s["mean"] < s["ci_high"]
    assert math.isnan(seed_summary([0.8])["sd"])


def test_study_aggregation():
    df = pd.DataFrame({"image_id": list("abcd"), "study_uid": ["s1", "s1", "s2", "s2"], "patient_id": ["p1", "p1", "p2", "p2"],
                       **{f"prob_{l}": [0.1, 0.9, 0.2, 0.3] for l in LABELS},
                       **{f"target_{l}": [0, 1, 0, 0] for l in LABELS}})
    s = aggregate_studies(df, "max").set_index("study_uid")
    assert s.loc["s1", "prob_any"] == 0.9 and s.loc["s1", "target_any"] == 1 and s.loc["s2", "target_any"] == 0
    nor = aggregate_studies(df, "noisy_or").set_index("study_uid")
    assert nor.loc["s2", "prob_any"] == pytest.approx(1 - 0.8 * 0.7)


def test_rsna_logloss_and_submission(tmp_path):
    from src.evaluation.predictions import predictions_frame
    from src.evaluation.submission import make_submission, save_submission

    y = np.array([[0, 0, 0, 0, 1, 1.0]])
    assert rsna_weighted_logloss(y, np.full((1, 6), 0.5)) == pytest.approx(math.log(2))
    sample = pd.DataFrame({"ID": [f"ID_x_{l}" for l in LABELS] + [f"ID_y_{l}" for l in LABELS], "Label": 0.5})
    sample.to_csv(tmp_path / "s.csv", index=False)
    pred = predictions_frame(["ID_x"], np.full((1, 6), 0.2))
    sub = make_submission(pred, tmp_path / "s.csv")
    assert list(sub["ID"]) == list(sample["ID"]) and sub.attrs["filled"] == 6
    assert (sub["Label"].iloc[:6] == 0.2).all()
    save_submission(sub, tmp_path / "sub.csv")
