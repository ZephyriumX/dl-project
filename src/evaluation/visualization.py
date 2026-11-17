"""Plotting helpers (matplotlib). Every function returns the Figure and optionally saves it."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .. import LABELS
from ..data.csv_parser import binarize_targets

PALETTE = ["#2a6fbb", "#d1495b", "#3c9d5d", "#e08e0b", "#7b5ea7", "#4d4d4d", "#00a6a6", "#b5651d"]


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def plot_label_distribution(dist: pd.DataFrame, path=None, title="Label prevalence by split"):
    """``dist`` from ``label_distribution_table(manifest, by='split')``."""
    fig, ax = plt.subplots(figsize=(9, 4))
    splits = list(dist["split"].unique())
    w = 0.8 / max(len(splits), 1)
    for i, s in enumerate(splits):
        d = dist[dist["split"] == s].set_index("label").loc[list(LABELS)]
        ax.bar(np.arange(len(LABELS)) + i * w, d["prevalence"], width=w, label=f"{s} (n={int(d['n_images'].iloc[0]):,})",
               color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(np.arange(len(LABELS)) + w * (len(splits) - 1) / 2)
    ax.set_xticklabels(LABELS, rotation=20)
    ax.set_ylabel("slice-level prevalence")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_windows(channels: np.ndarray, names: Sequence[str], title: str = "", path=None):
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4))
    axes = np.atleast_1d(axes)
    for a, ch, n in zip(axes, channels, names):
        a.imshow(ch, cmap="gray", vmin=0, vmax=1)
        a.set_title(n)
        a.axis("off")
    fig.suptitle(title)
    return _save(fig, path)


def plot_history(history: pd.DataFrame, path=None, title="Training history"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(history["epoch"] + 1, history["train_loss"], "o-", label="train", color=PALETTE[0])
    ax[0].plot(history["epoch"] + 1, history["val_loss"], "o-", label="validation", color=PALETTE[1])
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(history["epoch"] + 1, history["val_macro_auc"], "o-", color=PALETTE[2])
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation macro ROC-AUC"); ax[1].grid(alpha=0.3)
    fig.suptitle(title)
    return _save(fig, path)


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, path=None, title=""):
    yb, _ = binarize_targets(np.asarray(y_true))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for j, l in enumerate(LABELS):
        if len(np.unique(yb[:, j])) < 2:
            continue
        fpr, tpr, _ = roc_curve(yb[:, j], y_prob[:, j])
        pr, rc, _ = precision_recall_curve(yb[:, j], y_prob[:, j])
        ax[0].plot(fpr, tpr, label=l, color=PALETTE[j])
        ax[1].plot(rc, pr, label=l, color=PALETTE[j])
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC")
    ax[1].set(xlabel="recall", ylabel="precision", title="Precision-recall")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(title)
    return _save(fig, path)


def plot_confusion_matrices(cms: Mapping[str, np.ndarray], path=None, title=""):
    fig, axes = plt.subplots(1, len(cms), figsize=(2.6 * len(cms), 2.8))
    for a, (l, cm) in zip(np.atleast_1d(axes), cms.items()):
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8)
        a.set_xticks([0, 1]); a.set_yticks([0, 1])
        a.set_xticklabels(["pred 0", "pred 1"], fontsize=7); a.set_yticklabels(["true 0", "true 1"], fontsize=7)
        a.set_title(l, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)


def plot_forest(table: pd.DataFrame, label_col: str = "model", path=None, title="", xlabel="macro ROC-AUC (95% CI)"):
    """``table`` with columns label_col, estimate, ci_low, ci_high."""
    t = table.dropna(subset=["estimate"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(t) + 1.2))
    y = np.arange(len(t))[::-1]
    ax.errorbar(t["estimate"], y, xerr=[t["estimate"] - t["ci_low"], t["ci_high"] - t["estimate"]], fmt="o",
                color=PALETTE[0], capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(t[label_col])
    ax.set_xlabel(xlabel); ax.set_title(title); ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def overlay_cam(image: np.ndarray, cam: np.ndarray, ax=None, alpha: float = 0.4, title: str = ""):
    """``image`` (H,W) in [0,1] (e.g. brain window), ``cam`` (H,W) in [0,1]."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.imshow(cam, cmap="jet", alpha=alpha, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    return ax.figure


def plot_seed_comparison(df: pd.DataFrame, x: str = "head", y: str = "test_macro_auc", path=None, title=""):
    """Dot plot of per-seed results per configuration (shows variability, not only means)."""
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * df[x].nunique()), 4))
    cats = list(dict.fromkeys(df[x]))
    for i, c in enumerate(cats):
        v = df.loc[df[x] == c, y].to_numpy(float)
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.08, 0.08, len(v)), v, color=PALETTE[i % 8], s=22)
        if len(v):
            ax.hlines(np.nanmean(v), i - 0.25, i + 0.25, color="k", lw=1.5)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.set_ylabel(y); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)
