"""Parsing and validation of ``stage_2_train.csv`` / ``stage_2_sample_submission.csv``.

The RSNA label CSV is in *long* format: one row per (image, label), with the label name
appended to the image identifier, e.g. ``ID_000012eaf_epidural``. This module

* validates the identifier format (``ID_<hex>_<label>``) and reports malformed rows,
* removes exact duplicate rows (counted) and reports *conflicting* duplicates,
* pivots to one six-label vector per image in the canonical label order,
* preserves the original values (fractional values are kept and reported, not rounded),
* checks the semantic consistency of ``any`` (``any == max(subtypes)``),
* produces a label-distribution report.

The parsed table is cached (parquet) so the 4.5M-row CSV is parsed once per session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import LABELS, SUBTYPES

ID_PATTERN = re.compile(r"^(ID_[0-9a-fA-F]+)_(" + "|".join(LABELS) + r")$")


class LabelParsingError(ValueError):
    pass


@dataclass
class LabelReport:
    n_rows_raw: int = 0
    n_rows_malformed: int = 0
    malformed_examples: list[str] = field(default_factory=list)
    n_exact_duplicate_rows: int = 0
    n_conflicting_duplicate_pairs: int = 0
    conflicting_examples: list[dict] = field(default_factory=list)
    conflict_policy: str = "max"
    n_images: int = 0
    n_images_missing_labels: int = 0
    unique_values: list[float] = field(default_factory=list)
    has_fractional_values: bool = False
    n_any_inconsistent_subtype_without_any: int = 0
    n_any_inconsistent_any_without_subtype: int = 0
    positives: dict[str, int] = field(default_factory=dict)
    prevalence: dict[str, float] = field(default_factory=dict)
    n_labels_per_positive_image: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def split_id_column(ids: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Split ``ID_<hex>_<label>`` into (image_id, label, valid_mask) without assuming validity."""
    extracted = ids.astype(str).str.extract(ID_PATTERN)
    valid = extracted[0].notna()
    return extracted[0], extracted[1], valid


def parse_label_csv(csv_path: str | Path, conflict_policy: str = "max",
                    max_examples: int = 20) -> tuple[pd.DataFrame, LabelReport]:
    """Parse the long-format RSNA training CSV into one row per image.

    Parameters
    ----------
    conflict_policy:
        How to resolve duplicate (image, label) rows with *different* values:
        ``"max"`` (positive wins; default), ``"min"``, ``"drop_image"`` (remove the image),
        or ``"error"`` (raise). The resolution is always reported.

    Returns
    -------
    labels : DataFrame with columns ``image_id`` + LABELS (float32, original values preserved).
    report : LabelReport
    """
    if conflict_policy not in {"max", "min", "drop_image", "error"}:
        raise ValueError(f"unknown conflict_policy {conflict_policy!r}")
    raw = pd.read_csv(csv_path, dtype={"ID": str})
    if not {"ID", "Label"}.issubset(raw.columns):
        raise LabelParsingError(f"{csv_path} must have columns ID, Label; found {list(raw.columns)}")
    rep = LabelReport(n_rows_raw=len(raw), conflict_policy=conflict_policy)

    image_id, label, valid = split_id_column(raw["ID"])
    values = pd.to_numeric(raw["Label"], errors="coerce")
    bad = (~valid) | values.isna()
    rep.n_rows_malformed = int(bad.sum())
    rep.malformed_examples = raw.loc[bad, "ID"].astype(str).head(max_examples).tolist()
    long = pd.DataFrame({"image_id": image_id[~bad], "label": label[~bad], "value": values[~bad].astype(np.float64)})

    # exact duplicates (same image, label, value)
    dup_exact = long.duplicated(["image_id", "label", "value"], keep="first")
    rep.n_exact_duplicate_rows = int(dup_exact.sum())
    long = long.loc[~dup_exact]

    # conflicting duplicates (same image+label, different value)
    dup_key = long.duplicated(["image_id", "label"], keep=False)
    conflicts = long.loc[dup_key]
    if len(conflicts):
        rep.n_conflicting_duplicate_pairs = int(conflicts.groupby(["image_id", "label"]).ngroups)
        rep.conflicting_examples = (conflicts.groupby(["image_id", "label"])["value"].apply(list)
                                    .reset_index().head(max_examples).to_dict("records"))
        if conflict_policy == "error":
            raise LabelParsingError(f"{rep.n_conflicting_duplicate_pairs} conflicting duplicate label rows")
    agg = {"max": "max", "min": "min", "drop_image": "max", "error": "max"}[conflict_policy]
    wide = long.pivot_table(index="image_id", columns="label", values="value", aggfunc=agg)
    if conflict_policy == "drop_image" and len(conflicts):
        wide = wide.drop(index=conflicts["image_id"].unique(), errors="ignore")

    for c in LABELS:
        if c not in wide.columns:
            wide[c] = np.nan
    wide = wide[list(LABELS)]
    missing = wide.isna().any(axis=1)
    rep.n_images_missing_labels = int(missing.sum())
    wide = wide.loc[~missing].astype(np.float32).reset_index()
    wide.columns.name = None

    vals = np.unique(wide[list(LABELS)].to_numpy())
    rep.unique_values = [float(v) for v in vals[:50]]
    rep.has_fractional_values = bool(np.any((vals > 0) & (vals < 1)))
    rep.n_images = len(wide)

    sub_max = wide[list(SUBTYPES)].max(axis=1)
    rep.n_any_inconsistent_subtype_without_any = int(((sub_max >= 0.5) & (wide["any"] < 0.5)).sum())
    rep.n_any_inconsistent_any_without_subtype = int(((sub_max < 0.5) & (wide["any"] >= 0.5)).sum())
    pos = wide[list(LABELS)] >= 0.5
    rep.positives = {c: int(pos[c].sum()) for c in LABELS}
    rep.prevalence = {c: float(pos[c].mean()) for c in LABELS}
    n_sub = pos[list(SUBTYPES)].sum(axis=1)
    rep.n_labels_per_positive_image = {str(k): int(v) for k, v in n_sub[pos["any"]].value_counts().sort_index().items()}
    return wide, rep


def parse_submission_ids(csv_path: str | Path) -> pd.DataFrame:
    """Return unique competition-test image IDs from the sample submission (no labels!)."""
    raw = pd.read_csv(csv_path, dtype={"ID": str})
    image_id, label, valid = split_id_column(raw["ID"])
    if (~valid).any():
        raise LabelParsingError(f"{int((~valid).sum())} malformed IDs in {csv_path}")
    ids = pd.DataFrame({"image_id": image_id.unique()})
    return ids


def binarize_targets(y: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, bool]:
    """Binarise targets for metrics; returns (binary, was_non_binary)."""
    y = np.asarray(y, dtype=np.float64)
    non_binary = bool(np.any((y != 0) & (y != 1)))
    return (y >= threshold).astype(np.int8), non_binary


def label_distribution_table(labels: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Positives and prevalence per label (optionally grouped, e.g. by split)."""
    pos = (labels[list(LABELS)] >= 0.5)
    if by is None:
        return pd.DataFrame({"n_images": len(labels), "positives": pos.sum(), "prevalence": pos.mean()})
    out = []
    for key, idx in labels.groupby(by).groups.items():
        p = pos.loc[idx]
        for c in LABELS:
            out.append({by: key, "label": c, "n_images": len(p), "positives": int(p[c].sum()), "prevalence": float(p[c].mean())})
    return pd.DataFrame(out)
