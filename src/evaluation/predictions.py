"""Prediction serialisation (one file format for every model) and alignment for paired tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import LABELS
from ..utils.io import atomic_save_dataframe, atomic_write_json, read_json


def predictions_frame(image_ids, probs, targets=None, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"image_id": list(image_ids)})
    probs = np.asarray(probs, dtype=np.float32)
    for j, l in enumerate(LABELS):
        df[f"prob_{l}"] = probs[:, j]
    if targets is not None:
        t = np.asarray(targets, dtype=np.float32)
        for j, l in enumerate(LABELS):
            df[f"target_{l}"] = t[:, j]
    if extra is not None:
        df = df.merge(extra, on="image_id", how="left")
    return df


def save_predictions(df: pd.DataFrame, path: str | Path, meta: dict) -> Path:
    """Write predictions (parquet) + a sidecar JSON with manifest hash, experiment and split."""
    for k in ("manifest_hash", "experiment_id", "split"):
        if k not in meta:
            raise ValueError(f"prediction meta requires {k!r}")
    path = Path(path)
    atomic_save_dataframe(df, path)
    atomic_write_json(path.with_suffix(".json"), {**meta, "n_rows": int(len(df))})
    return path


def load_predictions(path: str | Path, expected_manifest_hash: str | None = None) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    meta = read_json(path.with_suffix(".json")) if path.with_suffix(".json").exists() else {}
    if expected_manifest_hash and meta.get("manifest_hash") != expected_manifest_hash:
        raise ValueError(f"{path.name}: predictions were produced on a different split manifest")
    return df, meta


def arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p = df[[f"prob_{l}" for l in LABELS]].to_numpy(np.float64)
    t = df[[f"target_{l}" for l in LABELS]].to_numpy(np.float64) if f"target_{LABELS[0]}" in df else None
    return p, t


def align(dfs: dict[str, pd.DataFrame]) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Restrict several prediction frames to their common image_ids, in the same order."""
    common = None
    for d in dfs.values():
        s = set(d["image_id"])
        common = s if common is None else common & s
    ids = sorted(common or [])
    return ids, {k: d.set_index("image_id").loc[ids].reset_index() for k, d in dfs.items()}
