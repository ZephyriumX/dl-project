"""Atomic, restart-safe file I/O helpers.

Every artifact is written to ``<name>.tmp`` first and then renamed, so an interrupted Kaggle
session never leaves a half-written file that later stages would trust.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def _clean_nans(obj: Any):
    """Replace float NaN/inf with None so the JSON is standard-compliant (NaN is not valid JSON)."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nans(v) for v in obj]
    return obj


def atomic_write_text(path: str | os.PathLike, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def atomic_write_json(path: str | os.PathLike, obj: Any, indent: int = 2) -> Path:
    obj = json.loads(json.dumps(obj, default=_json_default))  # normalise numpy types first
    return atomic_write_text(path, json.dumps(_clean_nans(obj), indent=indent, sort_keys=False))


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save_dataframe(df: pd.DataFrame, path: str | os.PathLike) -> Path:
    """Save a DataFrame as CSV or Parquet (by suffix) atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".parquet":
        df.to_parquet(tmp, index=False)
    elif path.suffix in (".csv", ".gz"):
        df.to_csv(tmp, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")
    os.replace(tmp, path)
    return path


def load_dataframe(path: str | os.PathLike) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Minimal GitHub-markdown table writer (avoids an optional ``tabulate`` dependency)."""

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "NaN" if v != v else format(float(v), floatfmt)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
