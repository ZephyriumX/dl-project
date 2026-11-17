"""Append-only experiment registry (JSON lines).

Every experiment is recorded with an explicit status so that failed, skipped and never-run
experiments are visible in the comparative analysis instead of silently disappearing.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from .io import _clean_nans, _json_default

STATUSES = ("planned", "running", "succeeded", "failed", "skipped")


class ExperimentRegistry:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, experiment_id: str, status: str, **fields: Any) -> dict:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        rec = {"experiment_id": experiment_id, "status": status, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
        line = json.dumps(_clean_nans(json.loads(json.dumps(rec, default=_json_default))))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return rec

    def record_failure(self, experiment_id: str, exc: BaseException, **fields: Any) -> dict:
        return self.record(experiment_id, "failed", error=f"{type(exc).__name__}: {exc}",
                           traceback=traceback.format_exc(limit=5), **fields)

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=["experiment_id", "status", "timestamp"])
        rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # a torn last line from an interrupted session
        return pd.DataFrame(rows)

    def latest(self) -> pd.DataFrame:
        """Most recent record per experiment_id."""
        df = self.read()
        if df.empty:
            return df
        return df.groupby("experiment_id", as_index=False).tail(1).reset_index(drop=True)

    def status_of(self, experiment_id: str) -> str | None:
        df = self.latest()
        if df.empty or experiment_id not in set(df["experiment_id"]):
            return None
        return str(df.loc[df["experiment_id"] == experiment_id, "status"].iloc[-1])
