"""Competition-format output for ``stage_2_test`` (inference only; there are no labels)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import LABELS
from ..utils.io import atomic_save_dataframe


def make_submission(pred: pd.DataFrame, sample_submission_csv: str | Path, clip: float = 1e-6) -> pd.DataFrame:
    """Long format ``ID_<image>_<label>, Label`` in the order of the sample submission.

    Missing images (e.g. undecodable DICOMs) are filled with the neutral value 0.5 **and counted**
    in ``out.attrs['filled']`` so the caller can report them.
    """
    sample = pd.read_csv(sample_submission_csv, dtype={"ID": str})
    long = pred.melt(id_vars="image_id", value_vars=[f"prob_{l}" for l in LABELS], var_name="label", value_name="Label")
    long["ID"] = long["image_id"] + "_" + long["label"].str.replace("prob_", "", regex=False)
    out = sample[["ID"]].merge(long[["ID", "Label"]], on="ID", how="left")
    n_missing = int(out["Label"].isna().sum())
    out["Label"] = out["Label"].fillna(0.5).clip(clip, 1 - clip)
    out.attrs["filled"] = n_missing
    return out


def save_submission(sub: pd.DataFrame, path: str | Path) -> Path:
    if sub["Label"].isna().any() or not np.isfinite(sub["Label"]).all():
        raise ValueError("submission contains invalid probabilities")
    return atomic_save_dataframe(sub, Path(path))
