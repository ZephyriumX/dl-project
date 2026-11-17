"""Patient-level, stratified, reproducible train/val/test partitioning and the split manifest.

The manifest is created **once** (Notebook 00) and reused unchanged by every model. It is
identified by a SHA-256 hash of its sorted ``(image_id, split)`` pairs; checkpoints, predictions
and thresholds store that hash and refuse to be combined with a different manifest.

The held-out research test split is carved out of the *labelled* training DICOMs. The
competition ``stage_2_test`` folder has no labels and never appears in a manifest.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .. import LABELS
from ..utils.io import atomic_save_dataframe, atomic_write_json, read_json

SPLITS = ("train", "val", "test")


class LeakageError(RuntimeError):
    pass


class ManifestHashError(RuntimeError):
    pass


def _group_strata(df: pd.DataFrame, group_col: str) -> pd.Series:
    """Patient-level stratum: epidural (rarest label) > any other ICH > negative."""
    pos = (df[list(LABELS)] >= 0.5)
    g = pos.groupby(df[group_col]).max()
    strata = np.where(g["epidural"], "epidural", np.where(g["any"], "ich", "negative"))
    return pd.Series(strata, index=g.index, name="stratum")


def choose_grouping(df: pd.DataFrame, max_missing_patient_fraction: float = 0.01) -> tuple[str, str, float]:
    """Return (group_col, grouping_level, missing_fraction)."""
    pid = df["patient_id"].fillna("").astype(str)
    frac = float((pid == "").mean()) if len(df) else 1.0
    if frac <= max_missing_patient_fraction:
        return "patient_id", "patient", frac
    return "study_uid", "study_fallback", frac


def make_patient_split(df: pd.DataFrame, fractions: Mapping[str, float] | None = None, seed: int = 42,
                       max_groups: int | None = None, max_missing_patient_fraction: float = 0.01,
                       min_positive_group_fraction: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Create the split manifest.

    ``df`` must contain ``image_id``, LABELS, ``patient_id``, ``study_uid`` (and optionally
    ``series_uid``, ``slice_rank``...). Only rows with a readable header should be passed.

    ``max_groups`` subsamples *patients* (stratified) before splitting - used for pilot runs.
    ``min_positive_group_fraction`` optionally enriches the subsample with ICH-positive patients
    (applied only when subsampling; prevalence shift is recorded in the manifest info).
    """
    fractions = dict(fractions or {"train": 0.70, "val": 0.15, "test": 0.15})
    if set(fractions) != set(SPLITS) or abs(sum(fractions.values()) - 1) > 1e-6:
        raise ValueError(f"fractions must cover {SPLITS} and sum to 1: {fractions}")
    group_col, level, missing_frac = choose_grouping(df, max_missing_patient_fraction)
    d = df.copy()
    d[group_col] = d[group_col].fillna("").astype(str)
    n_excluded = int((d[group_col] == "").sum())
    d = d[d[group_col] != ""].copy()
    d["group_id"] = d[group_col]

    strata = _group_strata(d, "group_id")
    rng = np.random.default_rng(seed)
    info: dict = {"seed": seed, "fractions": fractions, "grouping_level": level, "group_column": group_col,
                  "missing_patient_id_fraction": missing_frac, "n_images_excluded_no_group": n_excluded}

    if max_groups is not None and len(strata) > max_groups:
        keep = []
        prop = strata.value_counts(normalize=True)
        if min_positive_group_fraction:
            pos_share = max(min_positive_group_fraction, 1 - prop.get("negative", 0))
            neg_share = 1 - pos_share
            pos_total = 1 - prop.get("negative", 0)
            target = {s: (prop[s] / pos_total * pos_share if s != "negative" else neg_share) for s in prop.index}
        else:
            target = prop.to_dict()
        for s, share in target.items():
            groups = strata.index[strata == s].to_numpy()
            k = min(len(groups), int(round(share * max_groups)))
            keep.extend(rng.choice(groups, size=k, replace=False).tolist())
        strata = strata.loc[sorted(keep)]
        d = d[d["group_id"].isin(set(keep))]
        info["subsampled_groups"] = int(len(keep))
        info["subsampling_enriched"] = bool(min_positive_group_fraction)

    assign = {}
    for s in sorted(strata.unique()):
        groups = np.sort(strata.index[strata == s].to_numpy())
        groups = rng.permutation(groups)
        n = len(groups)
        n_val = int(round(fractions["val"] * n))
        n_test = int(round(fractions["test"] * n))
        if n >= 3:  # guarantee every split gets at least one group of each stratum when possible
            n_val, n_test = max(n_val, 1), max(n_test, 1)
        n_train = n - n_val - n_test
        for g in groups[:n_train]:
            assign[g] = "train"
        for g in groups[n_train:n_train + n_val]:
            assign[g] = "val"
        for g in groups[n_train + n_val:]:
            assign[g] = "test"
    d["split"] = d["group_id"].map(assign)
    d["stratum"] = d["group_id"].map(strata)

    cols = ["image_id", "split", "group_id", "patient_id", "study_uid"]
    cols += [c for c in ("series_uid", "series_key", "slice_rank", "n_slices_series", "slice_pos", "instance_number")
             if c in d.columns]
    cols += list(LABELS) + ["stratum"]
    manifest = d[cols].sort_values(["split", "group_id", "image_id"]).reset_index(drop=True)
    info.update(split_summary(manifest))
    info["manifest_hash"] = manifest_hash(manifest)
    info["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    verify_no_leakage(manifest, level)
    return manifest, info


def manifest_hash(manifest: pd.DataFrame) -> str:
    pairs = manifest[["image_id", "split"]].astype(str).sort_values(["image_id", "split"])
    blob = "\n".join(pairs["image_id"] + "," + pairs["split"])
    return hashlib.sha256(blob.encode()).hexdigest()


def overlap_report(manifest: pd.DataFrame) -> dict:
    rep = {}
    for col in ("group_id", "patient_id", "study_uid", "image_id"):
        if col not in manifest.columns:
            continue
        sets = {s: set(manifest.loc[(manifest["split"] == s) & (manifest[col].astype(str) != ""), col]) for s in SPLITS}
        rep[col] = {f"{a}&{b}": len(sets[a] & sets[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
    rep["duplicate_image_ids"] = int(manifest["image_id"].duplicated().sum())
    return rep


def verify_no_leakage(manifest: pd.DataFrame, grouping_level: str = "patient") -> dict:
    """Raise LeakageError if any group/patient/study/image appears in more than one split.

    With ``grouping_level='study_fallback'`` patient overlap cannot be ruled out (IDs missing), so
    only group/study/image overlap is enforced and the report says so.
    """
    rep = overlap_report(manifest)
    keys = ["group_id", "study_uid", "image_id"] + (["patient_id"] if grouping_level == "patient" else [])
    bad = {k: v for k, v in rep.items() if k in keys and isinstance(v, dict) and any(n > 0 for n in v.values())}
    if bad or rep["duplicate_image_ids"]:
        raise LeakageError(f"Split leakage detected: {bad} duplicates={rep['duplicate_image_ids']}")
    return rep


def split_summary(manifest: pd.DataFrame) -> dict:
    out: dict = {"n_images": int(len(manifest)), "n_groups": int(manifest["group_id"].nunique()), "splits": {}}
    for s in SPLITS:
        m = manifest[manifest["split"] == s]
        pos = m[list(LABELS)] >= 0.5
        out["splits"][s] = {
            "n_images": int(len(m)),
            "n_groups": int(m["group_id"].nunique()),
            "n_patients": int(m.loc[m["patient_id"].astype(str) != "", "patient_id"].nunique()),
            "n_studies": int(m["study_uid"].nunique()),
            "positives": {c: int(pos[c].sum()) for c in LABELS},
            "prevalence": {c: float(pos[c].mean()) if len(m) else float("nan") for c in LABELS},
        }
    return out


def save_manifest(manifest: pd.DataFrame, info: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    csv = atomic_save_dataframe(manifest, out_dir / "split_manifest.csv")
    info = dict(info)
    info["manifest_hash"] = manifest_hash(manifest)
    info["overlap"] = overlap_report(manifest)
    js = atomic_write_json(out_dir / "split_manifest.json", info)
    return csv, js


def load_manifest(csv_path: str | Path, json_path: str | Path | None = None, verify: bool = True) -> tuple[pd.DataFrame, dict]:
    csv_path = Path(csv_path)
    json_path = Path(json_path) if json_path else csv_path.with_suffix(".json")
    manifest = pd.read_csv(csv_path, dtype={"image_id": str, "patient_id": str, "study_uid": str,
                                            "group_id": str, "series_uid": str, "series_key": str})
    for c in ("patient_id", "study_uid", "group_id"):
        if c in manifest.columns:
            manifest[c] = manifest[c].fillna("")
    info = read_json(json_path) if json_path.exists() else {}
    if verify:
        h = manifest_hash(manifest)
        if info and info.get("manifest_hash") != h:
            raise ManifestHashError(f"Manifest hash mismatch: file={h[:12]} json={str(info.get('manifest_hash'))[:12]}")
        verify_no_leakage(manifest, info.get("grouping_level", "patient"))
    info.setdefault("manifest_hash", manifest_hash(manifest))
    return manifest, info


def subset(manifest: pd.DataFrame, split: str, max_images: int | None = None, seed: int = 0,
           max_groups: int | None = None) -> pd.DataFrame:
    """Rows of one split, optionally capped by *groups* (patients) first, then images (smoke only)."""
    if split not in SPLITS:
        raise ValueError(split)
    m = manifest[manifest["split"] == split]
    rng = np.random.default_rng(seed)
    if max_groups is not None:
        groups = np.sort(m["group_id"].unique())
        if len(groups) > max_groups:
            keep = set(rng.choice(groups, size=max_groups, replace=False))
            m = m[m["group_id"].isin(keep)]
    if max_images is not None and len(m) > max_images:
        m = m.sample(n=max_images, random_state=seed)
    return m.reset_index(drop=True)
