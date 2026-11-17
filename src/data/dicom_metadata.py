"""Robust, restartable DICOM header scanning.

Only headers are read (``stop_before_pixels=True`` + ``specific_tags``), in parallel threads,
in chunks that are persisted immediately. If a Kaggle session dies, the next call resumes
from the chunks already on disk instead of rescanning hundreds of thousands of files.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pydicom
from pydicom.tag import Tag

from ..utils.io import atomic_save_dataframe

HEADER_TAGS = [
    "PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "InstanceNumber",
    "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns",
    "RescaleSlope", "RescaleIntercept", "WindowCenter", "WindowWidth", "BitsStored",
    "PixelRepresentation", "PhotometricInterpretation", "Modality",
]
_SPECIFIC = [Tag(pydicom.datadict.tag_for_keyword(k)) for k in HEADER_TAGS]

META_COLUMNS = [
    "image_id", "read_ok", "error", "patient_id", "study_uid", "series_uid", "sop_uid", "instance_number",
    "ipp_x", "ipp_y", "ipp_z", "iop", "slice_pos", "pixel_spacing_r", "pixel_spacing_c", "rows", "cols",
    "rescale_slope", "rescale_intercept", "window_center", "window_width", "bits_stored",
    "pixel_representation", "photometric", "modality",
]


def _first(v):
    """DICOM multi-values (e.g. WindowCenter) -> first element as float, or NaN."""
    if v is None:
        return np.nan
    try:
        if isinstance(v, (list, tuple, pydicom.multival.MultiValue)):
            v = v[0]
        return float(v)
    except Exception:  # noqa: BLE001
        return np.nan


def _str_or_empty(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return s


def slice_position(ipp: Sequence[float] | None, iop: Sequence[float] | None) -> float:
    """Signed position along the slice normal: dot(IPP, row_cos x col_cos).

    Falls back to IPP z if the orientation is missing, and to NaN if both are missing.
    Using the normal (not raw z) orders gantry-tilted/oblique series correctly.
    """
    if ipp is None or len(ipp) < 3:
        return float("nan")
    p = np.asarray([float(x) for x in ipp[:3]])
    if iop is not None and len(iop) >= 6:
        r = np.asarray([float(x) for x in iop[:3]])
        c = np.asarray([float(x) for x in iop[3:6]])
        n = np.cross(r, c)
        if np.linalg.norm(n) > 1e-6:
            return float(p @ (n / np.linalg.norm(n)))
    return float(p[2])


def read_header(path: str | os.PathLike, image_id: str | None = None) -> dict:
    """Read one DICOM header. Never raises: failures are returned with ``read_ok=False``."""
    image_id = image_id or Path(path).stem
    rec = {c: np.nan for c in META_COLUMNS}
    rec.update(image_id=image_id, read_ok=False, error="", patient_id="", study_uid="", series_uid="",
               sop_uid="", iop="", photometric="", modality="")
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=_SPECIFIC)
        if not any(k in ds for k in ("SOPInstanceUID", "Rows", "PatientID", "StudyInstanceUID")):
            # force=True "parses" arbitrary bytes; a file without any core tag is not a usable DICOM
            rec["error"] = "unreadable: no DICOM tags found"
            return rec
        ipp = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        ps = getattr(ds, "PixelSpacing", None)
        rec.update(
            patient_id=_str_or_empty(getattr(ds, "PatientID", None)),
            study_uid=_str_or_empty(getattr(ds, "StudyInstanceUID", None)),
            series_uid=_str_or_empty(getattr(ds, "SeriesInstanceUID", None)),
            sop_uid=_str_or_empty(getattr(ds, "SOPInstanceUID", None)),
            instance_number=_first(getattr(ds, "InstanceNumber", None)),
            iop=",".join(f"{float(x):.6f}" for x in iop) if iop is not None else "",
            slice_pos=slice_position(ipp, iop),
            rows=_first(getattr(ds, "Rows", None)), cols=_first(getattr(ds, "Columns", None)),
            rescale_slope=_first(getattr(ds, "RescaleSlope", None)),
            rescale_intercept=_first(getattr(ds, "RescaleIntercept", None)),
            window_center=_first(getattr(ds, "WindowCenter", None)),
            window_width=_first(getattr(ds, "WindowWidth", None)),
            bits_stored=_first(getattr(ds, "BitsStored", None)),
            pixel_representation=_first(getattr(ds, "PixelRepresentation", None)),
            photometric=_str_or_empty(getattr(ds, "PhotometricInterpretation", None)),
            modality=_str_or_empty(getattr(ds, "Modality", None)),
        )
        if ipp is not None and len(ipp) >= 3:
            rec.update(ipp_x=float(ipp[0]), ipp_y=float(ipp[1]), ipp_z=float(ipp[2]))
        if ps is not None and len(ps) >= 2:
            rec.update(pixel_spacing_r=float(ps[0]), pixel_spacing_c=float(ps[1]))
        rec["read_ok"] = True
    except FileNotFoundError:
        rec["error"] = "missing_file"
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return rec


def _parts_dir(cache_dir: Path) -> Path:
    d = Path(cache_dir) / "header_parts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _done_ids(parts_dir: Path) -> set[str]:
    done: set[str] = set()
    for p in sorted(parts_dir.glob("part_*.parquet")):
        try:
            done.update(pd.read_parquet(p, columns=["image_id"])["image_id"].tolist())
        except Exception:  # noqa: BLE001 - a corrupt part is ignored and its IDs rescanned
            continue
    return done


def scan_headers(image_ids: Iterable[str], dicom_dir: str | os.PathLike, cache_dir: str | os.PathLike,
                 chunk_size: int = 5000, workers: int = 8, verbose: bool = True) -> pd.DataFrame:
    """Scan headers for ``image_ids`` (resumable). Returns metadata for all requested IDs."""
    dicom_dir, cache_dir = Path(dicom_dir), Path(cache_dir)
    parts = _parts_dir(cache_dir)
    ids = list(dict.fromkeys(image_ids))
    done = _done_ids(parts)
    todo = [i for i in ids if i not in done]
    if verbose:
        print(f"[header scan] requested={len(ids):,} cached={len(ids) - len(todo):,} to_read={len(todo):,}")
    t0 = time.time()
    native = [int(q.stem.split("_")[1]) for q in parts.glob("part_[0-9]*.parquet")]
    next_part = max(native) + 1 if native else 0
    for s in range(0, len(todo), chunk_size):
        chunk = todo[s:s + chunk_size]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            recs = list(ex.map(lambda i: read_header(dicom_dir / f"{i}.dcm", i), chunk))
        df = pd.DataFrame(recs, columns=META_COLUMNS)
        atomic_save_dataframe(df, parts / f"part_{next_part:05d}.parquet")
        next_part += 1
        if verbose:
            n = min(s + chunk_size, len(todo))
            rate = n / max(time.time() - t0, 1e-6)
            eta = (len(todo) - n) / max(rate, 1e-6)
            print(f"  scanned {n:,}/{len(todo):,}  ({rate:,.0f} files/s, ETA {eta / 60:.1f} min)", flush=True)
    return load_metadata(cache_dir, ids)


def load_metadata(cache_dir: str | os.PathLike, image_ids: Iterable[str] | None = None) -> pd.DataFrame:
    parts = sorted(_parts_dir(Path(cache_dir)).glob("part_*.parquet"))
    if not parts:
        return pd.DataFrame(columns=META_COLUMNS)
    frames = []
    for p in parts:
        try:
            frames.append(pd.read_parquet(p))
        except Exception:  # noqa: BLE001
            continue
    meta = pd.concat(frames, ignore_index=True).drop_duplicates("image_id", keep="last")
    if image_ids is not None:
        meta = meta[meta["image_id"].isin(set(image_ids))]
    return meta.reset_index(drop=True)


def add_slice_order(meta: pd.DataFrame) -> pd.DataFrame:
    """Order slices within each series by slice position (InstanceNumber as tie-break/fallback)."""
    m = meta.copy()
    m["_pos"] = m["slice_pos"].where(m["slice_pos"].notna(), m["instance_number"])
    key = m["series_uid"].where(m["series_uid"] != "", m["study_uid"])
    m["_series_key"] = key
    m = m.sort_values(["_series_key", "_pos", "instance_number"], kind="stable")
    m["slice_rank"] = m.groupby("_series_key").cumcount()
    m["n_slices_series"] = m.groupby("_series_key")["image_id"].transform("size")
    return m.drop(columns=["_pos"]).rename(columns={"_series_key": "series_key"}).reset_index(drop=True)


def metadata_integrity_report(meta: pd.DataFrame, n_expected: int | None = None) -> dict:
    ok = meta[meta["read_ok"].astype(bool)]
    rep = {
        "n_requested": int(n_expected) if n_expected is not None else int(len(meta)),
        "n_scanned": int(len(meta)),
        "n_read_ok": int(len(ok)),
        "n_missing_file": int((meta["error"] == "missing_file").sum()),
        "n_unreadable": int(((~meta["read_ok"].astype(bool)) & (meta["error"] != "missing_file")).sum()),
        "unreadable_examples": meta.loc[~meta["read_ok"].astype(bool), ["image_id", "error"]].head(10).to_dict("records"),
        "n_missing_patient_id": int((ok["patient_id"] == "").sum()),
        "n_missing_study_uid": int((ok["study_uid"] == "").sum()),
        "n_missing_series_uid": int((ok["series_uid"] == "").sum()),
        "n_missing_position": int(ok["slice_pos"].isna().sum()),
        "n_patients": int(ok.loc[ok["patient_id"] != "", "patient_id"].nunique()),
        "n_studies": int(ok.loc[ok["study_uid"] != "", "study_uid"].nunique()),
        "n_series": int(ok.loc[ok["series_uid"] != "", "series_uid"].nunique()),
        "n_duplicate_image_ids": int(meta["image_id"].duplicated().sum()),
        "n_duplicate_sop_uids": int(ok.loc[ok["sop_uid"] != "", "sop_uid"].duplicated().sum()),
        "image_shapes": {f"{int(r)}x{int(c)}": int(n) for (r, c), n in
                         ok.groupby(["rows", "cols"]).size().sort_values(ascending=False).head(10).items()
                         if not (math.isnan(r) or math.isnan(c))},
        "photometric": ok["photometric"].value_counts().to_dict(),
        "modality": ok["modality"].value_counts().to_dict(),
    }
    has_study = ok[(ok["study_uid"] != "") & (ok["patient_id"] != "")]
    pats_per_study = has_study.groupby("study_uid")["patient_id"].nunique()
    rep["n_studies_with_multiple_patients"] = int((pats_per_study > 1).sum())
    studies_per_patient = has_study.groupby("patient_id")["study_uid"].nunique()
    rep["studies_per_patient"] = {str(k): int(v) for k, v in studies_per_patient.value_counts().sort_index().head(10).items()}
    rep["n_patients_with_multiple_studies"] = int((studies_per_patient > 1).sum())
    if len(ok):
        rep["slices_per_study"] = {q: float(v) for q, v in
                                   ok.groupby("study_uid").size().describe()[["min", "25%", "50%", "75%", "max"]].items()}
    return rep
