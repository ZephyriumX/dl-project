"""Create a tiny SYNTHETIC dataset with the RSNA folder layout, for local testing only.

    python tools/make_synthetic_rsna.py --out <dir> [--patients 24 --slices 10 --size 128]

It writes ``stage_2_train/``, ``stage_2_test/``, ``stage_2_train.csv`` and
``stage_2_sample_submission.csv``. The DICOMs contain a geometric head phantom (skull ring,
brain, ventricles) with bright blobs standing in for "hemorrhage". Some deliberate quirks are
added to exercise the integrity checks: one exact-duplicate CSV row, one conflicting duplicate,
one corrupt DICOM, one label row without a DICOM file, and a few 12-bit slices with the RSNA
intercept quirk.

These images are NOT medical data and must never be used to report results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

LABELS = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]


def phantom(size: int, rng: np.random.Generator, lesions: dict[str, bool], z_frac: float) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    cy, cx = size / 2, size / 2
    r = np.hypot((yy - cy) / (0.42 * size), (xx - cx) / (0.36 * size))
    hu = np.full((size, size), -1000.0)
    hu[r < 1.0] = 1000.0                     # skull
    hu[r < 0.9] = 32.0 + rng.normal(0, 4, (size, size))[r < 0.9]  # brain
    vent = np.hypot((yy - cy) / (0.12 * size), (xx - cx) / (0.06 * size)) < 1.0
    if 0.3 < z_frac < 0.7:
        hu[vent] = 5.0
    def blob(cy_, cx_, ry, rx, val=72.0):
        m = (np.hypot((yy - cy_) / ry, (xx - cx_) / rx) < 1.0) & (r < 0.9)
        hu[m] = val + rng.normal(0, 3, int(m.sum()))
    if lesions.get("epidural"):
        blob(cy - 0.30 * size, cx + 0.18 * size, 0.05 * size, 0.10 * size, 80)
    if lesions.get("intraparenchymal"):
        blob(cy + 0.10 * size, cx - 0.15 * size, 0.07 * size, 0.07 * size, 70)
    if lesions.get("intraventricular"):
        blob(cy, cx, 0.06 * size, 0.03 * size, 65)
    if lesions.get("subarachnoid"):
        blob(cy + 0.25 * size, cx + 0.05 * size, 0.02 * size, 0.12 * size, 60)
    if lesions.get("subdural"):
        blob(cy - 0.05 * size, cx - 0.31 * size, 0.18 * size, 0.025 * size, 75)
    return hu


def write_dicom(path: Path, hu: np.ndarray, patient: str, study: str, series: str, inst: int, z: float,
                quirk: bool = False):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    sop = generate_uid()
    meta.MediaStorageSOPInstanceUID = sop
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, sop
    ds.Modality = "CT"
    ds.PatientID, ds.StudyInstanceUID, ds.SeriesInstanceUID = patient, study, series
    ds.InstanceNumber = inst
    ds.ImagePositionPatient = [-125.0, -125.0, float(z)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [0.5, 0.5]
    ds.Rows, ds.Columns = hu.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.HighBit = 16, 15
    ds.WindowCenter, ds.WindowWidth = 40, 80
    ds.RescaleSlope = 1
    if quirk:  # RSNA 12-bit quirk: intercept ~0 although values wrap; the fix recovers HU exactly
        ds.BitsStored, ds.PixelRepresentation, ds.RescaleIntercept = 12, 0, 0
        # fix: px' = (stored + 1000) mod 4096 and HU = px' - 1000  =>  stored = HU mod 4096
        px = np.mod(np.round(np.clip(hu, -1000, 3000)).astype(np.int32), 4096).astype(np.uint16)
    else:
        ds.BitsStored, ds.PixelRepresentation, ds.RescaleIntercept = 16, 0, -1024
        px = np.clip(np.round(hu + 1024), 0, 65535).astype(np.uint16)
    ds.PixelData = px.tobytes()
    ds.save_as(path, enforce_file_format=True)


def make(out: Path, patients: int = 24, slices: int = 10, size: int = 128, test_images: int = 12, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    tr, te = out / "stage_2_train", out / "stage_2_test"
    tr.mkdir(parents=True, exist_ok=True)
    te.mkdir(parents=True, exist_ok=True)
    rows, n_img = [], 0
    for p in range(patients):
        pid = f"ID_pat{p:05d}"
        n_studies = 2 if p % 7 == 0 else 1
        sick = rng.random() < 0.5
        for s in range(n_studies):
            study, series = generate_uid(), generate_uid()
            subtypes = {l: (sick and rng.random() < 0.45) for l in LABELS[:5]}
            if sick and not any(subtypes.values()):
                subtypes["subdural"] = True
            lo, hi = sorted(rng.integers(0, slices, 2))
            for k in range(slices):
                iid = f"ID_{rng.integers(0, 16**9):09x}"
                z = 5.0 * k + rng.normal(0, 0.01)
                les = {l: v and lo <= k <= hi for l, v in subtypes.items()}
                hu = phantom(size, rng, les, k / max(slices - 1, 1))
                write_dicom(tr / f"{iid}.dcm", hu, pid, study, series, k + 1, z, quirk=(n_img % 37 == 5))
                lab = [int(les[l]) for l in LABELS[:5]]
                lab.append(int(max(lab)))
                rows += [(f"{iid}_{l}", v) for l, v in zip(LABELS, lab)]
                n_img += 1
    df = pd.DataFrame(rows, columns=["ID", "Label"])
    # quirks: exact duplicate row, conflicting duplicate, label row without dicom, corrupt dicom
    df = pd.concat([df, df.iloc[[0]], pd.DataFrame([(df.iloc[7]["ID"], 1 - df.iloc[7]["Label"])], columns=["ID", "Label"])])
    ghost = "ID_ffffffff0"
    df = pd.concat([df, pd.DataFrame([(f"{ghost}_{l}", 0) for l in LABELS], columns=["ID", "Label"])])
    corrupt_id = df.iloc[60]["ID"].rsplit("_", 1)[0]
    (tr / f"{corrupt_id}.dcm").write_bytes(b"this is not a dicom file")
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(out / "stage_2_train.csv", index=False)
    sub = []
    for i in range(test_images):
        iid = f"ID_ffff{i:05x}"  # hex like real RSNA IDs
        hu = phantom(size, rng, {"subdural": i % 3 == 0}, 0.5)
        write_dicom(te / f"{iid}.dcm", hu, f"ID_tpat{i:04d}", generate_uid(), generate_uid(), 1, 0.0)
        sub += [(f"{iid}_{l}", 0.5) for l in LABELS]
    pd.DataFrame(sub, columns=["ID", "Label"]).to_csv(out / "stage_2_sample_submission.csv", index=False)
    return {"n_train_images": n_img, "n_test_images": test_images, "corrupt_id": corrupt_id, "ghost_id": ghost}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients", type=int, default=24)
    ap.add_argument("--slices", type=int, default=10)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make(Path(a.out), a.patients, a.slices, a.size, seed=a.seed))
