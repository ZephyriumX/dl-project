import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture(scope="session")
def synthetic_rsna(tmp_path_factory):
    """Small SYNTHETIC dataset in RSNA layout (generated, not medical data)."""
    from make_synthetic_rsna import make

    out = tmp_path_factory.mktemp("syn_rsna")
    info = make(out, patients=8, slices=5, size=64, test_images=4, seed=3)
    return out, info


@pytest.fixture(scope="session")
def synthetic_manifest(synthetic_rsna, tmp_path_factory):
    from src.data import add_slice_order, make_patient_split, parse_label_csv, scan_headers

    root, _ = synthetic_rsna
    labels, rep = parse_label_csv(root / "stage_2_train.csv")
    meta = scan_headers(labels["image_id"], root / "stage_2_train", tmp_path_factory.mktemp("hdr"), chunk_size=16,
                        workers=2, verbose=False)
    ok = add_slice_order(meta[meta["read_ok"].astype(bool)])
    manifest, info = make_patient_split(labels.merge(ok, on="image_id"), seed=0)
    return manifest, info, labels, rep, meta
