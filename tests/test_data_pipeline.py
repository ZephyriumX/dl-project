"""Data pipeline: label parsing, DICOM metadata, splitting/leakage, preprocessing, datasets, config, paths."""

import numpy as np
import pandas as pd
import pytest
import torch

from src import LABELS
from src.data import (LeakageError, ManifestHashError, PreprocessConfig, RSNASliceDataset, load_manifest,
                      manifest_hash, metadata_integrity_report, parse_label_csv, preprocessing_fingerprint,
                      save_manifest, slice_position, subset, verify_no_leakage)
from src.data.csv_parser import LabelParsingError, binarize_targets
from src.data.preprocessing import apply_rsna_intercept_fix, hu_to_channels, pad_square, window
from src.utils.config import config_fingerprint, deep_merge, load_config


def _write_csv(path, rows):
    pd.DataFrame(rows, columns=["ID", "Label"]).to_csv(path, index=False)


def _rows(img, vals):
    return [(f"{img}_{l}", v) for l, v in zip(LABELS, vals)]


# ------------------------------------------------------------------------------------------ labels
def test_label_parsing_pivot_duplicates_and_report(tmp_path):
    rows = _rows("ID_aaa", [0, 0, 0, 0, 1, 1]) + _rows("ID_bbb", [0, 0, 0, 0, 0, 0])
    rows += [("ID_aaa_subdural", 1)]  # exact duplicate
    rows += [("ID_bbb_epidural", 1)]  # conflicting duplicate
    rows += [("garbage_row", 1), ("ID_ccc_notalabel", 0)]  # malformed
    rows += _rows("ID_ddd", [0, 0.5, 0, 0, 0, 0.5])  # fractional (soft) values
    rows += _rows("ID_eee", [0, 0, 0, 1, 0, 0])  # subtype without any
    _write_csv(tmp_path / "t.csv", rows)
    lab, rep = parse_label_csv(tmp_path / "t.csv")
    assert list(lab.columns) == ["image_id", *LABELS]
    assert rep.n_rows_malformed == 2 and rep.n_exact_duplicate_rows == 1
    assert rep.n_conflicting_duplicate_pairs == 1
    assert lab.set_index("image_id").loc["ID_bbb", "epidural"] == 1.0  # policy max
    assert rep.has_fractional_values and lab.set_index("image_id").loc["ID_ddd", "intraparenchymal"] == 0.5
    assert rep.n_any_inconsistent_subtype_without_any >= 1  # ID_eee (and ID_bbb after conflict resolution)
    assert rep.n_images == 4


def test_label_conflict_policies(tmp_path):
    rows = _rows("ID_aaa", [0, 0, 0, 0, 0, 0]) + [("ID_aaa_epidural", 1)]
    _write_csv(tmp_path / "t.csv", rows)
    with pytest.raises(LabelParsingError):
        parse_label_csv(tmp_path / "t.csv", conflict_policy="error")
    lab, _ = parse_label_csv(tmp_path / "t.csv", conflict_policy="drop_image")
    assert len(lab) == 0
    lab, _ = parse_label_csv(tmp_path / "t.csv", conflict_policy="min")
    assert lab["epidural"].iloc[0] == 0.0


def test_binarize_flags_soft_targets():
    yb, nb = binarize_targets(np.array([[0, 0.5, 1.0]]))
    assert nb and yb.tolist() == [[0, 1, 1]]


def test_synthetic_label_report(synthetic_manifest):
    _, _, labels, rep, _ = synthetic_manifest
    assert rep.n_exact_duplicate_rows == 1 and rep.n_conflicting_duplicate_pairs == 1
    assert labels["image_id"].is_unique


# ------------------------------------------------------------------------------------------ metadata
def test_header_scan_integrity(synthetic_rsna, synthetic_manifest):
    _, info = synthetic_rsna
    _, _, labels, _, meta = synthetic_manifest
    rep = metadata_integrity_report(meta, len(labels))
    assert rep["n_missing_file"] == 1  # ghost label row
    assert rep["n_unreadable"] == 1  # corrupt file
    assert set(meta.loc[~meta["read_ok"].astype(bool), "image_id"]) == {info["ghost_id"], info["corrupt_id"]}
    assert rep["n_patients"] == 8 and rep["n_studies_with_multiple_patients"] == 0


def test_header_scan_is_restartable(synthetic_rsna, tmp_path):
    from src.data import scan_headers

    root, _ = synthetic_rsna
    ids = [p.stem for p in sorted((root / "stage_2_train").glob("*.dcm"))][:10]
    m1 = scan_headers(ids[:6], root / "stage_2_train", tmp_path, chunk_size=3, workers=1, verbose=False)
    m2 = scan_headers(ids, root / "stage_2_train", tmp_path, chunk_size=3, workers=1, verbose=False)
    assert len(m1) == 6 and len(m2) == 10
    assert len(list((tmp_path / "header_parts").glob("part_*.parquet"))) == 4  # 2 parts + 2 new parts, none rescanned


def test_slice_position_uses_orientation():
    assert slice_position([0, 0, 5.0], [1, 0, 0, 0, 1, 0]) == pytest.approx(5.0)
    # tilted: normal of rows (1,0,0) x cols (0,cos,sin)
    c, s = np.cos(0.2), np.sin(0.2)
    z = slice_position([0, 0, 10.0], [1, 0, 0, 0, c, s])
    assert z == pytest.approx(10.0 * c)
    assert np.isnan(slice_position(None, None))


# ------------------------------------------------------------------------------------------ splitting
def test_patient_split_no_overlap(synthetic_manifest):
    manifest, info, *_ = synthetic_manifest
    assert info["grouping_level"] == "patient"
    rep = verify_no_leakage(manifest)
    assert all(v == 0 for v in rep["patient_id"].values())
    for s in ("train", "val", "test"):
        assert (manifest["split"] == s).sum() > 0
    # every image of a patient is in one split
    assert (manifest.groupby("patient_id")["split"].nunique() == 1).all()


def test_leakage_detection_raises(synthetic_manifest):
    manifest, *_ = synthetic_manifest
    bad = manifest.copy()
    pid = bad.loc[bad["split"] == "train", "patient_id"].iloc[0]
    idx = bad.index[(bad["patient_id"] == pid)][0]
    bad.loc[idx, "split"] = "test"
    with pytest.raises(LeakageError):
        verify_no_leakage(bad)


def test_study_fallback_when_patient_ids_missing(synthetic_manifest):
    from src.data import make_patient_split

    manifest, *_ = synthetic_manifest
    df = manifest.drop(columns=["split", "group_id", "stratum"]).copy()
    df["patient_id"] = ""
    m2, info = make_patient_split(df, seed=0)
    assert info["grouping_level"] == "study_fallback"
    assert (m2.groupby("study_uid")["split"].nunique() == 1).all()


def test_manifest_hash_save_load_and_tamper(synthetic_manifest, tmp_path):
    manifest, info, *_ = synthetic_manifest
    save_manifest(manifest, info, tmp_path)
    m, i = load_manifest(tmp_path / "split_manifest.csv")
    assert i["manifest_hash"] == manifest_hash(manifest) == manifest_hash(m)
    t = m.copy()
    t.loc[0, "split"] = "val" if t.loc[0, "split"] != "val" else "train"
    t.to_csv(tmp_path / "split_manifest.csv", index=False)
    with pytest.raises((ManifestHashError, LeakageError)):
        load_manifest(tmp_path / "split_manifest.csv")


def test_split_reproducible(synthetic_manifest):
    from src.data import make_patient_split

    manifest, info, *_ = synthetic_manifest
    df = manifest.drop(columns=["split", "group_id", "stratum"])
    _, i1 = make_patient_split(df, seed=0)
    _, i2 = make_patient_split(df, seed=0)
    _, i3 = make_patient_split(df, seed=1)
    assert i1["manifest_hash"] == i2["manifest_hash"] == info["manifest_hash"]
    assert i3["manifest_hash"] != i1["manifest_hash"]


def test_subset_caps_groups_first(synthetic_manifest):
    manifest, *_ = synthetic_manifest
    s = subset(manifest, "train", max_groups=2, seed=0)
    assert s["group_id"].nunique() <= 2


# ------------------------------------------------------------------------------------------ preprocessing
def test_window_and_pad():
    hu = np.array([[-100.0, 40.0, 200.0]])
    w = window(hu, 40, 80)
    assert w.tolist() == [[0.0, 0.5, 1.0]]
    sq = pad_square(np.zeros((2, 4)), -1024.0)
    assert sq.shape == (4, 4) and sq[0, 0] == -1024.0


def test_rsna_intercept_fix_recovers_hu():
    import pydicom

    ds = pydicom.Dataset()
    ds.BitsStored, ds.PixelRepresentation, ds.RescaleIntercept = 12, 0, 0
    hu_true = np.array([-1000.0, -20.0, 0.0, 40.0, 1000.0])
    stored = np.mod(hu_true.astype(int), 4096).astype(np.float32)
    px, inter, applied = apply_rsna_intercept_fix(stored.copy(), ds)
    assert applied and np.allclose(px + inter, hu_true)


def test_dicom_to_channels_and_fingerprint(synthetic_manifest, synthetic_rsna):
    from src.data import load_dicom_channels

    root, _ = synthetic_rsna
    manifest, *_ = synthetic_manifest
    cfg = PreprocessConfig(image_size=96)
    x = load_dicom_channels(str(root / "stage_2_train" / f"{manifest.image_id[0]}.dcm"), cfg)
    assert x.shape == (3, 96, 96) and x.dtype == np.float32 and 0 <= x.min() <= x.max() <= 1
    assert preprocessing_fingerprint(cfg) != preprocessing_fingerprint(PreprocessConfig(image_size=96, channel_mode="brain3"))
    assert hu_to_channels(np.zeros((50, 70), np.float32), cfg).shape == (3, 96, 96)


def test_dataset_items_and_decode_failure(synthetic_manifest, synthetic_rsna, tmp_path):
    root, info = synthetic_rsna
    manifest, *_ = synthetic_manifest
    frame = manifest.head(4).copy()
    ds = RSNASliceDataset(frame, root / "stage_2_train", {"image_size": 64}, disk_cache_dir=str(tmp_path),
                          disk_cache_max_gb=0.01, memory_cache_items=2)
    it = ds[0]
    assert it["image"].shape == (3, 64, 64) and it["target"].shape == (6,) and it["ok"]
    it2 = ds[0]  # cached path gives the identical tensor
    assert torch.equal(it["image"], it2["image"])
    assert len(list(tmp_path.rglob("*.npy"))) >= 1
    bad = pd.DataFrame({"image_id": [info["corrupt_id"]], **{l: [0.0] for l in LABELS}})
    b = RSNASliceDataset(bad, root / "stage_2_train", {"image_size": 64})[0]
    assert not b["ok"] and b["image"].shape == (3, 64, 64)
    with pytest.raises(Exception):
        RSNASliceDataset(bad, root / "stage_2_train", {"image_size": 64}, on_error="raise")[0]


def test_unlabelled_dataset_for_competition_test(synthetic_rsna):
    from src.data import parse_submission_ids

    root, _ = synthetic_rsna
    ids = parse_submission_ids(root / "stage_2_sample_submission.csv")
    ds = RSNASliceDataset(ids, root / "stage_2_test", {"image_size": 64})
    assert len(ds) == 4 and torch.isnan(ds[0]["target"]).all()


def test_augmentation_deterministic_per_epoch(synthetic_manifest, synthetic_rsna):
    from src.data import build_augment

    root, _ = synthetic_rsna
    manifest, *_ = synthetic_manifest
    ds = RSNASliceDataset(manifest.head(2), root / "stage_2_train", {"image_size": 64}, augment=build_augment({"enabled": True}))
    a, b = ds[0]["image"], ds[0]["image"]
    assert torch.equal(a, b)  # same (seed, epoch, index) -> identical augmentation
    changed = []
    for e in range(1, 6):
        ds.set_epoch(e)
        changed.append(not torch.equal(a, ds[0]["image"]))
    assert any(changed)  # different epochs draw different augmentations


# ------------------------------------------------------------------------------------------ config / paths
def test_config_modes_and_overrides():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    smoke = load_config(root / "data_config.yaml", "smoke")
    full = load_config(root / "data_config.yaml", "full", {"split": {"seed_extra": 1}})
    assert smoke["header_scan"]["max_images"] == 4000 and full["header_scan"]["max_images"] is None
    assert full["split"]["fractions"]["train"] == 0.7 and full["split"]["seed_extra"] == 1
    for f in ("transformer_config.yaml", "hybrid_config.yaml", "quantum_config.yaml"):
        for m in ("smoke", "pilot", "full"):
            assert isinstance(load_config(root / f, m), dict)
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    assert config_fingerprint({"a": 1, "_x": 2}) == config_fingerprint({"a": 1})
    with pytest.raises(ValueError):
        load_config(root / "data_config.yaml", "huge")


def test_dataset_discovery(synthetic_rsna, monkeypatch, tmp_path):
    from src.utils.paths import DatasetNotFoundError, resolve_rsna_paths

    root, _ = synthetic_rsna
    monkeypatch.setenv("RSNA_DATA_ROOT", str(root))
    p = resolve_rsna_paths(candidates=[])
    assert p.complete and p.train_dir.is_dir()
    monkeypatch.delenv("RSNA_DATA_ROOT")
    with pytest.raises(DatasetNotFoundError):
        resolve_rsna_paths(str(tmp_path / "nope"), candidates=[str(tmp_path)])


def test_shared_module_imports():
    import importlib

    for m in ["src", "src.data", "src.models", "src.quantum", "src.training", "src.evaluation", "src.utils.context",
              "src.training.experiment", "src.evaluation.explainability", "src.evaluation.visualization"]:
        importlib.import_module(m)


# ------------------------------------------------------------------------------------------ context / registry
def test_project_context_modes_restore_and_registry(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from src.utils.context import ProjectContext
    from src.utils.registry import ExperimentRegistry

    root = Path(__file__).resolve().parents[1]
    prev = tmp_path / "previous_output"  # stands in for an attached earlier notebook version
    exp = prev / "smoke" / "experiments" / "A_fake"
    exp.mkdir(parents=True)
    (exp / "last.pt").write_bytes(b"x")
    (prev / "smoke" / "data").mkdir(parents=True)
    (prev / "smoke" / "data" / "split_manifest.csv").write_text("image_id,split\n", encoding="utf-8")
    ctx = ProjectContext(root, "smoke", {"data": {"paths": {"artifact_input_roots": [str(prev)]}}},
                         output_root=str(tmp_path / "work"))
    assert ctx.output_root == tmp_path / "work" / "smoke"
    assert ctx.find("data/split_manifest.csv") is not None
    assert ctx.restore_experiments(("A_",)) == ["A_fake"]
    assert (ctx.output_root / "experiments" / "A_fake" / "last.pt").exists()
    assert ctx.restore_experiments(("A_",)) == []  # never overwrites
    full = ProjectContext(root, "full", output_root=str(tmp_path / "work"))
    assert full.find("data/split_manifest.csv", required=False) is None  # modes are isolated
    with pytest.raises(FileNotFoundError):
        full.find("data/split_manifest.csv")

    reg = ExperimentRegistry(tmp_path / "r.jsonl")
    reg.record("e1", "running")
    reg.record("e1", "succeeded", val=float("nan"))
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        reg.record_failure("e2", e)
    with open(tmp_path / "r.jsonl", "a", encoding="utf-8") as f:
        f.write('{"experiment_id": "torn"')  # interrupted write
    assert reg.status_of("e1") == "succeeded" and reg.status_of("e2") == "failed" and reg.status_of("x") is None
    for line in (tmp_path / "r.jsonl").read_text(encoding="utf-8").splitlines()[:3]:
        json.loads(line)  # standard JSON (NaN converted to null)
