"""Generate the six Kaggle notebooks (no outputs) from one source of truth.

    python tools/build_notebooks.py

Every notebook starts with the same setup cell (find project root, import ``src``) and the same
configuration cell (MODE, overrides, ProjectContext). All logic lives in ``src/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb_common import DEPENDENCY_CELL, SETUP_CELL, context_cell  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def header(title, purpose, inputs, outputs):
    return md(f"""
# {title}

**Purpose.** {purpose}

**Requires (inputs).** {inputs}

**Produces (outputs, under `/kaggle/working/rsna_ich_artifacts/<MODE>/`).** {outputs}

> Research software. Outputs are model predictions for a research study and **not** clinical diagnoses.
> Results in this notebook are only valid for the MODE they were produced in (`smoke` results check correctness and say nothing about performance).
""")


COMMON_IMPORTS = r'''
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from IPython.display import display
from src import LABELS
from src.utils.io import atomic_write_json, atomic_save_dataframe, read_json, dataframe_to_markdown
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
'''


# =====================================================================================================
# 00 Data audit
# =====================================================================================================
def nb00():
    c = [header("00 - Data Audit and Preprocessing",
                "Discover and validate the RSNA data, parse the six labels, scan DICOM headers (restartable), check "
                "integrity, build the single patient-level split manifest used by every model, and validate preprocessing visually.",
                "RSNA competition data attached (`stage_2_train/`, `stage_2_test/`, `stage_2_train.csv`, `stage_2_sample_submission.csv`); project code dataset.",
                "`data/labels.parquet`, `data/label_report.json`, `data/metadata.parquet`, `data/metadata_integrity.json`, "
                "`data/split_manifest.csv|json`, `data/data_audit_report.json`, figures in `data/figures/`. Header cache in `shared/header_cache/`."),
         code(SETUP_CELL), code(DEPENDENCY_CELL), code(context_cell("00_data_audit")), code(COMMON_IMPORTS),
         md("## 1. Dataset discovery and component verification"),
         code(r'''
paths = ctx.rsna  # raises a clear error if the competition data is not attached
print(json.dumps(paths.as_dict(), indent=2))
if paths.missing:
    print("WARNING - missing components:", paths.missing)
dcfg = ctx.cfg["data"]
data_dir = ctx.dir("data"); fig_dir = ctx.dir("data", "figures")
print("Raw CSV schema (first rows):")
display(pd.read_csv(paths.train_csv, nrows=12))
if paths.sample_submission.exists():
    display(pd.read_csv(paths.sample_submission, nrows=7))
'''),
         md("## 2. Labels: parse, validate and pivot to one six-label vector per image\n"
            "The CSV is long format (`ID_<image>_<label>`, one row per label). Duplicates, malformed IDs, fractional "
            "values and `any` consistency are reported, not silently fixed."),
         code(r'''
from src.data.csv_parser import parse_label_csv, label_distribution_table
lab_path, rep_path = data_dir / "labels.parquet", data_dir / "label_report.json"
policy = dcfg["labels"]["conflict_policy"]
if lab_path.exists() and rep_path.exists() and read_json(rep_path).get("conflict_policy") == policy:
    labels, label_report = pd.read_parquet(lab_path), read_json(rep_path)
    print("Loaded cached labels.")
else:
    labels, rep = parse_label_csv(paths.train_csv, conflict_policy=policy)
    label_report = rep.as_dict()
    atomic_save_dataframe(labels, lab_path); atomic_write_json(rep_path, label_report)
for k in ["n_rows_raw", "n_rows_malformed", "n_exact_duplicate_rows", "n_conflicting_duplicate_pairs", "n_images",
          "n_images_missing_labels", "has_fractional_values", "unique_values",
          "n_any_inconsistent_subtype_without_any", "n_any_inconsistent_any_without_subtype"]:
    print(f"{k:45s} {label_report[k]}")
if label_report["conflicting_examples"]:
    print("Conflicting duplicates (resolved with policy =", policy, "):", label_report["conflicting_examples"][:5])
display(label_distribution_table(labels))
print("Number of subtypes per ICH-positive image:", label_report["n_labels_per_positive_image"])
'''),
         md("## 3. Files on disk vs. labelled IDs"),
         code(r'''
listing_path = data_dir / "train_dir_listing.parquet"
if listing_path.exists():
    files = pd.read_parquet(listing_path)
else:
    files = pd.DataFrame({"image_id": [e.name[:-4] for e in os.scandir(paths.train_dir) if e.name.endswith(".dcm")]})
    atomic_save_dataframe(files, listing_path)
ids_csv, ids_dir = set(labels["image_id"]), set(files["image_id"])
labelled_without_dicom = sorted(ids_csv - ids_dir)
dicom_without_label = sorted(ids_dir - ids_csv)
print(f"DICOM files in stage_2_train: {len(ids_dir):,} | labelled images: {len(ids_csv):,}")
print(f"labelled but no DICOM file: {len(labelled_without_dicom):,} {labelled_without_dicom[:5]}")
print(f"DICOM file but no label:    {len(dicom_without_label):,} {dicom_without_label[:5]}")
'''),
         md("## 4. DICOM header scan (restartable, cached)\n"
            "Only headers are read. Chunks are saved as they complete, so an interrupted session resumes. In `smoke` mode "
            "a random subset is scanned: this is leakage-safe (unscanned images are simply never used), but studies are incomplete."),
         code(r'''
from src.data.dicom_metadata import scan_headers, metadata_integrity_report, add_slice_order
hs = dcfg["header_scan"]
ctx.restore_shared_cache()  # reuse a previous version's scan if its output is attached
scan_ids = labels.loc[labels["image_id"].isin(ids_dir), "image_id"]
if hs["max_images"]:
    scan_ids = scan_ids.sample(n=min(int(hs["max_images"]), len(scan_ids)), random_state=dcfg["seed"])
meta = scan_headers(scan_ids, paths.train_dir, ctx.shared_dir("header_cache"), hs["chunk_size"], hs["workers"])
integrity = metadata_integrity_report(meta, len(scan_ids))
integrity["labelled_without_dicom"] = len(labelled_without_dicom)
integrity["dicom_without_label"] = len(dicom_without_label)
n_available = int(labels["image_id"].isin(ids_dir).sum())
integrity["partial_scan"] = bool(hs["max_images"]) and int(hs["max_images"]) < n_available
atomic_write_json(data_dir / "metadata_integrity.json", integrity)
meta_ok = add_slice_order(meta[meta["read_ok"].astype(bool)])
atomic_save_dataframe(meta_ok, data_dir / "metadata.parquet")
print(json.dumps({k: v for k, v in integrity.items() if k != "unreadable_examples"}, indent=1, default=str))
if integrity["partial_scan"]:
    print("NOTE: partial header scan (smoke). Studies are incomplete; study-level results are not meaningful in this mode.")
'''),
         md("## 5. Patient-level split manifest (created once, reused by every model)\n"
            "All images of a patient go to one split. Stratification is at patient level (epidural / other ICH / negative). "
            "If an existing manifest was built with a different configuration the notebook stops instead of overwriting it."),
         code(r'''
from src.data.splitting import make_patient_split, save_manifest, load_manifest, overlap_report
from src.utils.config import config_fingerprint
FORCE_NEW_MANIFEST = False
sp = dcfg["split"]
split_fp = config_fingerprint({"split": sp, "seed": dcfg["seed"], "header_scan": hs, "labels": dcfg["labels"]})
man_csv = data_dir / "split_manifest.csv"
if man_csv.exists() and not FORCE_NEW_MANIFEST:
    manifest, man_info = load_manifest(man_csv)
    if man_info.get("split_config_fingerprint") != split_fp:
        raise RuntimeError("An existing split manifest was built with a different configuration. Every trained model "
                           "depends on it. Set FORCE_NEW_MANIFEST=True only if you intend to invalidate those results.")
    print("Reusing existing manifest", man_info["manifest_hash"][:12])
else:
    if man_csv.exists():
        old = read_json(man_csv.with_suffix(".json"))
        arch = ctx.dir("data", "archive", old["manifest_hash"][:12])
        man_csv.replace(arch / man_csv.name); man_csv.with_suffix(".json").replace(arch / "split_manifest.json")
        print("Archived previous manifest to", arch)
    df = labels.merge(meta_ok, on="image_id", how="inner")
    manifest, man_info = make_patient_split(df, sp["fractions"], dcfg["seed"], sp["max_groups"],
                                            sp["max_missing_patient_fraction"], sp["min_positive_group_fraction"])
    man_info.update(split_config_fingerprint=split_fp, mode=MODE, partial_header_scan=integrity["partial_scan"],
                    label_report_conflict_policy=policy)
    save_manifest(manifest, man_info, data_dir)
    manifest, man_info = load_manifest(man_csv)
print("Grouping level:", man_info["grouping_level"], "| manifest hash:", man_info["manifest_hash"][:16])
if man_info["grouping_level"] != "patient":
    print("WARNING: PatientID missing for too many images -> STUDY-level fallback split. Do not call it patient-level.")
summary = pd.DataFrame({s: {"images": v["n_images"], "patients/groups": v["n_groups"], "studies": v["n_studies"],
                            **{f"pos_{l}": v["positives"][l] for l in LABELS}} for s, v in man_info["splits"].items()}).T
display(summary)
print("Overlap report (must be all zeros):", overlap_report(manifest))
'''),
         md("## 6. Label distribution by partition"),
         code(r'''
from src.evaluation.visualization import plot_label_distribution
dist = label_distribution_table(manifest, by="split")
display(dist.pivot(index="label", columns="split", values="prevalence").loc[list(LABELS)])
plot_label_distribution(dist, fig_dir / "label_distribution_by_split.png"); plt.show()
'''),
         md("## 7. Preprocessing validation (visual)\n"
            "HU conversion (with the RSNA 12-bit intercept fix), brain / subdural / soft-tissue windows, resize. "
            "The same function is used for training and inference (fingerprinted)."),
         code(r'''
from src.data.preprocessing import PreprocessConfig, load_dicom_channels, preprocessing_fingerprint, CHANNEL_MODES
from src.evaluation.visualization import plot_windows
pp = PreprocessConfig.from_dict(dcfg["preprocessing"])
print("Preprocessing:", pp, "| fingerprint", preprocessing_fingerprint(pp))
rng = np.random.default_rng(0)
pos = manifest[manifest["any"] >= 0.5]["image_id"].tolist(); neg = manifest[manifest["any"] < 0.5]["image_id"].tolist()
show = list(rng.choice(pos, min(3, len(pos)), replace=False)) + list(rng.choice(neg, min(3, len(neg)), replace=False))
for iid in show[: dcfg["visual_check_images"]]:
    ch = load_dicom_channels(str(paths.train_dir / f"{iid}.dcm"), pp)
    row = manifest.set_index("image_id").loc[iid]
    title = f"{iid}: " + (", ".join(l for l in LABELS[:-1] if row[l] >= 0.5) or "no ICH label")
    plot_windows(ch, pp.window_names, title, fig_dir / f"windows_{iid}.png"); plt.show()
single = PreprocessConfig.from_dict({**dcfg["preprocessing"], "channel_mode": "brain3"})
print("Single-window ablation input uses channels:", single.window_names)
'''),
         md("## 8. Competition test folder (inference only)\n`stage_2_test` has no labels. It is never part of the research manifest."),
         code(r'''
from src.data.csv_parser import parse_submission_ids
if paths.sample_submission.exists() and paths.test_dir.exists():
    test_ids = parse_submission_ids(paths.sample_submission)
    test_files = {e.name[:-4] for e in os.scandir(paths.test_dir) if e.name.endswith(".dcm")}
    print(f"competition test images in sample submission: {len(test_ids):,}; DICOMs on disk: {len(test_files):,}; "
          f"missing files: {len(set(test_ids.image_id) - test_files):,}")
    assert not (set(test_ids.image_id) & set(manifest.image_id)), "competition test IDs must not appear in the manifest"
'''),
         md("## 9. Save the data-audit report"),
         code(r'''
report = {"mode": MODE, "paths": paths.as_dict(), "label_report": label_report, "integrity": integrity,
          "manifest": {k: v for k, v in man_info.items() if k != "overlap"}, "overlap": overlap_report(manifest),
          "preprocessing": {"config": dcfg["preprocessing"], "fingerprint": preprocessing_fingerprint(pp)}}
atomic_write_json(data_dir / "data_audit_report.json", report)
print("Saved:", sorted(p.name for p in data_dir.iterdir()))
'''),
         md("**Next:** Save this notebook version (Kaggle: *Save Version → Save & Run All*) so `/kaggle/working` becomes a "
            "persistent output, then attach it as input to notebooks 01-05.")]
    return c


# =====================================================================================================
# 01 Transformers
# =====================================================================================================
def nb01():
    c = [header("01 - Classical Transformer Benchmark (Track A)",
                "Train and evaluate DeiT-Small, Swin-Tiny and MaxViT-Tiny (plus optional extras) with one data pipeline, one split "
                "manifest, one loss and one evaluation protocol. Model selection uses the validation split only.",
                "Output of notebook 00 (split manifest) for the same MODE; RSNA data; optional offline timm weights dataset.",
                "`experiments/A_<model>/` (best/last checkpoints, history, val/test predictions, thresholds, metrics, CIs), "
                "`selection/best_transformer.json`, optional ablation H `experiments/H_<model>_brain3/`."),
         code(SETUP_CELL), code(DEPENDENCY_CELL), code(context_cell("01_transformers")), code(COMMON_IMPORTS),
         md("## 1. Load the shared split manifest and configuration"),
         code(r'''
manifest, man_info = ctx.manifest()
ctx.restore_experiments(("A_", "H_"))  # resume interrupted runs from an attached previous version
tcfg = ctx.cfg["transformer"]
print("manifest", man_info["manifest_hash"][:12], "| grouping:", man_info["grouping_level"], "| images:", len(manifest))
print("Runtime:", ctx.runtime.summary())
models = list(tcfg["models"]) + [m for m in tcfg.get("optional_models", []) if m["name"] in tcfg.get("run_optional", [])]
print("Planned models:", [m["name"] for m in models])
'''),
         md("## 2. Train / evaluate each model (restartable)\n"
            "A failure (missing weights, incompatible timm, OOM) is recorded and the loop continues. Completed experiments with "
            "an identical configuration are loaded, not retrained. Weight status is reported for every model."),
         code(r'''
from src.training.experiment import run_image_experiment
results = {}
for m in models:
    m = {**m, "weights_dir": tcfg.get("weights_dir")}
    exp_id = f"A_{m['name']}"
    print("=" * 100, "\n", exp_id)
    results[exp_id] = run_image_experiment(
        ctx, exp_id, m, tcfg["train"], tcfg["loss"], manifest=manifest, manifest_info=man_info,
        preprocess=ctx.cfg["data"]["preprocessing"], augment=ctx.cfg["data"]["augmentation"], limits=tcfg["limits"],
        loader_cfg=tcfg["loader"], threshold_method=tcfg["threshold_method"], n_boot=tcfg["n_boot"], seed=tcfg["seed"])
'''),
         md("## 3. Benchmark table (sorted by **validation** macro ROC-AUC; test columns are reported, never used for selection)"),
         code(r'''
def row(exp_id, r):
    out = {"experiment": exp_id, "status": r.get("status"), "weights": (r.get("build") or {}).get("status"),
           "params_M": round((r.get("build") or {}).get("n_params", 0) / 1e6, 1)}
    if r.get("status") == "succeeded":
        ci = r.get("test_macro_auc_ci", {})
        out.update(val_macro_auc=r["val_metrics"]["macro_roc_auc"], test_macro_auc=r["test_metrics"]["macro_roc_auc"],
                   test_ci=f"[{ci.get('ci_low', np.nan):.3f}, {ci.get('ci_high', np.nan):.3f}]",
                   test_macro_ap=r["test_metrics"]["macro_ap"], test_any_auc=r["test_metrics"].get("any_roc_auc"),
                   best_epoch=r["fit"]["best_epoch"], epochs_run=r["fit"]["epochs_run"], train_min=round(r["seconds_total"] / 60, 1))
    else:
        out["reason"] = r.get("error") or r.get("reason")
    return out
bench = pd.DataFrame([row(k, v) for k, v in results.items()])
if "val_macro_auc" in bench:
    bench = bench.sort_values("val_macro_auc", ascending=False, na_position="last")
display(bench)
atomic_save_dataframe(bench, ctx.dir("analysis") / "trackA_benchmark.csv")
'''),
         md("## 4. Per-label results, training curves and ROC/PR curves (held-out test, frozen validation thresholds)"),
         code(r'''
from src.evaluation.visualization import plot_history, plot_roc_pr
from src.evaluation.predictions import load_predictions, arrays
for exp_id, r in results.items():
    if r.get("status") != "succeeded":
        continue
    d = ctx.dir("experiments", exp_id)
    display(pd.read_csv(d / "metrics_test_per_label.csv")[["label", "n_pos", "roc_auc", "average_precision", "sensitivity",
                                                           "specificity", "precision", "f1", "brier", "threshold", "undefined"]].round(4))
    plot_history(pd.read_csv(d / "history.csv"), d / "history.png", exp_id); plt.show()
    pdf, _ = load_predictions(d / "predictions_test.parquet", man_info["manifest_hash"])
    p, t = arrays(pdf)
    plot_roc_pr(t, p, d / "roc_pr_test.png", f"{exp_id} - held-out test"); plt.show()
'''),
         md("## 5. Select the best Transformer by validation macro ROC-AUC"),
         code(r'''
from src.training.experiment import select_best_by_validation
best = select_best_by_validation(ctx, prefixes=("A_",))
if best is None:
    print("No Track-A experiment succeeded - nothing to select.")
else:
    sel = {"experiment_id": best["experiment_id"], "selected_by": "validation macro ROC-AUC",
           "val_macro_auc": best["val_metrics"]["macro_roc_auc"], "manifest_hash": man_info["manifest_hash"],
           "weights_status": best["build"]["status"]}
    atomic_write_json(ctx.dir("selection") / "best_transformer.json", sel)
    print(sel)
'''),
         md("## 6. Ablation H - single-window vs multi-window input (best Transformer, identical budget)"),
         code(r'''
if tcfg.get("ablation_single_window") and best is not None:
    mcfg = dict(best["config"]["model"]); mcfg.pop("in_chans", None)
    pp_single = {**ctx.cfg["data"]["preprocessing"], "channel_mode": "brain3"}
    exp_id = f"H_{mcfg['name']}_brain3"
    rH = run_image_experiment(ctx, exp_id, mcfg, tcfg["train"], tcfg["loss"], manifest=manifest, manifest_info=man_info,
                              preprocess=pp_single, augment=ctx.cfg["data"]["augmentation"], limits=tcfg["limits"],
                              loader_cfg=tcfg["loader"], threshold_method=tcfg["threshold_method"], n_boot=tcfg["n_boot"],
                              seed=tcfg["seed"])
    display(pd.DataFrame([row(best["experiment_id"] + " (bsb)", best), row(exp_id, rH)]))
else:
    ctx.registry.record("H_single_window", "skipped", reason="disabled in this MODE or no Track-A model succeeded")
    print("Ablation H skipped in this mode (recorded in the registry).")
'''),
         md("**Interpretation guidance.** Compare models on validation for selection; the test column with patient-level CIs "
            "is for reporting. Overlapping CIs mean the data cannot distinguish the models. Paired tests are in notebook 04.")]
    return c


# =====================================================================================================
# 02 CNN-Transformer hybrid
# =====================================================================================================
def nb02():
    c = [header("02 - CNN-Transformer Hybrid (Track B)",
                "Train the CNN-Transformer hybrid (EfficientNet-B0 stem + Transformer encoder over stride-16 tokens, fused "
                "with the CNN global vector) and its standalone CNN reference on the same manifest; compare against Track A.",
                "Notebook 00 output (manifest); optionally notebook 01 output for the Transformer comparison.",
                "`experiments/B_cnn_transformer_hybrid/`, `experiments/B_cnn_reference_<cnn>/`, optional fusion ablations "
                "`experiments/B_hybrid_fusion_<mode>/`, `analysis/trackB_comparison.csv`."),
         code(SETUP_CELL), code(DEPENDENCY_CELL), code(context_cell("02_cnn_transformer")), code(COMMON_IMPORTS),
         md("## 1. Manifest and configuration"),
         code(r'''
manifest, man_info = ctx.manifest()
ctx.restore_experiments(("B_",))
hcfg = ctx.cfg["hybrid"]
from src.models.cnn_transformer import CNNTransformerHybrid
m = CNNTransformerHybrid(**{k: v for k, v in hcfg["hybrid"].items() if k not in ("kind", "name")}, )
print(m.record.as_dict()); del m
'''),
         md("## 2. Architecture motivation\n"
            "* **CNN stem (local):** acute blood is a small, local hyperdensity; convolutions have the right inductive bias.\n"
            "* **Transformer over CNN tokens (global):** comparison across hemispheres, ventricles vs parenchyma, thin rims vs "
            "skull require long-range context; attention over a 14x14 token grid provides it cheaply.\n"
            "* **Fusion:** concatenating the CNN global-average vector with the [CLS] token keeps both kinds of evidence; the "
            "`gap_only`/`cls_only` ablations test whether the Transformer branch contributes."),
         md("## 3. Train the hybrid, the standalone CNN reference and (optional) fusion ablations"),
         code(r'''
from src.training.experiment import run_image_experiment
common = dict(manifest=manifest, manifest_info=man_info, preprocess=ctx.cfg["data"]["preprocessing"],
              augment=ctx.cfg["data"]["augmentation"], limits=hcfg["limits"], loader_cfg=hcfg["loader"],
              threshold_method=hcfg["threshold_method"], n_boot=hcfg["n_boot"], seed=hcfg["seed"])
runs = {}
hyb = {**hcfg["hybrid"], "weights_dir": hcfg.get("weights_dir")}
runs["B_cnn_transformer_hybrid"] = run_image_experiment(ctx, "B_cnn_transformer_hybrid", hyb, hcfg["train"], hcfg["loss"], **common)
ref = {**hcfg["cnn_reference"], "weights_dir": hcfg.get("weights_dir")}
runs[f"B_cnn_reference_{ref['name']}"] = run_image_experiment(ctx, f"B_cnn_reference_{ref['name']}", ref, hcfg["train"], hcfg["loss"], **common)
for fusion in hcfg.get("fusion_ablation", []):
    eid = f"B_hybrid_fusion_{fusion}"
    runs[eid] = run_image_experiment(ctx, eid, {**hyb, "fusion": fusion}, hcfg["train"], hcfg["loss"], **common)
'''),
         md("## 4. Comparison with standalone references (validation for selection, test for reporting)"),
         code(r'''
allres = ctx.results(("A_", "B_"))
rows = []
for _, r in allres.iterrows():
    ok = r["status"] == "succeeded"
    rows.append({"experiment": r["experiment_id"], "status": r["status"],
                 "weights": (r.get("build") or {}).get("status") if isinstance(r.get("build"), dict) else None,
                 "val_macro_auc": r["val_metrics"]["macro_roc_auc"] if ok else np.nan,
                 "test_macro_auc": r["test_metrics"]["macro_roc_auc"] if ok else np.nan,
                 "test_ci_low": r["test_macro_auc_ci"]["ci_low"] if ok and isinstance(r.get("test_macro_auc_ci"), dict) else np.nan,
                 "test_ci_high": r["test_macro_auc_ci"]["ci_high"] if ok and isinstance(r.get("test_macro_auc_ci"), dict) else np.nan})
comp = pd.DataFrame(rows).sort_values("val_macro_auc", ascending=False, na_position="last")
display(comp)
atomic_save_dataframe(comp, ctx.dir("analysis") / "trackB_comparison.csv")
if not allres["experiment_id"].str.startswith("A_").any():
    print("Track-A results not found: attach notebook 01 output to compare against Transformers.")
'''),
         md("## 5. Paired patient-level comparison on the held-out test set (hybrid vs each reference)"),
         code(r'''
from src.evaluation.predictions import load_predictions, align, arrays
from src.evaluation.statistics import paired_bootstrap, holm
dirs = ctx.experiment_dirs()
target = "B_cnn_transformer_hybrid"
others = [e for e in comp.loc[comp["status"] == "succeeded", "experiment"] if e != target and not e.startswith("B_hybrid_fusion")]
recs = []
if target in dirs and (dirs[target] / "predictions_test.parquet").exists():
    base, _ = load_predictions(dirs[target] / "predictions_test.parquet", man_info["manifest_hash"])
    for o in others:
        other, _ = load_predictions(dirs[o] / "predictions_test.parquet", man_info["manifest_hash"])
        ids, al = align({"a": base, "b": other})
        pa, t = arrays(al["a"]); pb, _ = arrays(al["b"])
        r = paired_bootstrap(t, pa, pb, al["a"]["group_id"].to_numpy(), "macro_auc", n_boot=hcfg["n_boot"], seed=0)
        recs.append({"comparison": f"{target} - {o}", "n_images": len(ids), **r})
pc = pd.DataFrame(recs)
if len(pc):
    pc["p_holm"] = holm(pc["p_value"])
display(pc)
atomic_save_dataframe(pc, ctx.dir("analysis") / "trackB_paired_tests.csv") if len(pc) else None
'''),
         md("## 6. Attention maps of the hybrid (qualitative, not segmentation)"),
         code(r'''
from src.training.experiment import load_model_from_result
from src.data.dataset import RSNASliceDataset
from src.data.preprocessing import PreprocessConfig
res = ctx.results(("B_cnn_transformer_hybrid",))
if len(res) and res.iloc[0]["status"] == "succeeded" and res.iloc[0]["config"]["model"].get("fusion", "concat") != "gap_only":
    r0 = res.iloc[0].to_dict()
    model = load_model_from_result(r0, man_info["manifest_hash"])
    test = manifest[(manifest["split"] == "test") & (manifest["any"] >= 0.5)].head(4)
    ds = RSNASliceDataset(test, ctx.rsna.train_dir, PreprocessConfig.from_dict(r0["config"]["preprocess"]))
    fig, axes = plt.subplots(1, max(len(ds), 1), figsize=(3.2 * max(len(ds), 1), 3.4))
    for i in range(len(ds)):
        x = ds[i]["image"][None].to(next(model.parameters()).device)
        with torch.no_grad():
            _, att = model.forward_features(x, return_attention=True)
        a = torch.nn.functional.interpolate(att[None], size=x.shape[-2:], mode="bilinear")[0, 0].cpu().numpy()
        img = ds[i]["image"][0].numpy(); img = (img - img.min()) / (np.ptp(img) + 1e-6)
        ax = np.atleast_1d(axes)[i]; ax.imshow(img, cmap="gray"); ax.imshow(a / (a.max() + 1e-8), cmap="jet", alpha=0.35); ax.axis("off")
        ax.set_title(test.iloc[i]["image_id"], fontsize=7)
    plt.suptitle("[CLS] attention, last block (qualitative; not a segmentation)"); plt.show()
'''),
        ]
    return c


# =====================================================================================================
# 03 Quantum
# =====================================================================================================
def nb03():
    c = [header("03 - Quantum-Classical Hybrid (Track C)",
                "Attach a trainable variational quantum circuit (4 qubits, angle encoding, Rot layers, cyclic CNOT, Pauli-Z "
                "readout) to the best classical backbone (selected on validation) and compare it with parameter-matched classical "
                "controls under one predetermined budget, over several seeds. Record simulator cost and trainability diagnostics.",
                "Notebook 00 output (manifest) and notebook 01/02 outputs (trained backbone checkpoint), same MODE.",
                "`features/<backbone>/features_<split>.npz`, `experiments/C_<head>_s<seed>/`, `analysis/trackC_*.csv`, "
                "`analysis/vqc_circuit.txt`, `analysis/simulator_benchmark.csv`."),
         code(SETUP_CELL), code(DEPENDENCY_CELL), code(context_cell("03_quantum")), code(COMMON_IMPORTS),
         md("## 1. Manifest, configuration and backbone selection (validation only)"),
         code(r'''
from src.training.experiment import select_best_by_validation, load_model_from_result, split_frames, extract_features, run_head_experiment
manifest, man_info = ctx.manifest()
ctx.restore_experiments(("C_",))
qcfg = ctx.cfg["quantum"]
src_cfg = qcfg["backbone"]["source"]
if src_cfg == "auto":
    bb = select_best_by_validation(ctx, tuple(qcfg["backbone"]["candidates_prefix"]))
else:
    r = ctx.results((src_cfg,)); bb = r.iloc[0].to_dict() if len(r) else None
if bb is None:
    raise RuntimeError("No trained backbone found. Run/attach notebook 01 (and/or 02) for this MODE first.")
print("Backbone:", bb["experiment_id"], "| val macro AUC", round(bb["val_metrics"]["macro_roc_auc"], 4),
      "| weights", bb["build"]["status"])
'''),
         md("## 2. Extract backbone features once (frozen backbone; restartable cache)\n"
            "The same per-split frames as the backbone's own evaluation are used, so all heads see identical images."),
         code(r'''
from src.data.preprocessing import PreprocessConfig
backbone = load_model_from_result(bb, man_info["manifest_hash"])
frames = split_frames(manifest, bb["config"]["limits"], bb["config"]["seed"])
pp = PreprocessConfig.from_dict(bb["config"]["preprocess"])
feat_dir = ctx.dir("features", bb["experiment_id"])
source_fp = f'{bb["experiment_id"]}:{bb["config_fingerprint"]}'
feats = extract_features(ctx, backbone, frames, pp, feat_dir,
                         {"source_fingerprint": source_fp, "manifest_hash": man_info["manifest_hash"]},
                         batch_size=qcfg["backbone"]["feature_batch_size"])
print({s: v["features"].shape for s, v in feats.items()})
del backbone; torch.cuda.empty_cache() if torch.cuda.is_available() else None
'''),
         md("## 3. The circuit, its resources and a backend cross-check\n"
            "Two exact simulators are available: a batched PyTorch state-vector simulator (default) and PennyLane. They must "
            "agree; any disagreement stops the notebook."),
         code(r'''
from src.quantum import VQCConfig, VQCLayer, circuit_resources, benchmark_simulator
vdef = {**qcfg["head_defaults"]["vqc"], "n_qubits": qcfg["head_defaults"]["k"]}
print(circuit_resources(VQCConfig(**vdef)))
crosscheck = {"enabled": qcfg["pennylane_crosscheck"]["enabled"]}
try:
    import pennylane as qml
    lt = VQCLayer(VQCConfig(**{**vdef, "backend": "torch"}))
    lp = VQCLayer(VQCConfig(**{**vdef, "backend": "pennylane"})); lp.weights.data = lt.weights.data.clone()
    print(lp.draw()); (ctx.dir("analysis") / "vqc_circuit.txt").write_text(lp.draw(), encoding="utf-8")
    x = torch.randn(qcfg["pennylane_crosscheck"]["n_samples"], vdef["n_qubits"])
    diff = float((lt(x) - lp(x)).abs().max())
    crosscheck.update(pennylane=qml.__version__, max_abs_diff=diff)
    assert diff < 1e-4, f"backends disagree: {diff}"
    bench_rows = benchmark_simulator(vdef, (64, 256)) + benchmark_simulator({**vdef, "backend": "pennylane"}, (64,), repeats=2)
except ImportError:
    crosscheck.update(pennylane="not installed", note="cross-check skipped; torch backend unit-tested against PennyLane offline")
    bench_rows = benchmark_simulator(vdef, (64, 256))
print(crosscheck)
bench = pd.DataFrame(bench_rows); display(bench)
atomic_save_dataframe(bench, ctx.dir("analysis") / "simulator_benchmark.csv")
atomic_write_json(ctx.dir("analysis") / "vqc_crosscheck.json", crosscheck)
'''),
         md("## 4. Predetermined head grid x seeds (primary comparison + bounded ablations)\n"
            "| id | role |\n|---|---|\n| `linear_probe` | upper reference: Linear(d->6) on the same features |\n"
            "| `bottleneck_identity` | control: same k-dim encoder + readout, no processing block |\n"
            "| `bottleneck_mlp` | control: VQC replaced by Linear(k,k)+tanh (parameter-matched) |\n"
            "| `vqc_q4_l2_cyclic` | quantum-classical hybrid (Paper-2 circuit) |\n"
            "| ablations | no entanglement (D), depth 1/4 (E), qubits 2/8 (F) + k=8 classical control |"),
         code(r'''
import copy
def head_cfg(spec):
    d = copy.deepcopy(qcfg["head_defaults"])
    d["kind"] = spec["kind"]
    if "k" in spec: d["k"] = spec["k"]
    d["vqc"] = {**d["vqc"], **spec.get("vqc", {})}
    if spec["kind"] != "vqc": d.pop("vqc")
    return d
grid = list(qcfg["primary"]) + (list(qcfg["ablations"]) if qcfg["run_ablations"] else [])
print(f"{len(grid)} head configurations x {len(qcfg['seeds'])} seeds = {len(grid) * len(qcfg['seeds'])} runs")
head_results = []
for spec in grid:
    for seed in qcfg["seeds"]:
        exp_id = f"C_{spec['id']}_s{seed}"
        r = run_head_experiment(ctx, exp_id, feats, head_cfg(spec), qcfg["train"], qcfg["loss"], frames=frames,
                                manifest_info=man_info, source_fingerprint=source_fp, seed=seed,
                                batch_size=qcfg["batch_size"], threshold_method=qcfg["threshold_method"], n_boot=qcfg["n_boot"])
        head_results.append({"head": spec["id"], "seed": seed, "status": r["status"],
                             "val_macro_auc": r.get("val_metrics", {}).get("macro_roc_auc", np.nan),
                             "test_macro_auc": r.get("test_metrics", {}).get("macro_roc_auc", np.nan),
                             "test_macro_ap": r.get("test_metrics", {}).get("macro_ap", np.nan),
                             "params": (r.get("head_params") or {}).get("total"),
                             "block_params": (r.get("head_params") or {}).get("block"),
                             "train_s": r.get("seconds_total"), "vqc_sim_s": r.get("vqc_sim_seconds"),
                             "epochs": (r.get("fit") or {}).get("epochs_run"), "error": r.get("error")})
hr = pd.DataFrame(head_results)
atomic_save_dataframe(hr, ctx.dir("analysis") / "trackC_runs.csv")
display(hr)
'''),
         md("## 5. Summary across seeds and paired comparisons\n"
            "Seed-level summary (mean, SD, t-interval) and paired tests across seeds; plus a patient-level paired bootstrap on the "
            "held-out test set using the seed-averaged predictions of each head."),
         code(r'''
from src.evaluation.statistics import seed_summary, paired_seed_test, paired_bootstrap, holm
from src.evaluation.predictions import load_predictions, align, arrays
summ = []
for h, g in hr[hr["status"] == "succeeded"].groupby("head", sort=False):
    sv, st = seed_summary(g["val_macro_auc"]), seed_summary(g["test_macro_auc"])
    summ.append({"head": h, "n_seeds": st["n"], "val_mean": sv["mean"], "val_sd": sv["sd"], "test_mean": st["mean"],
                 "test_sd": st["sd"], "test_ci_low": st["ci_low"], "test_ci_high": st["ci_high"],
                 "params": int(g["params"].iloc[0]) if g["params"].notna().any() else None,
                 "mean_train_s": g["train_s"].mean(), "mean_vqc_sim_s": g["vqc_sim_s"].mean()})
summary = pd.DataFrame(summ); display(summary.round(4))
atomic_save_dataframe(summary, ctx.dir("analysis") / "trackC_summary.csv")

def seed_avg_preds(head):
    dfs = [load_predictions(ctx.dir("experiments", f"C_{head}_s{s}") / "predictions_test.parquet", man_info["manifest_hash"])[0]
           for s in qcfg["seeds"] if (ctx.dir("experiments", f"C_{head}_s{s}") / "predictions_test.parquet").exists()]
    if not dfs: return None
    ids, al = align({i: d for i, d in enumerate(dfs)})
    out = al[0].copy()
    probs = np.mean([arrays(d)[0] for d in al.values()], axis=0)
    for j, l in enumerate(LABELS): out[f"prob_{l}"] = probs[:, j]
    return out

QUANTUM = "vqc_q4_l2_cyclic"
comps = []
qa = seed_avg_preds(QUANTUM)
for ctrl in ["bottleneck_mlp", "bottleneck_identity", "linear_probe"]:
    ca = seed_avg_preds(ctrl)
    a, b = hr[(hr["head"] == QUANTUM) & (hr["status"] == "succeeded")], hr[(hr["head"] == ctrl) & (hr["status"] == "succeeded")]
    m = a.merge(b, on="seed", suffixes=("_q", "_c"))
    st = paired_seed_test(m["test_macro_auc_q"], m["test_macro_auc_c"]) if len(m) else {}
    rec = {"comparison": f"{QUANTUM} - {ctrl}", **{f"seeds_{k}": v for k, v in st.items()}}
    if qa is not None and ca is not None:
        ids, al = align({"q": qa, "c": ca})
        pq, t = arrays(al["q"]); pc_, _ = arrays(al["c"])
        rec.update({f"boot_{k}": v for k, v in paired_bootstrap(t, pq, pc_, al["q"]["group_id"].to_numpy(), "macro_auc",
                                                                 n_boot=max(qcfg["n_boot"], 100), seed=0).items() if k != "method"})
    comps.append(rec)
cmp_df = pd.DataFrame(comps)
if "boot_p_value" in cmp_df: cmp_df["boot_p_holm"] = holm(cmp_df["boot_p_value"])
display(cmp_df.round(4))
atomic_save_dataframe(cmp_df, ctx.dir("analysis") / "trackC_paired.csv")
'''),
         md("## 6. Result statement generated from the numbers above (no pre-written conclusion)"),
         code(r'''
def statement(row):
    d, lo, hi = row.get("boot_difference", np.nan), row.get("boot_ci_low", np.nan), row.get("boot_ci_high", np.nan)
    if not np.isfinite(d):
        return f"{row['comparison']}: not evaluable (missing runs)."
    if lo > 0:
        verdict = "the quantum head is better, and the 95% patient-level CI excludes 0"
    elif hi < 0:
        verdict = "the quantum head is WORSE, and the 95% patient-level CI excludes 0"
    else:
        verdict = "no statistically reliable difference (CI includes 0)"
    return f"{row['comparison']}: delta macro AUC = {d:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {verdict}. MODE={MODE}."
for _, r in cmp_df.iterrows():
    print(statement(r))
print("Simulated, noise-free circuit on classical hardware; no claim about quantum hardware or speed-up is made.")
'''),
         md("## 7. Trainability and cost diagnostics"),
         code(r'''
from src.models.hybrid_classifier import FeatureHead
from src.quantum import angle_saturation
diag = []
for spec in grid:
    if spec["kind"] == "vqc":
        d = ctx.dir("experiments", f"C_{spec['id']}_s{qcfg['seeds'][0]}")
        if (d / "history.csv").exists():
            h = pd.read_csv(d / "history.csv")
            diag.append({"head": spec["id"], "mean_grad_norm": h["grad_norm"].mean(), "last_grad_norm": h["grad_norm"].iloc[-1],
                         "epochs": len(h)})
display(pd.DataFrame(diag))
hd = head_cfg(qcfg["primary"][-1]); head = FeatureHead(in_dim=feats["train"]["features"].shape[1], **hd)
from src.training.checkpointing import load_checkpoint
ck = ctx.dir("experiments", f"C_{qcfg['primary'][-1]['id']}_s{qcfg['seeds'][0]}") / "best.pt"
if ck.exists():
    load_checkpoint(ck, head, expected_manifest_hash=man_info["manifest_hash"])
    print("Angle saturation on validation features:", angle_saturation(head, torch.as_tensor(feats["val"]["features"][:4096])))
'''),
        ]
    return c


# =====================================================================================================
# 04 Analysis
# =====================================================================================================
def nb04():
    c = [header("04 - Ablation and Comparative Analysis",
                "Assemble research tables and figures from saved results only (no retraining): Track A comparison, CNN-Transformer "
                "comparison, quantum vs classical controls, quantum ablations, feature-module and preprocessing ablations, "
                "patient-level CIs and paired tests. Missing or failed experiments are listed explicitly.",
                "Outputs of notebooks 00-03 for the same MODE (attach them as inputs on Kaggle).",
                "`analysis/final_*.csv`, `analysis/final_*.md`, `analysis/figures/*.png`, `analysis/experiment_status.csv`."),
         code(SETUP_CELL), code(context_cell("04_analysis")), code(COMMON_IMPORTS),
         code(r'''
manifest, man_info = ctx.manifest()
an_dir, fig_dir = ctx.dir("analysis"), ctx.dir("analysis", "figures")
res = ctx.results()
print(f"Found {len(res)} experiment result files across working dir and attached inputs.")
N_BOOT = {"smoke": 50, "pilot": 300, "full": 1000}[MODE]
'''),
         md("## 1. Planned vs executed experiments (nothing is filled in for missing runs)"),
         code(r'''
tc, hc, qc = ctx.cfg["transformer"], ctx.cfg["hybrid"], ctx.cfg["quantum"]
planned = [(f"A_{m['name']}", "A: standalone Transformer") for m in tc["models"]]
planned += [(f"A_{m['name']}", "A: optional Transformer") for m in tc.get("optional_models", []) if m["name"] in tc.get("run_optional", [])]
planned += [("B_cnn_transformer_hybrid", "G: CNN-Transformer hybrid"), (f"B_cnn_reference_{hc['cnn_reference']['name']}", "CNN reference")]
planned += [(f"B_hybrid_fusion_{f}", "feature-module ablation") for f in hc.get("fusion_ablation", [])]
grid = list(qc["primary"]) + (list(qc["ablations"]) if qc["run_ablations"] else [])
planned += [(f"C_{s['id']}_s{seed}", "C/B/D/E/F: heads on frozen features") for s in grid for seed in qc["seeds"]]
status = {r["experiment_id"]: r["status"] for _, r in res.iterrows()} if len(res) else {}
h_runs = [e for e in status if e.startswith("H_")]
planned += [(e, "H: single-window input") for e in h_runs] or [("H_<best>_brain3", "H: single-window input")]
st = pd.DataFrame([{"experiment": e, "role": role, "status": status.get(e, "NOT RUN / NOT FOUND")} for e, role in planned])
display(st["status"].value_counts().to_frame("count")); display(st)
atomic_save_dataframe(st, an_dir / "experiment_status.csv")
'''),
         md("## 2. Image-model comparison (Tracks A, B, G, H) with patient-level bootstrap CIs on the held-out test set"),
         code(r'''
from src.evaluation.predictions import load_predictions, arrays, align
from src.evaluation.statistics import bootstrap_table, paired_bootstrap, holm
from src.evaluation.visualization import plot_forest
img = res[res["experiment_id"].str.match(r"^(A|B|H)_") & (res["status"] == "succeeded")] if len(res) else res
rows, preds = [], {}
for _, r in img.iterrows():
    d = Path(r["_dir"])
    df, _ = load_predictions(d / "predictions_test.parquet", man_info["manifest_hash"])
    preds[r["experiment_id"]] = df
    p, t = arrays(df)
    ci = bootstrap_table(t, p, df["group_id"].to_numpy(), ["macro_auc", "macro_ap", "auc:any"], n_boot=N_BOOT, seed=0).set_index("metric")
    per = pd.read_csv(d / "metrics_test_per_label.csv").set_index("label")["roc_auc"]
    rows.append({"model": r["experiment_id"], "weights": r["build"]["status"], "val_macro_auc": r["val_metrics"]["macro_roc_auc"],
                 "estimate": ci.loc["macro_auc", "estimate"], "ci_low": ci.loc["macro_auc", "ci_low"], "ci_high": ci.loc["macro_auc", "ci_high"],
                 "test_macro_ap": ci.loc["macro_ap", "estimate"], **{f"auc_{l}": per[l] for l in LABELS},
                 "test_macro_f1": r["test_metrics"]["macro_f1"], "test_hamming": r["test_metrics"]["hamming_loss"],
                 "test_brier": r["test_metrics"]["mean_brier"]})
image_table = pd.DataFrame(rows).sort_values("val_macro_auc", ascending=False) if rows else pd.DataFrame()
display(image_table.round(4))
if len(image_table):
    atomic_save_dataframe(image_table, an_dir / "final_image_models.csv")
    (an_dir / "final_image_models.md").write_text(dataframe_to_markdown(image_table), encoding="utf-8")
    plot_forest(image_table, "model", fig_dir / "forest_image_models.png", "Held-out test macro ROC-AUC (patient bootstrap 95% CI)"); plt.show()
'''),
         md("## 3. Paired comparisons on identical test images (patient-cluster bootstrap, Holm-adjusted)\n"
            "Reference = the best model by **validation** macro AUC."),
         code(r'''
recs = []
if len(image_table) >= 2:
    ref = image_table.iloc[0]["model"]
    for o in image_table["model"].iloc[1:]:
        ids, al = align({"a": preds[ref], "b": preds[o]})
        pa, t = arrays(al["a"]); pb, _ = arrays(al["b"])
        recs.append({"comparison": f"{ref} - {o}", "n_images": len(ids),
                     **paired_bootstrap(t, pa, pb, al["a"]["group_id"].to_numpy(), "macro_auc", N_BOOT, 0)})
paired = pd.DataFrame(recs)
if len(paired):
    paired["p_holm"] = holm(paired["p_value"])
    atomic_save_dataframe(paired, an_dir / "final_paired_image_models.csv")
display(paired.round(4))
'''),
         md("## 4. Quantum vs classical controls and quantum ablations (from notebook 03 runs)"),
         code(r'''
from src.evaluation.statistics import seed_summary, paired_seed_test
c_res = res[res["experiment_id"].str.startswith("C_")] if len(res) else res
if len(c_res):
    c_res = c_res.assign(head=c_res["experiment_id"].str.replace(r"_s\d+$", "", regex=True).str[2:],
                         seed=c_res["experiment_id"].str.extract(r"_s(\d+)$")[0].astype(int))
    ok = c_res[c_res["status"] == "succeeded"].copy()
    ok["test_macro_auc"] = ok["test_metrics"].apply(lambda m: m["macro_roc_auc"])
    ok["val_macro_auc"] = ok["val_metrics"].apply(lambda m: m["macro_roc_auc"])
    ok["params"] = ok["head_params"].apply(lambda m: m["total"])
    q_table = pd.DataFrame([{"head": h, **{f"test_{k}": v for k, v in seed_summary(g["test_macro_auc"]).items()},
                             "val_mean": g["val_macro_auc"].mean(), "params": g["params"].iloc[0],
                             "failed_runs": int((c_res[(c_res["head"] == h)]["status"] != "succeeded").sum())}
                            for h, g in ok.groupby("head", sort=False)])
    display(q_table.round(4)); atomic_save_dataframe(q_table, an_dir / "final_quantum_heads.csv")
    (an_dir / "final_quantum_heads.md").write_text(dataframe_to_markdown(q_table), encoding="utf-8")
    from src.evaluation.visualization import plot_seed_comparison
    plot_seed_comparison(ok, "head", "test_macro_auc", fig_dir / "quantum_heads_by_seed.png", "Per-seed held-out test macro AUC"); plt.show()
    tests = []
    for a, b, label in [("vqc_q4_l2_cyclic", "bottleneck_mlp", "C vs B (parameter-matched classical block)"),
                        ("vqc_q4_l2_cyclic", "bottleneck_identity", "C vs bottleneck only"),
                        ("vqc_q4_l2_cyclic", "linear_probe", "C vs linear probe (A-features)"),
                        ("vqc_q4_l2_cyclic", "vqc_q4_l2_noent", "D: entanglement vs none"),
                        ("vqc_q4_l4_cyclic", "vqc_q4_l2_cyclic", "E: depth 4 vs 2"), ("vqc_q4_l1_cyclic", "vqc_q4_l2_cyclic", "E: depth 1 vs 2"),
                        ("vqc_q8_l2_cyclic", "bottleneck_mlp_k8", "F: 8 qubits vs k=8 classical"), ("vqc_q2_l2_cyclic", "vqc_q4_l2_cyclic", "F: 2 vs 4 qubits")]:
        A, B = ok[ok["head"] == a], ok[ok["head"] == b]
        if len(A) and len(B):
            m = A.merge(B, on="seed", suffixes=("_a", "_b"))
            tests.append({"ablation": label, "a": a, "b": b, **paired_seed_test(m["test_macro_auc_a"], m["test_macro_auc_b"])})
        else:
            tests.append({"ablation": label, "a": a, "b": b, "n": 0, "note": "not run"})
    abl = pd.DataFrame(tests); display(abl.round(4)); atomic_save_dataframe(abl, an_dir / "final_quantum_ablations.csv")
else:
    print("No Track-C results found (attach notebook 03 output).")
'''),
         md("## 5. Statistical method (for the report)\n"
            "* **CIs:** patient-cluster percentile bootstrap (patients resampled with replacement, all their slices kept). Slices are not treated as independent.\n"
            "* **Paired model comparisons:** paired patient-cluster bootstrap of the macro-AUC difference on identical test images; two-sided bootstrap p-value; Holm correction across comparisons.\n"
            "* **Seed variability (Track C):** mean, SD and t-interval across training seeds; paired t-test / Wilcoxon across seeds (small n - interpret with caution).\n"
            "* **Thresholds** were selected on validation (Youden J per label) and frozen before test evaluation.\n"
            "* **Assumptions:** patients are independent; percentile intervals may be anti-conservative when few patients are positive for a label (epidural)."),
        ]
    return c


# =====================================================================================================
# 05 Triage & visualisation
# =====================================================================================================
def nb05():
    c = [header("05 - ICH Triage and Visualization",
                "Aggregate slice-level probabilities of the selected model into study-level triage scores, report subtype "
                "probabilities and high-risk slices, produce Grad-CAM explanations, and (optionally) run competition-test inference.",
                "Outputs of notebooks 00 and 01/02 (same MODE).",
                "`triage/study_level_<agg>.csv`, `triage/study_metrics.csv`, `triage/triage_table_test.csv`, `triage/figures/*`, optional `submission/submission.csv`."),
         code(SETUP_CELL), code(context_cell("05_triage")), code(COMMON_IMPORTS),
         md("> **Not a clinical tool.** Study-level outputs are aggregated model predictions on retrospective research data. "
            "They have not been clinically validated and must not be used for diagnosis or patient management."),
         code(r'''
from src.training.experiment import select_best_by_validation, load_model_from_result
from src.evaluation.predictions import load_predictions
manifest, man_info = ctx.manifest()
best = select_best_by_validation(ctx, ("A_", "B_"))
if best is None:
    raise RuntimeError("No trained image model found for this MODE (attach notebook 01/02 output).")
print("Selected (validation):", best["experiment_id"], round(best["val_metrics"]["macro_roc_auc"], 4))
d = Path(best["_dir"]); tri_dir, fig_dir = ctx.dir("triage"), ctx.dir("triage", "figures")
pv, _ = load_predictions(d / "predictions_val.parquet", man_info["manifest_hash"])
pt, _ = load_predictions(d / "predictions_test.parquet", man_info["manifest_hash"])
for df in (pv, pt):
    if "study_uid" not in df or df["study_uid"].isna().all():
        df.drop(columns=[c for c in ("study_uid", "patient_id", "slice_rank") if c in df], inplace=True)
        df[["study_uid", "patient_id", "slice_rank"]] = manifest.set_index("image_id").loc[df["image_id"], ["study_uid", "patient_id", "slice_rank"]].to_numpy()
'''),
         md("## 1. Study-level aggregation (threshold chosen on validation studies, applied to test studies)"),
         code(r'''
from src.evaluation.study_aggregation import aggregate_studies, AGGREGATORS, triage_table, high_risk_slices
from src.evaluation.metrics import compute_metrics
from src.evaluation.thresholds import select_thresholds
from src.evaluation.statistics import bootstrap_table
N_BOOT = {"smoke": 50, "pilot": 300, "full": 1000}[MODE]
rows = {}
for agg in AGGREGATORS:
    sv, st = aggregate_studies(pv, agg), aggregate_studies(pt, agg)
    P = lambda s: s[[f"prob_{l}" for l in LABELS]].to_numpy(); T = lambda s: s[[f"target_{l}" for l in LABELS]].to_numpy()
    ts = select_thresholds(T(sv), P(sv), split="val", manifest_hash=man_info["manifest_hash"], experiment_id=f"study_{agg}")
    m = compute_metrics(T(st), P(st), ts.thresholds, threshold_source="validation studies (frozen)")
    ci = bootstrap_table(T(st), P(st), st["patient_id"].to_numpy(), ["macro_auc", "auc:any"], n_boot=N_BOOT).set_index("metric")
    rows[agg] = {"n_test_studies": len(st), "macro_auc": m.aggregate["macro_roc_auc"], "macro_auc_ci": (ci.loc["macro_auc", "ci_low"], ci.loc["macro_auc", "ci_high"]),
                 "any_auc": m.aggregate.get("any_roc_auc"), "any_auc_ci": (ci.loc["auc:any", "ci_low"], ci.loc["auc:any", "ci_high"]),
                 "any_sensitivity": m.per_label.set_index("label").loc["any", "sensitivity"],
                 "any_specificity": m.per_label.set_index("label").loc["any", "specificity"], "any_threshold": ts.thresholds["any"]}
    atomic_save_dataframe(st, tri_dir / f"study_level_{agg}_test.csv")
study_metrics = pd.DataFrame(rows).T; display(study_metrics)
atomic_save_dataframe(study_metrics.reset_index(names="aggregation"), tri_dir / "study_metrics.csv")
if man_info.get("partial_header_scan"):
    print("NOTE: partial header scan (smoke) - studies are incomplete, so study-level numbers are not meaningful.")
'''),
         md("## 2. Triage table and high-risk slices (max aggregation; validation-selected study threshold)"),
         code(r'''
AGG = "max"
st = aggregate_studies(pt, AGG)
tt = triage_table(st, float(study_metrics.loc[AGG, "any_threshold"]))
display(tt.head(10)[["study_uid", "n_slices", "prob_any", "flag_any", "most_likely_subtype", *[f"prob_{l}" for l in LABELS[:-1]], "target_any"]].round(3))
atomic_save_dataframe(tt, tri_dir / "triage_table_test.csv")
for s in tt["study_uid"].head(2):
    print("Study", s); display(high_risk_slices(pt, s, top_k=5).round(3))
'''),
         md("## 3. Grad-CAM explanations (qualitative; not segmentation, not validated localisation)"),
         code(r'''
from src.evaluation.explainability import GradCAM, default_cam_target
from src.evaluation.visualization import overlay_cam
from src.data.dataset import RSNASliceDataset
from src.data.preprocessing import PreprocessConfig
model = load_model_from_result(best, man_info["manifest_hash"])
pp = PreprocessConfig.from_dict(best["config"]["preprocess"])
pos = pt[pt["target_any"] >= 0.5].sort_values("prob_any", ascending=False).head(4)
if len(pos):
    layer, layout, prefix = default_cam_target(model)
    cam = GradCAM(model, layer, layout, prefix)
    ds = RSNASliceDataset(pos[["image_id"]], ctx.rsna.train_dir, pp)
    raw = RSNASliceDataset(pos[["image_id"]], ctx.rsna.train_dir, {**best["config"]["preprocess"], "normalization": "none"})
    fig, axes = plt.subplots(1, len(ds), figsize=(3.3 * len(ds), 3.6))
    dev = next(model.parameters()).device
    for i in range(len(ds)):
        row = pos.iloc[i]
        lab = max(range(5), key=lambda j: row[f"prob_{LABELS[j]}"])
        heat = cam(ds[i]["image"][None].to(dev), lab)[0].numpy()
        overlay_cam(raw[i]["image"][0].numpy(), heat, np.atleast_1d(axes)[i],
                    title=f"{row['image_id']}\nGrad-CAM for {LABELS[lab]} (p={row[f'prob_{LABELS[lab]}']:.2f})")
    cam.remove(); fig.suptitle("Model saliency (qualitative). NOT a segmentation."); fig.savefig(fig_dir / "gradcam_examples.png", dpi=120); plt.show()
'''),
         md("## 4. Optional: competition-test inference (`stage_2_test`, no labels) -> submission file"),
         code(r'''
RUN_COMPETITION_INFERENCE = False
RUN_COMPETITION_INFERENCE = RUN_COMPETITION_INFERENCE or os.environ.get("RSNA_ICH_RUN_COMPETITION_INFERENCE") == "1"  # test hook
MAX_TEST_IMAGES = {"smoke": 64, "pilot": None, "full": None}[MODE]
if RUN_COMPETITION_INFERENCE:
    from src.data.csv_parser import parse_submission_ids
    from src.data.dataset import make_loader
    from src.training.trainer import predict
    from src.evaluation.predictions import predictions_frame
    from src.evaluation.submission import make_submission, save_submission
    ids = parse_submission_ids(ctx.rsna.sample_submission)
    if MAX_TEST_IMAGES: ids = ids.head(MAX_TEST_IMAGES)
    dl = make_loader(RSNASliceDataset(ids, ctx.rsna.test_dir, pp), 64, False, 2)
    out = predict(model, dl, next(model.parameters()).device)
    sub = make_submission(predictions_frame(out["image_ids"], out["probs"]), ctx.rsna.sample_submission)
    print("images with no prediction (filled with 0.5):", sub.attrs["filled"], "| undecodable:", int((~out["ok"]).sum()))
    save_submission(sub, ctx.dir("submission") / "submission.csv"); display(sub.head())
else:
    print("Competition inference disabled (set RUN_COMPETITION_INFERENCE = True).")
'''),
        ]
    return c


NOTEBOOKS = {
    "00_Data_Audit_and_Preprocessing.ipynb": nb00,
    "01_Classical_Transformer_Benchmark.ipynb": nb01,
    "02_CNN_Transformer_Hybrid.ipynb": nb02,
    "03_Quantum_Classical_Hybrid.ipynb": nb03,
    "04_Ablation_and_Comparative_Analysis.ipynb": nb04,
    "05_ICH_Triage_and_Visualization.ipynb": nb05,
}


def build():
    NB_DIR.mkdir(exist_ok=True)
    for name, fn in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = fn()
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"},
                       "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True}}
        nbf.validate(nb)
        nbf.write(nb, NB_DIR / name)
        print("wrote", name, len(nb.cells), "cells")


if __name__ == "__main__":
    build()
