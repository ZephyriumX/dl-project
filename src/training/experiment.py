"""High-level experiment runners shared by notebooks 01-05.

``run_image_experiment``  : image model (Track A/B and CNN references) end-to-end on the manifest.
``extract_features``      : frozen backbone -> cached feature arrays (Track C input).
``run_head_experiment``   : feature head (linear / classical bottleneck / VQC) on cached features.

Every runner: fingerprints its full configuration, skips work that already succeeded with the same
fingerprint and manifest, records ``running/succeeded/failed/skipped`` in the registry, saves
predictions (val + held-out test), selects thresholds on validation only, evaluates the test split
with the frozen thresholds, and computes patient-cluster bootstrap CIs.
"""

from __future__ import annotations

import gc
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .. import LABELS
from ..data.dataset import FeatureDataset, RSNASliceDataset, build_augment, make_loader
from ..data.preprocessing import PreprocessConfig, preprocessing_fingerprint
from ..data.splitting import subset
from ..evaluation.metrics import compute_metrics
from ..evaluation.predictions import predictions_frame, save_predictions
from ..evaluation.statistics import bootstrap_table
from ..evaluation.thresholds import select_thresholds
from ..utils.config import config_fingerprint
from ..utils.io import atomic_save_dataframe, atomic_write_json, read_json
from ..utils.repro import set_seed
from ..utils.runtime import probe_batch_size, suggest_settings
from .losses import build_loss
from .trainer import TrainConfig, Trainer, predict

GROUP_COLS = ["image_id", "group_id", "patient_id", "study_uid", "series_key", "slice_rank"]


# ----------------------------------------------------------------------------------------------
# data helpers
# ----------------------------------------------------------------------------------------------
def split_frames(manifest: pd.DataFrame, limits: dict | None, seed: int = 0,
                 splits=("train", "val", "test")) -> dict[str, pd.DataFrame]:
    """Per-split frames with optional *patient-level* caps (``limits[split] = {max_groups, max_images}``)."""
    limits = limits or {}
    out = {}
    for s in splits:
        lim = limits.get(s, {}) or {}
        out[s] = subset(manifest, s, max_images=lim.get("max_images"), seed=seed, max_groups=lim.get("max_groups"))
    neg_ratio = (limits.get("train", {}) or {}).get("train_negative_ratio")
    if neg_ratio and "train" in out:
        f = out["train"]
        pos = f[f["any"] >= 0.5]
        neg = f[f["any"] < 0.5]
        k = min(len(neg), int(neg_ratio * max(len(pos), 1)))
        out["train"] = pd.concat([pos, neg.sample(n=k, random_state=seed)]).reset_index(drop=True)
    return out


def build_datasets(ctx, frames: dict[str, pd.DataFrame], pp: PreprocessConfig, augment_cfg: dict | None,
                   seed: int = 0) -> dict[str, RSNASliceDataset]:
    cache = ctx.cfg["data"].get("cache", {})
    disk_dir = str(ctx.dir("cache")) if cache.get("disk_cache_max_gb", 0) > 0 else None
    ds = {}
    for s, f in frames.items():
        ds[s] = RSNASliceDataset(f, ctx.rsna.train_dir, pp, augment=build_augment(augment_cfg) if s == "train" else None,
                                 memory_cache_items=cache.get("memory_cache_items", 0), disk_cache_dir=disk_dir,
                                 disk_cache_max_gb=cache.get("disk_cache_max_gb", 0.0), seed=seed,
                                 on_error=ctx.cfg["data"].get("on_decode_error", "blank"))
    return ds


def _extra_cols(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[[c for c in GROUP_COLS if c in frame.columns]].copy()


# ----------------------------------------------------------------------------------------------
# evaluation shared by all runners
# ----------------------------------------------------------------------------------------------
def evaluate_and_save(out_dir: Path, exp_id: str, manifest_hash: str, preds: dict[str, dict],
                      frames: dict[str, pd.DataFrame], threshold_method: str = "youden", n_boot: int = 1000,
                      seed: int = 0) -> dict:
    """``preds[split] = predict(...) output``. Requires 'val'; 'test' optional."""
    out: dict[str, Any] = {}
    files = {}
    for s, p in preds.items():
        df = predictions_frame(p["image_ids"], p["probs"], p["targets"], _extra_cols(frames[s]))
        df["decode_ok"] = p["ok"]
        files[s] = str(save_predictions(df, out_dir / f"predictions_{s}.parquet",
                                        {"manifest_hash": manifest_hash, "experiment_id": exp_id, "split": s}))
    ts = select_thresholds(preds["val"]["targets"], preds["val"]["probs"], split="val", manifest_hash=manifest_hash,
                           experiment_id=exp_id, method=threshold_method)
    ts.save(out_dir / "thresholds.json")
    out["thresholds"] = ts.thresholds
    for s, p in preds.items():
        src = "validation-selected (frozen)" if s != "val" else "validation-selected (in-sample for val)"
        m = compute_metrics(p["targets"], p["probs"], ts.thresholds, threshold_source=src)
        atomic_save_dataframe(m.per_label, out_dir / f"metrics_{s}_per_label.csv")
        atomic_write_json(out_dir / f"metrics_{s}.json", m.to_dict())
        out[f"{s}_metrics"] = m.aggregate
        out[f"{s}_notes"] = m.notes
        out[f"{s}_n_decode_failed"] = int((~p["ok"]).sum())
    if "test" in preds and n_boot > 0 and len(preds["test"]["image_ids"]):
        groups = frames["test"].set_index("image_id").loc[preds["test"]["image_ids"], "group_id"].to_numpy()
        ci = bootstrap_table(preds["test"]["targets"], preds["test"]["probs"], groups, n_boot=n_boot, seed=seed)
        atomic_save_dataframe(ci, out_dir / "test_bootstrap_ci.csv")
        out["test_macro_auc_ci"] = ci.loc[ci["metric"] == "macro_auc", ["estimate", "ci_low", "ci_high"]].iloc[0].to_dict()
    out["prediction_files"] = files
    return out


def _already_done(ctx, exp_id: str, fingerprint: str, out_dir: Path) -> dict | None:
    rp = out_dir / "result.json"
    if rp.exists():
        r = read_json(rp)
        if r.get("config_fingerprint") == fingerprint and r.get("status") == "succeeded":
            return r
    return None


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------
# image experiments
# ----------------------------------------------------------------------------------------------
def run_image_experiment(ctx, exp_id: str, model_cfg: dict, train_cfg: dict, loss_cfg: dict, *,
                         manifest: pd.DataFrame, manifest_info: dict, preprocess: dict | None = None,
                         augment: dict | None = None, limits: dict | None = None, loader_cfg: dict | None = None,
                         threshold_method: str = "youden", n_boot: int = 1000, seed: int = 42,
                         raise_on_error: bool = False, force: bool = False) -> dict:
    from ..models import create_model

    pp = PreprocessConfig.from_dict(preprocess or ctx.cfg["data"].get("preprocessing"))
    model_cfg = {**model_cfg, "in_chans": pp.n_channels}
    loader_cfg = dict(loader_cfg or {})
    fp_cfg = {"model": model_cfg, "train": train_cfg, "loss": loss_cfg, "preprocess": asdict(pp), "augment": augment,
              "limits": limits, "seed": seed, "threshold_method": threshold_method,
              "batch_size": loader_cfg.get("batch_size")}
    fingerprint = config_fingerprint(fp_cfg)
    out_dir = ctx.dir("experiments", exp_id)
    if not force:
        done = _already_done(ctx, exp_id, fingerprint, out_dir)
        if done:
            print(f"[{exp_id}] already succeeded with identical configuration - loading saved results.")
            return done
    mh = manifest_info["manifest_hash"]
    base = {"experiment_id": exp_id, "config_fingerprint": fingerprint, "manifest_hash": mh, "mode": ctx.mode,
            "config": fp_cfg}
    ctx.registry.record(exp_id, "running", config_fingerprint=fingerprint, manifest_hash=mh,
                        model=model_cfg.get("name", model_cfg.get("kind")))
    t0 = time.time()
    model = None
    try:
        set_seed(seed)
        model, rec = create_model(model_cfg)
        rec_d = rec.as_dict()
        if model is None:
            status = "skipped" if model_cfg.get("fallback") == "skip" else "failed"
            r = {**base, "status": status, "build": rec_d, "reason": rec.error}
            atomic_write_json(out_dir / "result.json", r)
            ctx.registry.record(exp_id, status, reason=rec.error, config_fingerprint=fingerprint, manifest_hash=mh)
            print(f"[{exp_id}] model could not be built ({status}): {rec.error}")
            return r
        print(f"[{exp_id}] built {rec.timm_name}: status={rec.status} params={rec.n_params / 1e6:.1f}M")

        frames = split_frames(manifest, limits, seed)
        ds = build_datasets(ctx, frames, pp, augment, seed)
        rt = ctx.runtime
        st = suggest_settings(rt, model_cfg.get("name", model_cfg.get("kind", "")), pp.image_size,
                              requested_batch=loader_cfg.get("batch_size"))
        tcfg = TrainConfig.from_dict({**train_cfg, "amp": train_cfg.get("amp", True) and st.amp,
                                      "amp_dtype": train_cfg.get("amp_dtype") or st.amp_dtype})
        device = torch.device(st.device)
        bs = st.batch_size
        if loader_cfg.get("probe_batch_size", True) and device.type == "cuda":
            model.to(device)

            def step(b):
                x = torch.randn(b, pp.n_channels, pp.image_size, pp.image_size, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16 if tcfg.amp_dtype == "bfloat16" else torch.float16,
                                    enabled=tcfg.amp):
                    model(x).float().mean().backward()
                model.zero_grad(set_to_none=True)

            model.train()
            bs = probe_batch_size(step, bs)
            _free()
        workers = loader_cfg.get("num_workers")
        workers = st.num_workers if workers is None else int(workers)
        print(f"[{exp_id}] device={device} batch={bs} workers={workers} amp={tcfg.amp}({tcfg.amp_dtype}) "
              f"train/val/test images={len(ds['train'])}/{len(ds['val'])}/{len(ds['test'])}")
        dl = {s: make_loader(d, bs if s == "train" else bs * 2, shuffle=(s == "train"), num_workers=workers,
                             seed=seed, drop_last=(s == "train" and len(d) > bs)) for s, d in ds.items()}
        criterion, loss_desc = build_loss(loss_cfg, frames["train"][list(LABELS)].to_numpy())
        meta = {**base, "build": rec_d, "preprocess_fingerprint": preprocessing_fingerprint(pp),
                "preprocess": asdict(pp), "loss": loss_desc, "batch_size": bs}
        trainer = Trainer(model, criterion, tcfg, out_dir, meta, device=device)
        fit = trainer.fit(dl["train"], dl["val"], resume=True)
        preds = {s: predict(model, dl[s], device, tcfg.amp, tcfg.amp_dtype, tcfg.max_eval_steps) for s in ("val", "test")}
        ev = evaluate_and_save(out_dir, exp_id, mh, preds, frames, threshold_method, n_boot, seed)
        r = {**base, "status": "succeeded", "build": rec_d, "fit": {k: v for k, v in fit.as_dict().items() if k != "history"},
             "loss": loss_desc, "batch_size": bs, "n_images": {s: len(d) for s, d in ds.items()},
             "decode_failures_train": ds["train"].n_failed, "seconds_total": time.time() - t0,
             "weights_note": "pretrained" if rec.is_pretrained else f"NOT pretrained ({rec.status})", **ev}
        atomic_write_json(out_dir / "result.json", r)
        ctx.registry.record(exp_id, "succeeded", config_fingerprint=fingerprint, manifest_hash=mh,
                            val_macro_auc=ev["val_metrics"]["macro_roc_auc"],
                            weights_status=rec.status, seconds=time.time() - t0)
        return r
    except Exception as e:  # noqa: BLE001
        oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
        r = {**base, "status": "failed", "error": f"{type(e).__name__}: {e}", "oom": oom,
             "traceback": traceback.format_exc(limit=8)}
        atomic_write_json(out_dir / "result.json", r)
        ctx.registry.record_failure(exp_id, e, config_fingerprint=fingerprint, manifest_hash=mh, oom=oom)
        print(f"[{exp_id}] FAILED: {type(e).__name__}: {e}")
        if raise_on_error:
            raise
        return r
    finally:
        del model
        _free()


# ----------------------------------------------------------------------------------------------
# selection and reloading (validation-based only)
# ----------------------------------------------------------------------------------------------
def select_best_by_validation(ctx, prefixes: tuple[str, ...] = ("A_", "B_"), metric: str = "macro_roc_auc",
                              exclude_substrings: tuple[str, ...] = ("_brain3",)) -> dict | None:
    """Best *succeeded* image experiment by VALIDATION metric. The test split plays no role here."""
    res = ctx.results(prefixes)
    if res.empty:
        return None
    res = res[res["status"] == "succeeded"]
    res = res[~res["experiment_id"].apply(lambda e: any(s in e for s in exclude_substrings))]
    if res.empty:
        return None
    res = res.assign(_val=res["val_metrics"].apply(lambda m: (m or {}).get(metric, float("nan"))))
    res = res.dropna(subset=["_val"]).sort_values("_val", ascending=False)
    return res.iloc[0].to_dict() if len(res) else None


def load_model_from_result(result: dict, manifest_hash: str, device: str | torch.device | None = None):
    """Rebuild an image model from its result.json and load ``best.pt`` (manifest hash enforced)."""
    from ..models import create_model
    from .checkpointing import load_checkpoint

    cfg = dict(result["config"]["model"])
    cfg["pretrained"] = False  # weights come from the checkpoint
    model, rec = create_model(cfg)
    if model is None:
        raise RuntimeError(f"cannot rebuild model {cfg}: {rec.error}")
    d = Path(result["_dir"]) if "_dir" in result else None
    ck = (d / "best.pt") if d is not None else Path(result["fit"]["best_path"])
    load_checkpoint(ck, model, expected_manifest_hash=manifest_hash, map_location="cpu")
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
    return model


# ----------------------------------------------------------------------------------------------
# feature extraction (Track C)
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def extract_features(ctx, model: torch.nn.Module, frames: dict[str, pd.DataFrame], pp: PreprocessConfig, out_dir: Path,
                     meta: dict, batch_size: int = 128, num_workers: int = 2, amp: bool = True) -> dict[str, dict]:
    """Run ``model.forward_features`` once per split and cache arrays (restartable, meta-checked)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    res = {}
    for s, f in frames.items():
        path, mpath = out_dir / f"features_{s}.npz", out_dir / f"features_{s}.json"
        if path.exists() and mpath.exists():
            m = read_json(mpath)
            if m.get("source_fingerprint") == meta["source_fingerprint"] and m.get("manifest_hash") == meta["manifest_hash"] \
                    and m.get("n") == len(f):
                d = np.load(path, allow_pickle=False)
                res[s] = {"features": d["features"], "targets": d["targets"], "image_ids": d["image_ids"].astype(str)}
                print(f"[features] {s}: loaded cached {d['features'].shape}")
                continue
        ds = RSNASliceDataset(f, ctx.rsna.train_dir, pp, on_error=ctx.cfg["data"].get("on_decode_error", "blank"))
        dl = make_loader(ds, batch_size, False, num_workers)
        feats, tg, ids = [], [], []
        t0 = time.time()
        for b in dl:
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                z = model.forward_features(b["image"].to(device, non_blocking=True))
            feats.append(z.float().cpu().numpy())
            tg.append(b["target"].numpy())
            ids.extend(b["image_id"])
        F_, T_ = np.concatenate(feats), np.concatenate(tg)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, features=F_.astype(np.float32), targets=T_, image_ids=np.asarray(ids))
        tmp.replace(path)
        atomic_write_json(mpath, {**meta, "split": s, "n": len(f), "dim": int(F_.shape[1]), "seconds": time.time() - t0})
        res[s] = {"features": F_, "targets": T_, "image_ids": np.asarray(ids)}
        print(f"[features] {s}: extracted {F_.shape} in {time.time() - t0:.0f}s")
    return res


# ----------------------------------------------------------------------------------------------
# head experiments on cached features (Track C and its classical controls)
# ----------------------------------------------------------------------------------------------
def run_head_experiment(ctx, exp_id: str, feats: dict[str, dict], head_cfg: dict, train_cfg: dict, loss_cfg: dict, *,
                        frames: dict[str, pd.DataFrame], manifest_info: dict, source_fingerprint: str, seed: int = 0,
                        batch_size: int = 512, threshold_method: str = "youden", n_boot: int = 0,
                        force: bool = False) -> dict:
    from ..models.hybrid_classifier import FeatureHead
    from ..quantum.quantum_hybrid import find_vqc

    fp_cfg = {"head": head_cfg, "train": train_cfg, "loss": loss_cfg, "seed": seed, "batch_size": batch_size,
              "source": source_fingerprint, "threshold_method": threshold_method}
    fingerprint = config_fingerprint(fp_cfg)
    out_dir = ctx.dir("experiments", exp_id)
    if not force:
        done = _already_done(ctx, exp_id, fingerprint, out_dir)
        if done:
            return done
    mh = manifest_info["manifest_hash"]
    base = {"experiment_id": exp_id, "config_fingerprint": fingerprint, "manifest_hash": mh, "mode": ctx.mode,
            "config": fp_cfg}
    ctx.registry.record(exp_id, "running", config_fingerprint=fingerprint, manifest_hash=mh, head=head_cfg.get("kind"))
    t0 = time.time()
    try:
        set_seed(seed)
        in_dim = int(feats["train"]["features"].shape[1])
        head = FeatureHead(in_dim=in_dim, **head_cfg)
        dsets = {s: FeatureDataset(v["features"], v["targets"], v["image_ids"]) for s, v in feats.items()}
        dl = {s: make_loader(d, batch_size, shuffle=(s == "train"), num_workers=0, seed=seed, pin_memory=False)
              for s, d in dsets.items()}
        criterion, loss_desc = build_loss(loss_cfg, feats["train"]["targets"])
        tcfg = TrainConfig.from_dict({**train_cfg, "backbone_lr": None, "amp": False})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        meta = {**base, "head_params": head.param_breakdown()}
        trainer = Trainer(head, criterion, tcfg, out_dir, meta, device=device)
        fit = trainer.fit(dl["train"], dl["val"], resume=True)
        preds = {s: predict(head, dl[s], device, False) for s in ("val", "test")}
        ev = evaluate_and_save(out_dir, exp_id, mh, preds, frames, threshold_method, n_boot, seed)
        vqc = find_vqc(head)
        r = {**base, "status": "succeeded", "fit": {k: v for k, v in fit.as_dict().items() if k != "history"},
             "head_params": head.param_breakdown(), "loss": loss_desc, "seconds_total": time.time() - t0,
             "vqc_resources": vqc.resources() if vqc is not None else None,
             "vqc_sim_seconds": vqc.sim_seconds if vqc is not None else 0.0,
             "vqc_sim_calls": vqc.sim_calls if vqc is not None else 0,
             "history": fit.history, **ev}
        atomic_write_json(out_dir / "result.json", r)
        ctx.registry.record(exp_id, "succeeded", config_fingerprint=fingerprint, manifest_hash=mh,
                            val_macro_auc=ev["val_metrics"]["macro_roc_auc"], seconds=time.time() - t0)
        return r
    except Exception as e:  # noqa: BLE001
        r = {**base, "status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=8)}
        atomic_write_json(out_dir / "result.json", r)
        ctx.registry.record_failure(exp_id, e, config_fingerprint=fingerprint, manifest_hash=mh)
        print(f"[{exp_id}] FAILED: {type(e).__name__}: {e}")
        return r
