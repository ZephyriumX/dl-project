"""Generic, restartable trainer used by every track.

Features: AdamW with separate backbone/head learning rates, warm-up + cosine (per step) or
plateau scheduling, gradient clipping, AMP (bf16 or fp16 + GradScaler), gradient accumulation,
optional frozen-backbone warm-up epochs, early stopping on validation macro ROC-AUC,
``last.pt`` / ``best.pt`` checkpoints, exact resume, per-epoch history (CSV + JSON) and
gradient-norm logging (used for the VQC trainability analysis).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .. import LABELS
from ..evaluation.metrics import per_label_auc
from ..utils.io import atomic_save_dataframe, atomic_write_json, read_json
from ..utils.repro import restore_rng_state
from .checkpointing import load_checkpoint, read_checkpoint_meta, save_checkpoint


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 3e-4  # head / new layers
    backbone_lr: float | None = 5e-5  # None -> same as lr
    weight_decay: float = 0.05
    scheduler: str = "cosine"  # cosine | plateau | none
    warmup_epochs: float = 0.5
    min_lr_ratio: float = 0.01
    grad_clip: float | None = 1.0
    amp: bool = True
    amp_dtype: str = "float16"  # float16 | bfloat16
    accum_steps: int = 1
    early_stopping_patience: int = 3
    min_delta: float = 1e-4
    monitor: str = "val_macro_auc"
    max_train_steps_per_epoch: int | None = None
    max_eval_steps: int | None = None
    freeze_backbone_epochs: int = 0
    log_every: int = 100
    channels_last: bool = False

    @classmethod
    def from_dict(cls, d: dict | None) -> "TrainConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FitResult:
    best_metric: float
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    resumed_from_epoch: int | None
    best_path: str
    last_path: str
    train_seconds: float
    history: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def _batch_to(batch: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["image"].to(device, non_blocking=True), batch["target"].to(device, non_blocking=True)


@torch.no_grad()
def predict(model: nn.Module, loader, device, amp: bool = True, amp_dtype: str = "float16",
            max_steps: int | None = None, criterion: nn.Module | None = None) -> dict:
    """Run inference. Returns dict(image_ids, probs (N,6), targets (N,6), ok (N,), loss)."""
    model.eval()
    ids, probs, targets, oks, losses, n = [], [], [], [], 0.0, 0
    dt = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    dev_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x, y = _batch_to(batch, device)
        with torch.autocast(device_type=dev_type, dtype=dt, enabled=amp and dev_type == "cuda"):
            logits = model(x)
        logits = logits.float()
        if criterion is not None and torch.isfinite(y).all():
            losses += float(criterion(logits, y)) * len(x)
            n += len(x)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(batch["target"].numpy())
        ids.extend(batch["image_id"])
        oks.append(batch["ok"].numpy())
    if not probs:
        return {"image_ids": [], "probs": np.zeros((0, len(LABELS))), "targets": np.zeros((0, len(LABELS))),
                "ok": np.zeros(0, bool), "loss": float("nan")}
    return {"image_ids": ids, "probs": np.concatenate(probs), "targets": np.concatenate(targets),
            "ok": np.concatenate(oks).astype(bool), "loss": losses / n if n else float("nan")}


def _param_groups(model: nn.Module, cfg: TrainConfig) -> list[dict]:
    if hasattr(model, "backbone_parameters") and cfg.backbone_lr is not None:
        bb = {id(p) for p in model.backbone_parameters()}
        backbone = [p for p in model.parameters() if id(p) in bb]
        rest = [p for p in model.parameters() if id(p) not in bb]
        groups = [{"params": rest, "lr": cfg.lr, "name": "head"}]
        if backbone:
            groups.append({"params": backbone, "lr": cfg.backbone_lr, "name": "backbone"})
        return groups
    return [{"params": list(model.parameters()), "lr": cfg.lr, "name": "all"}]


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    if hasattr(model, "backbone_parameters"):
        for p in model.backbone_parameters():
            p.requires_grad_(trainable)


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, cfg: TrainConfig, out_dir: str | Path, meta: dict,
                 device: str | torch.device | None = None, on_epoch_end: Callable[[dict], None] | None = None):
        self.model, self.criterion, self.cfg = model, criterion, cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta)
        self.meta.setdefault("train_config", asdict(cfg))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.on_epoch_end = on_epoch_end
        self.model.to(self.device)
        if cfg.channels_last:
            self.model.to(memory_format=torch.channels_last)
        self.criterion.to(self.device)
        self.amp = bool(cfg.amp and self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.amp_dtype == torch.float16)
        self.optimizer = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)
        self.scheduler = None
        self.best_path = self.out_dir / "best.pt"
        self.last_path = self.out_dir / "last.pt"

    # ------------------------------------------------------------------ scheduling
    def _make_scheduler(self, steps_per_epoch: int):
        cfg = self.cfg
        if cfg.scheduler == "cosine":
            total = max(1, cfg.epochs * steps_per_epoch)
            warm = int(cfg.warmup_epochs * steps_per_epoch)

            def f(step):
                if step < warm:
                    return (step + 1) / max(1, warm)
                prog = min(1.0, (step - warm) / max(1, total - warm))
                return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, f)
        if cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.3, patience=1)
        return None

    # ------------------------------------------------------------------ main loop
    def fit(self, train_loader, val_loader, resume: bool = True) -> FitResult:
        cfg = self.cfg
        steps = len(train_loader)
        if cfg.max_train_steps_per_epoch:
            steps = min(steps, cfg.max_train_steps_per_epoch)
        opt_steps = max(1, steps // cfg.accum_steps)
        self.scheduler = self._make_scheduler(opt_steps)
        start_epoch, best, best_epoch, history, resumed = 0, -float("inf"), -1, [], None
        bad_epochs = 0

        done_marker = self.out_dir / "done.json"
        if resume and done_marker.exists() and self.best_path.exists():
            info = read_json(done_marker)
            if info.get("config_fingerprint") == self.meta["config_fingerprint"] and \
                    info.get("manifest_hash") == self.meta["manifest_hash"]:
                print(f"[trainer] {self.out_dir.name}: already completed (best epoch {info.get('best_epoch')}); "
                      "loading best checkpoint instead of retraining.")
                load_checkpoint(self.best_path, self.model, expected_manifest_hash=self.meta["manifest_hash"])
                return FitResult(**{k: info[k] for k in FitResult.__dataclass_fields__ if k in info})
        if resume and self.last_path.exists():
            m = read_checkpoint_meta(self.last_path) or {}
            if m.get("config_fingerprint") == self.meta["config_fingerprint"] and m.get("manifest_hash") == self.meta["manifest_hash"]:
                ck = load_checkpoint(self.last_path, self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                                     scaler=self.scaler, map_location=self.device,
                                     expected_manifest_hash=self.meta["manifest_hash"])
                start_epoch = ck["epoch"] + 1
                best = ck["best_metric"] if ck["best_metric"] is not None else best
                history = ck.get("history", [])
                best_epoch = int(ck.get("extra", {}).get("best_epoch", -1))
                bad_epochs = int(ck.get("extra", {}).get("bad_epochs", 0))
                restore_rng_state(ck.get("rng"))
                resumed = ck["epoch"]
                print(f"[trainer] resumed from epoch {resumed} (best {best:.4f} @ {best_epoch})")
            else:
                print("[trainer] existing last.pt has a different config/manifest - starting fresh in a new run is "
                      "required; refusing to mix. Remove the directory or change the experiment name.")
                raise RuntimeError(f"Incompatible checkpoint in {self.out_dir}")

        t_start = time.time()
        stopped_early = False
        for epoch in range(start_epoch, cfg.epochs):
            frozen = epoch < cfg.freeze_backbone_epochs
            _set_backbone_trainable(self.model, not frozen)
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            tr = self._train_epoch(train_loader, steps, epoch)
            ev = predict(self.model, val_loader, self.device, self.amp, self.cfg.amp_dtype, cfg.max_eval_steps,
                         self.criterion)
            aucs = per_label_auc(ev["targets"], ev["probs"])
            defined = [a for a in aucs.values() if a == a]
            macro = float(np.mean(defined)) if defined else float("nan")
            rec = {"epoch": epoch, "train_loss": tr["loss"], "val_loss": ev["loss"], "val_macro_auc": macro,
                   **{f"val_auc_{k}": v for k, v in aucs.items()}, "lr": tr["lr"], "grad_norm": tr["grad_norm"],
                   "backbone_frozen": frozen, "epoch_seconds": tr["seconds"]}
            history.append(rec)
            monitored = rec[cfg.monitor]
            improved = monitored == monitored and monitored > best + cfg.min_delta
            if improved:
                best, best_epoch, bad_epochs = monitored, epoch, 0
                save_checkpoint(self.best_path, model=self.model, meta=self.meta, epoch=epoch, best_metric=best,
                                history=history, allow_overwrite_different_config=False)
            else:
                bad_epochs += 1
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(monitored if monitored == monitored else 0.0)
            save_checkpoint(self.last_path, model=self.model, meta=self.meta, optimizer=self.optimizer,
                            scheduler=self.scheduler, scaler=self.scaler, epoch=epoch, best_metric=best, history=history,
                            extra={"best_epoch": best_epoch, "bad_epochs": bad_epochs})
            atomic_save_dataframe(pd.DataFrame(history), self.out_dir / "history.csv")
            print(f"[epoch {epoch + 1}/{cfg.epochs}] train_loss={tr['loss']:.4f} val_loss={ev['loss']:.4f} "
                  f"val_macro_auc={macro:.4f} {'*' if improved else ''} ({tr['seconds']:.0f}s)", flush=True)
            if self.on_epoch_end:
                self.on_epoch_end(rec)
            if bad_epochs >= cfg.early_stopping_patience:
                stopped_early = True
                print(f"[trainer] early stopping at epoch {epoch + 1} (no improvement for {bad_epochs} epochs)")
                break
        if not self.best_path.exists():  # e.g. every epoch had undefined AUC (tiny smoke data)
            save_checkpoint(self.best_path, model=self.model, meta=self.meta, epoch=len(history) - 1,
                            best_metric=None, history=history)
        res = FitResult(best_metric=float(best), best_epoch=int(best_epoch), epochs_run=len(history),
                        stopped_early=stopped_early, resumed_from_epoch=resumed, best_path=str(self.best_path),
                        last_path=str(self.last_path), train_seconds=time.time() - t_start, history=history)
        atomic_write_json(done_marker, {**res.as_dict(), "config_fingerprint": self.meta["config_fingerprint"],
                                        "manifest_hash": self.meta["manifest_hash"]})
        load_checkpoint(self.best_path, self.model, expected_manifest_hash=self.meta["manifest_hash"])
        return res

    def _train_epoch(self, loader, steps: int, epoch: int) -> dict:
        self.model.train()
        cfg = self.cfg
        t0, tot, n, gn_sum, gn_n = time.time(), 0.0, 0, 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            if step >= steps:
                break
            x, y = _batch_to(batch, self.device)
            if cfg.channels_last and x.dim() == 4:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                logits = self.model(x)
            loss = self.criterion(logits.float(), y) / cfg.accum_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            self.scaler.scale(loss).backward()
            if (step + 1) % cfg.accum_steps == 0 or step + 1 == steps:
                self.scaler.unscale_(self.optimizer)
                params = [p for g in self.optimizer.param_groups for p in g["params"] if p.grad is not None]
                if params:
                    gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip if cfg.grad_clip else float("inf"))
                    if torch.isfinite(gn):
                        gn_sum, gn_n = gn_sum + float(gn), gn_n + 1
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
            tot += float(loss) * cfg.accum_steps * len(x)
            n += len(x)
            if cfg.log_every and (step + 1) % cfg.log_every == 0:
                print(f"  step {step + 1}/{steps} loss={tot / max(n, 1):.4f}", flush=True)
        return {"loss": tot / max(n, 1), "seconds": time.time() - t0,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "grad_norm": gn_sum / gn_n if gn_n else float("nan")}
