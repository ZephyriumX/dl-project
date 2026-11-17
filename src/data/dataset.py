"""PyTorch datasets with on-demand DICOM loading and bounded caching.

Nothing is preloaded: every item is decoded when requested. Two optional caches:

* an in-process LRU of ``uint8`` channel arrays (``memory_cache_items``), and
* an on-disk ``uint8`` ``.npy`` cache under ``cache_dir/<preprocessing fingerprint>/`` capped at
  ``disk_cache_max_gb``; writes are atomic so the cache survives interrupted sessions.

Caching stores *windowed uint8* channels (before normalisation and augmentation), so a cached
item is identical to a freshly decoded one up to uint8 quantisation (<= 1/255 per channel).
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .. import LABELS
from ..utils.repro import seed_worker
from .preprocessing import (DicomDecodeError, PreprocessConfig, load_dicom_channels, normalize,
                            preprocessing_fingerprint, to_uint8)


class _LRU:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, k):
        if self.capacity <= 0:
            return None
        with self._lock:
            v = self._d.get(k)
            if v is not None:
                self._d.move_to_end(k)
            return v

    def __getstate__(self):  # locks cannot be pickled (DataLoader workers on spawn platforms)
        return {"capacity": self.capacity}

    def __setstate__(self, state):
        self.__init__(state["capacity"])

    def put(self, k, v):
        if self.capacity <= 0:
            return
        with self._lock:
            self._d[k] = v
            self._d.move_to_end(k)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)


class DiskCache:
    """Bounded, restartable uint8 cache keyed by image_id under a fingerprinted directory."""

    def __init__(self, root: str | os.PathLike, fingerprint: str, max_gb: float):
        self.dir = Path(root) / fingerprint
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb * 1e9)
        self._bytes = sum(p.stat().st_size for p in self.dir.glob("*.npy"))
        self._lock = threading.Lock()

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_lock", None)
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def path(self, image_id: str) -> Path:
        return self.dir / f"{image_id}.npy"

    def get(self, image_id: str):
        p = self.path(image_id)
        if p.exists():
            try:
                return np.load(p)
            except Exception:  # noqa: BLE001 - corrupt file: ignore, will be rewritten
                return None
        return None

    def put(self, image_id: str, arr: np.ndarray) -> bool:
        with self._lock:
            if self._bytes + arr.nbytes > self.max_bytes:
                return False
            self._bytes += arr.nbytes
        p = self.path(image_id)
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, p)
        return True


# ----------------------------------------------------------------------------------------------
# augmentation (torch, on CHW float tensors in [0,1] before normalisation)
# ----------------------------------------------------------------------------------------------
class CTAugment:
    """Conservative CT augmentation: flips, small affine, mild intensity gain.

    No colour jitter: channels are physical HU windows, not RGB.
    """

    def __init__(self, hflip: float = 0.5, rotate_deg: float = 10.0, translate: float = 0.05,
                 scale: tuple[float, float] = (0.9, 1.1), intensity_gain: float = 0.05, p_affine: float = 0.7):
        self.hflip, self.rotate_deg, self.translate = hflip, rotate_deg, translate
        self.scale, self.intensity_gain, self.p_affine = scale, intensity_gain, p_affine

    def __call__(self, x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        if rng.random() < self.hflip:
            x = torch.flip(x, dims=[-1])
        if rng.random() < self.p_affine:
            import torch.nn.functional as F

            ang = np.deg2rad(rng.uniform(-self.rotate_deg, self.rotate_deg))
            sc = rng.uniform(*self.scale)
            tx, ty = rng.uniform(-self.translate, self.translate, size=2) * 2
            cos, sin = np.cos(ang) / sc, np.sin(ang) / sc
            theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=x.dtype).unsqueeze(0)
            grid = F.affine_grid(theta, size=(1, *x.shape), align_corners=False)
            x = F.grid_sample(x.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
        if self.intensity_gain:
            x = (x * float(rng.uniform(1 - self.intensity_gain, 1 + self.intensity_gain))).clamp_(0, 1)
        return x


# ----------------------------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------------------------
class RSNASliceDataset(Dataset):
    """One item per CT slice (DICOM file). Works for labelled manifests and unlabelled test IDs.

    Returns ``dict(image=FloatTensor[C,S,S], target=FloatTensor[6] (NaN if unlabelled),
    image_id=str, index=int, ok=bool)``. If a file cannot be decoded and ``on_error='blank'``,
    a blank image is returned with ``ok=False`` so the failure is visible downstream;
    ``on_error='raise'`` re-raises.
    """

    def __init__(self, frame: pd.DataFrame, dicom_dir: str | os.PathLike, preprocess: PreprocessConfig | dict,
                 augment: CTAugment | None = None, memory_cache_items: int = 0, disk_cache_dir: str | None = None,
                 disk_cache_max_gb: float = 0.0, on_error: str = "blank", seed: int = 0,
                 loader: Callable[[str, PreprocessConfig], np.ndarray] | None = None):
        self.df = frame.reset_index(drop=True)
        self.ids = self.df["image_id"].astype(str).tolist()
        self.labelled = all(c in self.df.columns for c in LABELS)
        self.targets = (self.df[list(LABELS)].to_numpy(np.float32) if self.labelled
                        else np.full((len(self.df), len(LABELS)), np.nan, np.float32))
        self.dicom_dir = Path(dicom_dir)
        self.cfg = preprocess if isinstance(preprocess, PreprocessConfig) else PreprocessConfig.from_dict(preprocess)
        self.fingerprint = preprocessing_fingerprint(self.cfg)
        self.augment = augment
        self.mem = _LRU(memory_cache_items)
        self.disk = DiskCache(disk_cache_dir, self.fingerprint, disk_cache_max_gb) if (disk_cache_dir and disk_cache_max_gb > 0) else None
        self.on_error = on_error
        self.seed = seed
        self._epoch = 0
        self._loader = loader or load_dicom_channels  # module-level function: picklable for worker processes
        self.n_failed = 0

    def __len__(self) -> int:
        return len(self.ids)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _channels_uint8(self, image_id: str) -> tuple[np.ndarray, bool]:
        v = self.mem.get(image_id)
        if v is not None:
            return v, True
        if self.disk is not None:
            v = self.disk.get(image_id)
            if v is not None:
                self.mem.put(image_id, v)
                return v, True
        try:
            ch = self._loader(str(self.dicom_dir / f"{image_id}.dcm"), self.cfg)
            v = to_uint8(ch)
        except (DicomDecodeError, FileNotFoundError, ValueError, OSError):
            if self.on_error == "raise":
                raise
            self.n_failed += 1
            return np.zeros((self.cfg.n_channels, self.cfg.image_size, self.cfg.image_size), np.uint8), False
        self.mem.put(image_id, v)
        if self.disk is not None:
            self.disk.put(image_id, v)
        return v, True

    def __getitem__(self, idx: int) -> dict:
        image_id = self.ids[idx]
        u8, ok = self._channels_uint8(image_id)
        x = torch.from_numpy(u8.astype(np.float32) / 255.0)
        if self.augment is not None:
            rng = np.random.default_rng((self.seed, self._epoch, idx))
            x = self.augment(x, rng)
        x = torch.from_numpy(normalize(x.numpy(), self.cfg))
        return {"image": x, "target": torch.from_numpy(self.targets[idx]), "image_id": image_id,
                "index": idx, "ok": ok}


class SyntheticCTDataset(Dataset):
    """Random tensors with random multi-label targets: for unit tests and shape checks only."""

    def __init__(self, n: int = 32, channels: int = 3, size: int = 64, seed: int = 0, positive_rate: float = 0.3):
        g = np.random.default_rng(seed)
        self.x = g.normal(size=(n, channels, size, size)).astype(np.float32)
        sub = (g.random((n, 5)) < positive_rate).astype(np.float32)
        self.y = np.concatenate([sub, sub.max(1, keepdims=True)], axis=1)
        # make the signal learnable: positives get a brighter blob
        for i in range(n):
            if self.y[i, 5]:
                self.x[i, :, size // 4: size // 2, size // 4: size // 2] += 2.0
        self.ids = [f"SYN_{i:06d}" for i in range(n)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"image": torch.from_numpy(self.x[i]), "target": torch.from_numpy(self.y[i]),
                "image_id": self.ids[i], "index": i, "ok": True}


class FeatureDataset(Dataset):
    """Pre-extracted backbone features (for the quantum/classical head comparison)."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, image_ids: Sequence[str]):
        assert len(features) == len(targets) == len(image_ids)
        self.f = torch.as_tensor(features, dtype=torch.float32)
        self.y = torch.as_tensor(targets, dtype=torch.float32)
        self.ids = list(image_ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"image": self.f[i], "target": self.y[i], "image_id": self.ids[i], "index": i, "ok": True}


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "image_id": [b["image_id"] for b in batch],
        "index": torch.as_tensor([b["index"] for b in batch]),
        "ok": torch.as_tensor([bool(b["ok"]) for b in batch]),
    }


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int = 2, pin_memory: bool = True,
                seed: int = 0, drop_last: bool = False, persistent_workers: bool | None = None) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory and torch.cuda.is_available(), collate_fn=collate,
                      worker_init_fn=seed_worker, generator=g, drop_last=drop_last,
                      persistent_workers=persistent_workers if num_workers > 0 else False)


def build_augment(cfg: dict | None) -> CTAugment | None:
    if not cfg or not cfg.get("enabled", True):
        return None
    kw = {k: v for k, v in cfg.items() if k != "enabled"}
    if "scale" in kw:
        kw["scale"] = tuple(kw["scale"])
    return CTAugment(**kw)
