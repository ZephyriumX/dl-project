"""CT preprocessing: DICOM -> HU -> windowed channels -> resized float tensor.

Identical code is used for training, validation, held-out test and competition-test inference;
``preprocessing_fingerprint(cfg)`` is stored with every checkpoint so a mismatch is detectable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import cv2
import numpy as np
import pydicom

WINDOW_PRESETS: dict[str, tuple[float, float]] = {
    # name: (level/center, width) in HU
    "brain": (40.0, 80.0),
    "subdural": (80.0, 200.0),
    "soft_tissue": (40.0, 380.0),
    "bone": (600.0, 2800.0),
    "stroke": (32.0, 8.0),
}

CHANNEL_MODES: dict[str, tuple[str, ...]] = {
    "bsb": ("brain", "subdural", "soft_tissue"),  # default multi-window input (as in the reference notebooks)
    "bsbone": ("brain", "subdural", "bone"),
    "brain3": ("brain", "brain", "brain"),  # single-window ablation (3 identical channels for ImageNet stems)
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
AIR_HU = -1024.0


class DicomDecodeError(RuntimeError):
    pass


@dataclass
class PreprocessConfig:
    image_size: int = 224
    channel_mode: str = "bsb"
    windows: Sequence[str] | None = None  # explicit list overrides channel_mode
    rsna_intercept_fix: bool = True
    pad_to_square: bool = True
    normalization: str = "imagenet"  # "imagenet" | "none" | "zero_center"
    interpolation: str = "area"  # "area" (downsampling) | "linear"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping | None) -> "PreprocessConfig":
        d = dict(d or {})
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__}
        return cls(**known, extra=d) if d else cls(**known)

    @property
    def window_names(self) -> tuple[str, ...]:
        if self.windows:
            return tuple(self.windows)
        if self.channel_mode not in CHANNEL_MODES:
            raise ValueError(f"Unknown channel_mode {self.channel_mode!r}; options {list(CHANNEL_MODES)}")
        return CHANNEL_MODES[self.channel_mode]

    @property
    def n_channels(self) -> int:
        return len(self.window_names)


def preprocessing_fingerprint(cfg: PreprocessConfig | Mapping) -> str:
    if isinstance(cfg, PreprocessConfig):
        cfg = asdict(cfg)
    cfg = dict(cfg)
    names = cfg.get("windows") or CHANNEL_MODES.get(cfg.get("channel_mode", "bsb"))
    cfg["_resolved_windows"] = [(n, WINDOW_PRESETS[n]) for n in names]
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def apply_rsna_intercept_fix(px: np.ndarray, ds: pydicom.Dataset) -> tuple[np.ndarray, float, bool]:
    """Known RSNA quirk (also handled in the extended reference notebook): some 12-bit unsigned
    slices carry an intercept of about 0 while the stored values are offset. Shift by +1000 and wrap
    values >= 4096, then use intercept -1000. Returns (pixels, intercept, applied)."""
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    if (int(getattr(ds, "BitsStored", 16)) == 12 and int(getattr(ds, "PixelRepresentation", 0)) == 0
            and inter > -100):
        px = px + 1000
        px[px >= 4096] -= 4096
        return px, -1000.0, True
    return px, inter, False


def dicom_to_hu(ds: pydicom.Dataset, rsna_intercept_fix: bool = True) -> np.ndarray:
    """Convert a decoded dataset to Hounsfield units (float32)."""
    try:
        px = ds.pixel_array
    except Exception as e:  # noqa: BLE001
        raise DicomDecodeError(f"pixel data could not be decoded: {type(e).__name__}: {e}") from e
    px = px.astype(np.float32)
    if px.ndim != 2:
        if px.ndim == 3 and px.shape[0] == 1:
            px = px[0]
        else:
            raise DicomDecodeError(f"unexpected pixel array shape {px.shape}")
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    if rsna_intercept_fix:
        px, inter, _ = apply_rsna_intercept_fix(px, ds)
    else:
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = px * slope + inter
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        hu = hu.max() + hu.min() - hu  # invert intensities around the range centre
    return np.maximum(hu, AIR_HU).astype(np.float32)


def window(hu: np.ndarray, level: float, width: float) -> np.ndarray:
    lo, hi = level - width / 2.0, level + width / 2.0
    return ((np.clip(hu, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def pad_square(img: np.ndarray, fill: float) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    s = max(h, w)
    out = np.full((s, s) + img.shape[2:], fill, dtype=img.dtype)
    y, x = (s - h) // 2, (s - w) // 2
    out[y:y + h, x:x + w] = img
    return out


def hu_to_channels(hu: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """HU (H, W) -> windowed channels (C, S, S) in [0, 1], float32."""
    if cfg.pad_to_square:
        hu = pad_square(hu, AIR_HU)
    interp = cv2.INTER_AREA if cfg.interpolation == "area" else cv2.INTER_LINEAR
    if hu.shape[0] != cfg.image_size or hu.shape[1] != cfg.image_size:
        if hu.shape[0] < cfg.image_size:
            interp = cv2.INTER_LINEAR
        hu = cv2.resize(hu, (cfg.image_size, cfg.image_size), interpolation=interp)
    chans = [window(hu, *WINDOW_PRESETS[n]) for n in cfg.window_names]
    return np.stack(chans, axis=0)


def to_uint8(ch: np.ndarray) -> np.ndarray:
    return np.clip(ch * 255.0 + 0.5, 0, 255).astype(np.uint8)


def normalize(x: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Channel normalisation applied after windowing (and after uint8 caching, if used)."""
    if cfg.normalization == "none":
        return x
    if cfg.normalization == "zero_center":
        return (x - 0.5) / 0.5
    if cfg.normalization == "imagenet":
        c = x.shape[0]
        mean = np.asarray(IMAGENET_MEAN[:c] if c <= 3 else [0.5] * c, np.float32).reshape(-1, 1, 1)
        std = np.asarray(IMAGENET_STD[:c] if c <= 3 else [0.25] * c, np.float32).reshape(-1, 1, 1)
        return (x - mean) / std
    raise ValueError(f"unknown normalization {cfg.normalization!r}")


def load_dicom_channels(path: str, cfg: PreprocessConfig) -> np.ndarray:
    """Read one DICOM file and return windowed channels (C, S, S) float32 in [0, 1]."""
    try:
        ds = pydicom.dcmread(path, force=True)
    except FileNotFoundError:
        raise
    except Exception as e:  # noqa: BLE001
        raise DicomDecodeError(f"cannot parse {path}: {type(e).__name__}: {e}") from e
    hu = dicom_to_hu(ds, cfg.rsna_intercept_fix)
    return hu_to_channels(hu, cfg)
