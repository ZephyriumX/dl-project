"""Runtime resource detection and resource-aware training settings.

The project never assumes a specific Kaggle GPU (P100 16 GB, T4 15 GB, 2xT4, or CPU-only).
``detect_runtime()`` reports what is actually available, ``suggest_settings()`` derives
batch size / AMP / workers from it, and ``probe_batch_size()`` confirms the batch fits by
running forward+backward and halving on OOM.
"""

from __future__ import annotations

import gc
import importlib
import os
import platform
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

PACKAGES = ("torch", "torchvision", "timm", "pennylane", "pydicom", "numpy", "pandas",
            "sklearn", "scipy", "matplotlib", "yaml", "cv2")


@dataclass
class RuntimeInfo:
    python: str
    platform: str
    cuda_available: bool
    gpu_count: int
    gpu_names: list[str]
    gpu_total_mem_gb: list[float]
    gpu_free_mem_gb: list[float]
    bf16_supported: bool
    cpu_count: int
    ram_total_gb: float | None
    ram_available_gb: float | None
    working_dir: str
    working_free_gb: float | None
    packages: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        g = ", ".join(f"{n} ({m:.1f} GB)" for n, m in zip(self.gpu_names, self.gpu_total_mem_gb)) or "none"
        return (f"GPU: {g} | CUDA={self.cuda_available} bf16={self.bf16_supported} | CPUs={self.cpu_count} | "
                f"RAM avail={self.ram_available_gb} GB | free disk ({self.working_dir})={self.working_free_gb} GB")


def package_versions(names=PACKAGES) -> dict[str, str]:
    out = {}
    for n in names:
        try:
            m = importlib.import_module(n)
            out[n] = str(getattr(m, "__version__", "unknown"))
        except Exception as e:  # noqa: BLE001
            out[n] = f"unavailable ({type(e).__name__})"
    return out


def detect_runtime(working_dir: str | os.PathLike | None = None) -> RuntimeInfo:
    wd = Path(working_dir or ("/kaggle/working" if Path("/kaggle/working").exists() else Path.cwd()))
    try:
        import psutil

        vm = psutil.virtual_memory()
        ram_total, ram_avail = round(vm.total / 1e9, 2), round(vm.available / 1e9, 2)
    except Exception:  # noqa: BLE001
        ram_total = ram_avail = None
    try:
        free = round(shutil.disk_usage(wd).free / 1e9, 2)
    except Exception:  # noqa: BLE001
        free = None
    names, tot, fre = [], [], []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            names.append(p.name)
            tot.append(round(p.total_memory / 1e9, 2))
            try:
                f, _ = torch.cuda.mem_get_info(i)
                fre.append(round(f / 1e9, 2))
            except Exception:  # noqa: BLE001
                fre.append(float("nan"))
    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    # T4/P100 report bf16 "supported" via emulation in some builds; only trust it on compute capability >= 8.
    if bf16 and torch.cuda.get_device_capability(0)[0] < 8:
        bf16 = False
    return RuntimeInfo(
        python=platform.python_version(), platform=platform.platform(),
        cuda_available=torch.cuda.is_available(), gpu_count=len(names), gpu_names=names,
        gpu_total_mem_gb=tot, gpu_free_mem_gb=fre, bf16_supported=bf16,
        cpu_count=os.cpu_count() or 1, ram_total_gb=ram_total, ram_available_gb=ram_avail,
        working_dir=str(wd), working_free_gb=free, packages=package_versions(),
    )


# Approximate per-image training memory (GB) at 224 px with AMP, measured order-of-magnitude values
# used only as a starting point; probe_batch_size() confirms on the real device.
_MEM_PER_IMAGE_GB = {
    "deit_small_patch16_224": 0.055, "vit_small_patch16_224": 0.055,
    "swin_tiny_patch4_window7_224": 0.10, "maxvit_tiny_tf_224": 0.20,
    "beit_base_patch16_224": 0.16, "cnn_transformer_hybrid": 0.07, "efficientnet_b0": 0.06,
}


@dataclass
class TrainSettings:
    device: str
    batch_size: int
    amp: bool
    amp_dtype: str
    num_workers: int
    pin_memory: bool
    channels_last: bool
    notes: list[str] = field(default_factory=list)


def suggest_settings(rt: RuntimeInfo, model_key: str, img_size: int = 224, requested_batch: int | None = None,
                     max_batch: int = 128, min_batch: int = 4, max_workers: int = 4) -> TrainSettings:
    notes = []
    if not rt.cuda_available:
        notes.append("No CUDA device: running on CPU (only smoke-scale runs are realistic).")
        return TrainSettings("cpu", min(requested_batch or 8, 8), False, "float32",
                             min(2, rt.cpu_count), False, False, notes)
    mem = (rt.gpu_free_mem_gb[0] if rt.gpu_free_mem_gb and rt.gpu_free_mem_gb[0] == rt.gpu_free_mem_gb[0]
           else rt.gpu_total_mem_gb[0])
    per = _MEM_PER_IMAGE_GB.get(model_key, 0.12) * (img_size / 224) ** 2
    usable = max(mem - 1.5, 0.5)  # leave headroom for CUDA context / fragmentation
    bs = int(usable / per)
    bs = max(min_batch, min(max_batch, 2 ** max(0, bs.bit_length() - 1)))
    if requested_batch:
        if requested_batch > bs:
            notes.append(f"Requested batch {requested_batch} exceeds estimate {bs} for {mem:.1f} GB; using {bs}.")
        bs = min(bs, requested_batch)
    dtype = "bfloat16" if rt.bf16_supported else "float16"
    workers = max(0, min(max_workers, rt.cpu_count - 1))
    return TrainSettings("cuda", bs, True, dtype, workers, True, False, notes)


def probe_batch_size(step_fn: Callable[[int], None], start: int, min_batch: int = 2) -> int:
    """Halve the batch size until ``step_fn(batch)`` (one forward+backward) succeeds without OOM."""
    bs = start
    while bs >= min_batch:
        try:
            step_fn(bs)
            return bs
        except torch.cuda.OutOfMemoryError:
            pass
        except RuntimeError as e:  # older torch raises plain RuntimeError
            if "out of memory" not in str(e).lower():
                raise
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bs //= 2
    raise RuntimeError(f"Even batch size {min_batch} does not fit in GPU memory.")
