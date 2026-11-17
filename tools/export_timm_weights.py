"""Export pretrained timm weights to a folder that can be uploaded as a Kaggle Dataset.

Run once on a machine (or a Kaggle notebook) WITH internet access:

    python tools/export_timm_weights.py --out ./timm_weights

Upload ``timm_weights/`` as a Kaggle Dataset, attach it to notebooks 01-03, and set
``weights_dir`` in ``configs/transformer_config.yaml`` / ``hybrid_config.yaml`` to its mount path
(e.g. ``/kaggle/input/timm-weights-rsna``). ``src.models.transformers`` then loads the files
offline and records ``status = pretrained_local_file``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import timm
import torch

DEFAULT = ["deit_small_patch16_224", "swin_tiny_patch4_window7_224", "maxvit_tiny_tf_224", "efficientnet_b0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=DEFAULT)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.models:
        m = timm.create_model(name, pretrained=True)
        p = out / f"{name}.pth"
        torch.save(m.state_dict(), p)
        print(f"{name}: {p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
