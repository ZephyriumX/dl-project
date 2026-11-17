"""Track A: pretrained vision Transformers via ``timm`` with explicit weight-status records.

Kaggle notebooks frequently run without internet. Weight resolution order:

1. a local file in ``weights_dir`` (an attached Kaggle dataset), named ``<model_name>.(safetensors|pth|bin)``
   or inside ``<weights_dir>/<model_name>/``;
2. the normal timm / Hugging Face download (needs internet);
3. the ``fallback`` policy: ``"random_init"`` (train from scratch, status recorded) or ``"fail"``.

A model built from random initialisation is **never** labelled as pretrained.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .. import NUM_LABELS

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None

TRANSFORMER_CANDIDATES = {
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "maxvit_tiny": "maxvit_tiny_tf_224",
    # optional extras (enabled from config only if available and affordable)
    "vit_small": "vit_small_patch16_224",
    "beit_base": "beit_base_patch16_224",
}


@dataclass
class ModelBuildRecord:
    name: str
    timm_name: str
    requested_pretrained: bool
    status: str  # pretrained | pretrained_local_file | random_init | random_init_fallback | failed
    source: str = ""
    error: str = ""
    num_features: int = 0
    n_params: int = 0
    n_trainable: int = 0

    @property
    def is_pretrained(self) -> bool:
        return self.status in ("pretrained", "pretrained_local_file")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["is_pretrained"] = self.is_pretrained
        return d


def _find_local_weights(weights_dir: str | os.PathLike | None, timm_name: str) -> Path | None:
    if not weights_dir:
        return None
    root = Path(weights_dir)
    if not root.exists():
        return None
    names = [timm_name, timm_name.split(".")[0]]
    for n in names:
        for ext in (".safetensors", ".pth", ".bin", ".pt"):
            for p in (root / f"{n}{ext}", root / n / f"model{ext}", root / n / f"pytorch_model{ext}"):
                if p.is_file():
                    return p
    return None


def count_params(m: nn.Module) -> tuple[int, int]:
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_backbone(name: str, pretrained: bool = True, weights_dir: str | None = None, fallback: str = "random_init",
                   in_chans: int = 3, drop_path_rate: float = 0.0, **timm_kwargs) -> tuple[nn.Module | None, ModelBuildRecord]:
    """Create a timm model with ``num_classes=0`` (pooled feature output)."""
    if timm is None:
        raise ImportError("timm is required for Track A models")
    timm_name = TRANSFORMER_CANDIDATES.get(name, name)
    rec = ModelBuildRecord(name=name, timm_name=timm_name, requested_pretrained=pretrained, status="failed")
    if not timm.is_model(timm_name):
        rec.error = f"timm {timm.__version__} does not provide {timm_name!r}"
        return None, rec
    kw = dict(num_classes=0, in_chans=in_chans, drop_path_rate=drop_path_rate, **timm_kwargs)
    model = None
    if pretrained:
        local = _find_local_weights(weights_dir, timm_name)
        if local is not None:
            try:
                model = timm.create_model(timm_name, pretrained=True, pretrained_cfg_overlay={"file": str(local)}, **kw)
                rec.status, rec.source = "pretrained_local_file", str(local)
            except Exception as e:  # noqa: BLE001
                rec.error = f"local weights failed: {type(e).__name__}: {e}"
        if model is None:
            try:
                model = timm.create_model(timm_name, pretrained=True, **kw)
                rec.status, rec.source = "pretrained", "timm/hf-hub download"
            except Exception as e:  # noqa: BLE001
                rec.error = (rec.error + " | " if rec.error else "") + f"download failed: {type(e).__name__}: {str(e)[:200]}"
                if fallback == "random_init":
                    try:
                        model = timm.create_model(timm_name, pretrained=False, **kw)
                        rec.status, rec.source = "random_init_fallback", "random initialisation (weights unavailable)"
                    except Exception as e2:  # noqa: BLE001
                        rec.error += f" | random init failed: {e2}"
                        return None, rec
                else:
                    return None, rec
    else:
        try:
            model = timm.create_model(timm_name, pretrained=False, **kw)
            rec.status, rec.source = "random_init", "requested"
        except Exception as e:  # noqa: BLE001
            rec.error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
            return None, rec
    rec.num_features = int(getattr(model, "num_features", 0))
    rec.n_params, rec.n_trainable = count_params(model)
    return model, rec


class BackboneClassifier(nn.Module):
    """timm backbone (pooled features) -> dropout -> linear 6-label head."""

    def __init__(self, backbone: nn.Module, num_features: int, num_labels: int = NUM_LABELS, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.num_features = num_features
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(num_features, num_labels)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.forward_features(x)))

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return self.head.parameters()


def build_transformer_classifier(name: str, pretrained: bool = True, weights_dir: str | None = None,
                                 fallback: str = "random_init", dropout: float = 0.2, drop_path_rate: float = 0.1,
                                 in_chans: int = 3, num_labels: int = NUM_LABELS) -> tuple[BackboneClassifier | None, ModelBuildRecord]:
    bb, rec = build_backbone(name, pretrained, weights_dir, fallback, in_chans=in_chans, drop_path_rate=drop_path_rate)
    if bb is None:
        return None, rec
    model = BackboneClassifier(bb, rec.num_features, num_labels, dropout)
    rec.n_params, rec.n_trainable = count_params(model)
    return model, rec
