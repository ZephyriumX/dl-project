"""Track B: CNN-Transformer hybrid for six-label ICH classification.

Design (see research_documentation/architecture_notes.md, D6):

* a pretrained CNN stem (timm ``features_only``; default EfficientNet-B0, stride-16 stage) produces a
  14x14 grid of local features for a 224 px input - convolutions are well suited to small, local
  hyperdensities (acute blood);
* a 1x1 projection turns the grid into tokens; a learnable 2-D positional embedding and a [CLS] token
  are added; ``depth`` pre-LN Transformer encoder blocks let every location attend to the whole slice
  (hemispheric comparison, ventricles vs parenchyma, thin rims vs calvarium);
* fusion: ``concat`` of the CNN global-average-pooled vector (local evidence) and the [CLS] token
  (context) -> LayerNorm -> MLP -> 6 logits. ``fusion`` can be ``cls_only`` / ``gap_only`` for ablation.

The last block's attention weights can be returned for visualisation. They are attention weights,
not segmentations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import NUM_LABELS
from .transformers import ModelBuildRecord, count_params

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


class EncoderBlock(nn.Module):
    """Pre-LN Transformer block that can return its attention map."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        h = self.n1(x)
        a, w = self.attn(h, h, h, need_weights=need_weights, average_attn_weights=True)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x, w


class CNNTransformerHybrid(nn.Module):
    def __init__(self, cnn_name: str = "efficientnet_b0", pretrained: bool = True, out_index: int = 3,
                 d_model: int = 256, depth: int = 2, heads: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.1,
                 fusion: str = "concat", num_labels: int = NUM_LABELS, in_chans: int = 3, grid_size: int = 14,
                 weights_dir: str | None = None, fallback: str = "random_init"):
        super().__init__()
        if timm is None:
            raise ImportError("timm required")
        if fusion not in ("concat", "cls_only", "gap_only"):
            raise ValueError(fusion)
        self.fusion = fusion
        self.record = ModelBuildRecord(name="cnn_transformer_hybrid", timm_name=cnn_name,
                                       requested_pretrained=pretrained, status="failed")
        self.cnn = self._make_cnn(cnn_name, pretrained, out_index, in_chans, weights_dir, fallback)
        c = self.cnn.feature_info.channels()[-1]
        self.cnn_channels = c
        self.proj = nn.Sequential(nn.Conv2d(c, d_model, 1, bias=False), nn.BatchNorm2d(d_model), nn.GELU())
        self.grid_size = grid_size
        self.pos = nn.Parameter(torch.zeros(1, d_model, grid_size, grid_size))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        fused = {"concat": c + d_model, "cls_only": d_model, "gap_only": c}[fusion]
        self.head = nn.Sequential(nn.LayerNorm(fused), nn.Dropout(dropout), nn.Linear(fused, 256), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(256, num_labels))
        self.num_features = fused
        self.record.num_features = fused
        self.record.n_params, self.record.n_trainable = count_params(self)

    def _make_cnn(self, name, pretrained, out_index, in_chans, weights_dir, fallback):
        from .transformers import _find_local_weights

        kw = dict(features_only=True, out_indices=(out_index,), in_chans=in_chans)
        if pretrained:
            local = _find_local_weights(weights_dir, name)
            try:
                if local is not None:
                    m = timm.create_model(name, pretrained=True, pretrained_cfg_overlay={"file": str(local)}, **kw)
                    self.record.status, self.record.source = "pretrained_local_file", str(local)
                else:
                    m = timm.create_model(name, pretrained=True, **kw)
                    self.record.status, self.record.source = "pretrained", "timm/hf-hub download"
                return m
            except Exception as e:  # noqa: BLE001
                self.record.error = f"{type(e).__name__}: {str(e)[:200]}"
                if fallback != "random_init":
                    raise
                self.record.status, self.record.source = "random_init_fallback", "weights unavailable"
                return timm.create_model(name, pretrained=False, **kw)
        self.record.status, self.record.source = "random_init", "requested"
        return timm.create_model(name, pretrained=False, **kw)

    def _pos_embed(self, h: int, w: int) -> torch.Tensor:
        if (h, w) == (self.grid_size, self.grid_size):
            return self.pos
        return F.interpolate(self.pos, size=(h, w), mode="bicubic", align_corners=False)

    def forward_features(self, x: torch.Tensor, return_attention: bool = False):
        fmap = self.cnn(x)[-1]  # (B, C, h, w)
        gap = fmap.mean(dim=(2, 3))
        if self.fusion == "gap_only":
            return (gap, None) if return_attention else gap
        t = self.proj(fmap)
        b, d, h, w = t.shape
        t = (t + self._pos_embed(h, w)).flatten(2).transpose(1, 2)  # (B, hw, d)
        t = torch.cat([self.cls.expand(b, -1, -1), t], dim=1)
        attn = None
        for i, blk in enumerate(self.blocks):
            t, a = blk(t, need_weights=return_attention and i == len(self.blocks) - 1)
            if a is not None:
                attn = a
        cls = self.norm(t[:, 0])
        feats = torch.cat([gap, cls], dim=1) if self.fusion == "concat" else cls
        if return_attention:
            cls_attn = attn[:, 0, 1:].reshape(b, h, w) if attn is not None else None
            return feats, cls_attn
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def backbone_parameters(self):
        return self.cnn.parameters()

    def head_parameters(self):
        ids = {id(p) for p in self.cnn.parameters()}
        return (p for p in self.parameters() if id(p) not in ids)


def build_cnn_transformer(cfg: dict) -> tuple[CNNTransformerHybrid, ModelBuildRecord]:
    keys = ("cnn_name", "pretrained", "out_index", "d_model", "depth", "heads", "mlp_ratio", "dropout", "fusion",
            "in_chans", "grid_size", "weights_dir", "fallback")
    m = CNNTransformerHybrid(**{k: cfg[k] for k in keys if k in cfg})
    return m, m.record


def expected_grid(image_size: int, stride: int = 16) -> int:
    return int(math.ceil(image_size / stride))
