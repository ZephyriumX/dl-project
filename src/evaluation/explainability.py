"""Grad-CAM style explanation maps for CNN, Transformer and hybrid models.

These maps are **qualitative, model-derived saliency**. They are not segmentations, they are not
validated against pixel annotations (RSNA provides none), and no localisation accuracy can be
claimed from them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, layout: str = "auto", num_prefix_tokens: int = 0):
        self.model, self.layer, self.layout, self.prefix = model, target_layer, layout, num_prefix_tokens
        self._act, self._grad = None, None
        self._h1 = target_layer.register_forward_hook(self._fwd)

    def _fwd(self, m, i, o):
        o = o[-1] if isinstance(o, (list, tuple)) else o
        self._act = o
        if o.requires_grad:
            o.register_hook(lambda g: setattr(self, "_grad", g))

    def remove(self):
        self._h1.remove()

    def _to_nchw(self, t: torch.Tensor) -> torch.Tensor:
        lay = self.layout
        if lay == "auto":
            lay = "tokens" if t.dim() == 3 else "nchw"
        if lay == "tokens":
            t = t[:, self.prefix:, :]
            n = t.shape[1]
            s = int(round(n ** 0.5))
            if s * s != n:
                raise ValueError(f"cannot reshape {n} tokens to a square grid")
            return t.transpose(1, 2).reshape(t.shape[0], t.shape[2], s, s)
        if lay == "nhwc":
            return t.permute(0, 3, 1, 2)
        return t

    def __call__(self, x: torch.Tensor, label_index: int) -> torch.Tensor:
        self.model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.model(x)
            self.model.zero_grad(set_to_none=True)
            logits[:, label_index].sum().backward()
        act, grad = self._to_nchw(self._act.detach()), self._to_nchw(self._grad.detach())
        w = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        mn = cam.flatten(1).min(1)[0][:, None, None]
        mx = cam.flatten(1).max(1)[0][:, None, None]
        return ((cam - mn) / (mx - mn + 1e-8)).cpu()


def default_cam_target(model: nn.Module) -> tuple[nn.Module, str, int]:
    """Pick a sensible target layer: returns (layer, layout, num_prefix_tokens)."""
    bb = getattr(model, "backbone", None)
    if hasattr(model, "proj") and hasattr(model, "cnn"):  # CNNTransformerHybrid
        return model.proj, "nchw", 0
    if bb is not None:
        if hasattr(bb, "blocks") and hasattr(bb, "num_prefix_tokens"):  # ViT / DeiT / BEiT
            return bb.blocks[-1], "tokens", int(bb.num_prefix_tokens)
        if hasattr(bb, "layers") and "swin" in type(bb).__name__.lower():
            return bb.layers[-1], "nhwc", 0
        if hasattr(bb, "stages"):
            return bb.stages[-1], "nchw", 0
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("no suitable Grad-CAM target layer found")
    return convs[-1], "nchw", 0
