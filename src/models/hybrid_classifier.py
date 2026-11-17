"""Feature-processing heads on top of a backbone, with matched classical controls for the VQC.

All heads map a backbone feature vector f in R^d to six logits. The bottleneck heads share the
same encoder and readout so that **only the processing block differs**:

    encoder : Linear(d -> k) -> [LayerNorm(k)] -> angle scaling (pi * tanh)   (shared)
    block   : identity | Linear(k->k)+tanh | VQC(k qubits)                       (varies)
    readout : Linear(k -> h) -> SiLU -> Linear(h -> 6)                           (shared)

Head kinds:
    ``linear_probe``         Linear(d -> 6)                          (reference, no bottleneck)
    ``mlp``                  Linear(d->h) -> SiLU -> Dropout -> Linear(h->6)
    ``bottleneck_identity``  encoder -> readout                      (control: effect of the k-dim bottleneck)
    ``bottleneck_mlp``       encoder -> Linear(k,k)+tanh -> readout  (control: parameter-matched classical block)
    ``vqc``                  encoder -> VQC -> readout               (quantum-classical hybrid)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .. import NUM_LABELS

HEAD_KINDS = ("linear_probe", "mlp", "bottleneck_identity", "bottleneck_mlp", "vqc")


class AngleEncoder(nn.Module):
    """Linear projection to k features, optional LayerNorm, then scaling to rotation angles."""

    def __init__(self, in_dim: int, k: int, norm: str = "layernorm", scaling: str = "pi_tanh"):
        super().__init__()
        self.proj = nn.Linear(in_dim, k)
        self.norm = nn.LayerNorm(k) if norm == "layernorm" else (nn.BatchNorm1d(k) if norm == "batchnorm" else nn.Identity())
        if scaling not in ("pi_tanh", "pi_sigmoid", "tanh", "none"):
            raise ValueError(scaling)
        self.scaling = scaling

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        z = self.norm(self.proj(f))
        if self.scaling == "pi_tanh":
            return math.pi * torch.tanh(z)  # angles in (-pi, pi)
        if self.scaling == "pi_sigmoid":
            return math.pi * torch.sigmoid(z)  # angles in (0, pi)  (theta = pi * f, Paper 2)
        if self.scaling == "tanh":
            return torch.tanh(z)
        return z


class Readout(nn.Module):
    def __init__(self, k: int, hidden: int = 16, num_labels: int = NUM_LABELS, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, num_labels))

    def forward(self, z):
        return self.net(z)


class ClassicalBlock(nn.Module):
    """Parameter-matched classical replacement of the VQC: Linear(k,k) -> tanh (outputs in [-1,1] like <Z>)."""

    def __init__(self, k: int, depth: int = 1):
        super().__init__()
        self.net = nn.Sequential(*[m for _ in range(depth) for m in (nn.Linear(k, k), nn.Tanh())])

    def forward(self, a):
        return self.net(a)


class FeatureHead(nn.Module):
    def __init__(self, kind: str, in_dim: int, k: int = 4, hidden: int = 16, num_labels: int = NUM_LABELS,
                 dropout: float = 0.0, norm: str = "layernorm", scaling: str = "pi_tanh", vqc: dict | None = None,
                 classical_depth: int = 1, mlp_hidden: int = 256):
        super().__init__()
        if kind not in HEAD_KINDS:
            raise ValueError(f"kind must be one of {HEAD_KINDS}")
        self.kind, self.k = kind, k
        self.block_name = kind
        if kind == "linear_probe":
            self.net = nn.Linear(in_dim, num_labels)
            return
        if kind == "mlp":
            self.net = nn.Sequential(nn.Linear(in_dim, mlp_hidden), nn.SiLU(), nn.Dropout(dropout),
                                     nn.Linear(mlp_hidden, num_labels))
            return
        self.encoder = AngleEncoder(in_dim, k, norm, scaling)
        if kind == "bottleneck_identity":
            self.block = nn.Identity()
        elif kind == "bottleneck_mlp":
            self.block = ClassicalBlock(k, classical_depth)
        else:
            from ..quantum.vqc import VQCConfig, VQCLayer

            vcfg = dict(vqc or {})
            vcfg["n_qubits"] = k
            self.block = VQCLayer(VQCConfig(**vcfg))
        self.readout = Readout(k, hidden, num_labels, dropout)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        if self.kind in ("linear_probe", "mlp"):
            return self.net(f)
        return self.readout(self.block(self.encoder(f)))

    def intermediate(self, f: torch.Tensor) -> dict[str, torch.Tensor]:
        """Angles and processed features (for gradient-flow / saturation analysis)."""
        if self.kind in ("linear_probe", "mlp"):
            return {}
        a = self.encoder(f)
        return {"angles": a, "processed": self.block(a)}

    def param_breakdown(self) -> dict[str, int]:
        out = {"total": sum(p.numel() for p in self.parameters())}
        for name in ("encoder", "block", "readout", "net"):
            if hasattr(self, name):
                out[name] = sum(p.numel() for p in getattr(self, name).parameters())
        return out


class BackboneWithHead(nn.Module):
    """End-to-end model: any backbone exposing ``forward_features`` + a FeatureHead."""

    def __init__(self, backbone: nn.Module, head: FeatureHead, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # keep BN/dropout statistics fixed
        return self

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                f = self.backbone.forward_features(x)
        else:
            f = self.backbone.forward_features(x)
        return self.head(f.float())

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return self.head.parameters()
