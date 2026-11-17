"""Quantum-classical hybrid assembly, cost accounting and gradient diagnostics.

The hybrid is a "dressed quantum circuit" (Mari et al.; used by Dong et al. 2023):
``readout ∘ VQC ∘ encoder`` on backbone features. The concrete module is
:class:`src.models.hybrid_classifier.FeatureHead` with ``kind="vqc"``. This module adds helpers
specific to the quantum experiments.
"""

from __future__ import annotations

import time
from typing import Iterable

import torch
import torch.nn as nn

from ..models.hybrid_classifier import BackboneWithHead, FeatureHead
from .vqc import VQCConfig, VQCLayer, circuit_resources


def build_quantum_head(in_dim: int, n_qubits: int = 4, n_layers: int = 2, entanglement: str = "cyclic",
                       backend: str = "torch", hidden: int = 16, dropout: float = 0.0, norm: str = "layernorm",
                       scaling: str = "pi_tanh", **vqc_kwargs) -> FeatureHead:
    vqc = dict(n_layers=n_layers, entanglement=entanglement, backend=backend, **vqc_kwargs)
    return FeatureHead("vqc", in_dim=in_dim, k=n_qubits, hidden=hidden, dropout=dropout, norm=norm,
                       scaling=scaling, vqc=vqc)


def build_quantum_hybrid(backbone: nn.Module, in_dim: int, freeze_backbone: bool = True, **kw) -> BackboneWithHead:
    return BackboneWithHead(backbone, build_quantum_head(in_dim, **kw), freeze_backbone=freeze_backbone)


def find_vqc(module: nn.Module) -> VQCLayer | None:
    for m in module.modules():
        if isinstance(m, VQCLayer):
            return m
    return None


def gradient_norms(model: nn.Module) -> dict[str, float]:
    """L2 norm of gradients per head component (call after backward)."""
    out = {}
    for name, sub in model.named_children():
        g = [p.grad.detach().flatten() for p in sub.parameters() if p.grad is not None]
        out[name] = float(torch.cat(g).norm()) if g else float("nan")
    return out


@torch.no_grad()
def angle_saturation(head: FeatureHead, feats: torch.Tensor, thresh: float = 0.95) -> dict[str, float]:
    """Fraction of encoding angles close to +/-pi (tanh saturation hurts trainability)."""
    if head.kind not in ("vqc", "bottleneck_mlp", "bottleneck_identity"):
        return {}
    a = head.encoder(feats)
    rel = (a.abs() / torch.pi)
    return {"mean_abs_angle_over_pi": float(rel.mean()), "frac_saturated": float((rel > thresh).float().mean())}


def benchmark_simulator(cfg: VQCConfig | dict, batch_sizes: Iterable[int] = (64, 256), device: str | None = None,
                        repeats: int = 5) -> list[dict]:
    """Forward+backward wall-clock per batch for the configured backend (simulator cost)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    layer = VQCLayer(cfg if isinstance(cfg, VQCConfig) else VQCConfig(**cfg))
    if layer.cfg.backend == "torch":
        layer = layer.to(device)
    else:
        device = "cpu"
    rows = []
    for b in batch_sizes:
        x = torch.randn(b, layer.cfg.n_qubits, device=device, requires_grad=True)
        layer(x).sum().backward()  # warm-up
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            layer(x).sum().backward()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / repeats
        rows.append({"backend": layer.cfg.backend, "device": device, "batch": b, "fwd_bwd_ms": dt * 1e3,
                     "us_per_sample": dt / b * 1e6, **circuit_resources(layer.cfg)})
    return rows
