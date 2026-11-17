"""Exact, batched, differentiable state-vector simulation in PyTorch.

Used as the fast backend of the VQC. For n qubits the state has 2**n complex amplitudes per
sample; every gate is a tensor contraction over one (or two) qubit axes, so the cost per gate is
O(B * 2**n) and runs on GPU. Gradients come from autograd (equivalent to PennyLane's
``diff_method="backprop"`` on ``default.qubit``).

Conventions follow PennyLane exactly (verified in tests/test_models.py):

* ``RX(t) = [[c, -i s], [-i s, c]]``, ``RY(t) = [[c, -s], [s, c]]``, ``RZ(t) = diag(e^{-it/2}, e^{it/2})``
  with ``c = cos(t/2)``, ``s = sin(t/2)``;
* ``Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)``;
* ``CNOT(control, target)``; Pauli-Z expectation per wire.

This is a classical simulation of an ideal noise-free circuit. It gives no quantum speed-up.
"""

from __future__ import annotations

import math

import torch

CDTYPE = torch.complex64


def zero_state(batch: int, n: int, device=None) -> torch.Tensor:
    s = torch.zeros((batch,) + (2,) * n, dtype=CDTYPE, device=device)
    s.view(batch, -1)[:, 0] = 1.0
    return s


def _apply_1q(state: torch.Tensor, u: torch.Tensor, wire: int) -> torch.Tensor:
    """Apply a single-qubit gate. ``u``: (2,2) shared or (B,2,2) per-sample."""
    ax = wire + 1
    s = state.movedim(ax, -1)
    if u.dim() == 2:
        out = torch.einsum("ij,...j->...i", u, s)
    else:
        b = s.shape[0]
        out = torch.einsum("bij,bkj->bki", u, s.reshape(b, -1, 2)).reshape(s.shape)
    return out.movedim(-1, ax)


def cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
    if control == target:
        raise ValueError("control and target must differ")
    s = state.movedim((control + 1, target + 1), (-2, -1))
    out = torch.stack([s[..., 0, :], s[..., 1, :].flip(-1)], dim=-2)
    return out.movedim((-2, -1), (control + 1, target + 1))


def _cs(theta: torch.Tensor):
    return torch.cos(theta / 2), torch.sin(theta / 2)


def rx_mat(theta: torch.Tensor) -> torch.Tensor:
    c, s = _cs(theta)
    c, s = c.to(CDTYPE), s.to(CDTYPE)
    m = torch.stack([torch.stack([c, -1j * s], -1), torch.stack([-1j * s, c], -1)], -2)
    return m


def ry_mat(theta: torch.Tensor) -> torch.Tensor:
    c, s = _cs(theta)
    c, s = c.to(CDTYPE), s.to(CDTYPE)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def rz_mat(theta: torch.Tensor) -> torch.Tensor:
    e_m = torch.exp(-0.5j * theta.to(CDTYPE))
    e_p = torch.exp(0.5j * theta.to(CDTYPE))
    z = torch.zeros_like(e_m)
    return torch.stack([torch.stack([e_m, z], -1), torch.stack([z, e_p], -1)], -2)


def rot_mat(phi: torch.Tensor, theta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    return rz_mat(omega) @ ry_mat(theta) @ rz_mat(phi)


H_MAT = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=CDTYPE) / math.sqrt(2.0)


def hadamard(state: torch.Tensor, wire: int) -> torch.Tensor:
    return _apply_1q(state, H_MAT.to(state.device), wire)


def expval_z(state: torch.Tensor) -> torch.Tensor:
    """Pauli-Z expectation for every wire -> (B, n) real."""
    p = (state.real ** 2 + state.imag ** 2)
    n = state.dim() - 1
    out = []
    for w in range(n):
        dims = [d for d in range(1, n + 1) if d != w + 1]
        pw = p.sum(dim=dims) if dims else p
        out.append(pw[:, 0] - pw[:, 1])
    return torch.stack(out, dim=1)
