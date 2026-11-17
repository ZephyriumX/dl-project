"""Variational quantum circuit (VQC) layer with two interchangeable exact backends.

Circuit (default configuration = the 4-qubit circuit of Tikariha et al., IEEE Access 2025)::

    [H on all wires]                              (optional, Dong et al. 2023)
    for layer l in 0..L-1:
        if l == 0 or data_reuploading:  RY(theta_i) on wire i      (angle encoding)
        Rot(phi_li, theta_li, omega_li) on every wire               (trainable)
        CNOT ring  (0,1),(1,2),...,(n-1,0)                          (cyclic entanglement; or linear / none)
    measure <Z_i> on every wire

Backends:
* ``"torch"``: :mod:`src.quantum.torch_simulator` - batched, GPU, autograd;
* ``"pennylane"``: a PennyLane QNode (``default.qubit`` or ``lightning.qubit``) with the torch interface,
  used as the reference implementation (equivalence is unit-tested).

The layer always runs in float32 / complex64, outside autocast.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from . import torch_simulator as ts

ENTANGLEMENTS = ("cyclic", "linear", "none")
ENCODINGS = ("ry", "rx")
ROTATIONS = ("rot", "ry")


@dataclass
class VQCConfig:
    n_qubits: int = 4
    n_layers: int = 2
    encoding: str = "ry"
    rotation: str = "rot"
    entanglement: str = "cyclic"
    hadamard_init: bool = False
    data_reuploading: bool = False
    backend: str = "torch"  # "torch" | "pennylane"
    pennylane_device: str = "default.qubit"
    diff_method: str = "backprop"
    init_scale: float = 2 * math.pi  # weights ~ U(0, init_scale)

    def __post_init__(self):
        if self.entanglement not in ENTANGLEMENTS:
            raise ValueError(f"entanglement must be one of {ENTANGLEMENTS}")
        if self.encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}")
        if self.rotation not in ROTATIONS:
            raise ValueError(f"rotation must be one of {ROTATIONS}")
        if self.n_qubits < 1 or self.n_layers < 1:
            raise ValueError("n_qubits and n_layers must be >= 1")
        if self.backend not in ("torch", "pennylane"):
            raise ValueError("backend must be 'torch' or 'pennylane'")

    def as_dict(self) -> dict:
        return asdict(self)


def entangling_pairs(n: int, kind: str) -> list[tuple[int, int]]:
    if kind == "none" or n < 2:
        return []
    if kind == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if n == 2:
        return [(0, 1), (1, 0)]
    return [(i, (i + 1) % n) for i in range(n)]


def circuit_resources(cfg: VQCConfig) -> dict:
    n, L = cfg.n_qubits, cfg.n_layers
    n_enc = n * (L if cfg.data_reuploading else 1)
    n_rot = n * L
    n_cnot = len(entangling_pairs(n, cfg.entanglement)) * L
    params_per_rot = 3 if cfg.rotation == "rot" else 1
    return {
        "n_qubits": n, "n_layers": L, "trainable_params": n * L * params_per_rot,
        "encoding_gates": n_enc, "rotation_gates": n_rot, "cnot_gates": n_cnot,
        "hadamard_gates": n if cfg.hadamard_init else 0, "state_dim": 2 ** n,
        "measurements": n,
    }


class VQCLayer(nn.Module):
    """Maps encoding angles (B, n_qubits) -> Pauli-Z expectations (B, n_qubits) in [-1, 1]."""

    def __init__(self, cfg: VQCConfig | dict | None = None):
        super().__init__()
        self.cfg = cfg if isinstance(cfg, VQCConfig) else VQCConfig(**(cfg or {}))
        n, L = self.cfg.n_qubits, self.cfg.n_layers
        k = 3 if self.cfg.rotation == "rot" else 1
        self.weights = nn.Parameter(torch.rand(L, n, k) * self.cfg.init_scale)
        self.pairs = entangling_pairs(n, self.cfg.entanglement)
        self._qnode = None
        self.sim_seconds = 0.0
        self.sim_calls = 0
        if self.cfg.backend == "pennylane":
            self._qnode = self._build_qnode()

    # ---------------------------------------------------------------- torch backend
    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        cfg, n = self.cfg, self.cfg.n_qubits
        b = x.shape[0]
        state = ts.zero_state(b, n, device=x.device)
        if cfg.hadamard_init:
            for w in range(n):
                state = ts.hadamard(state, w)
        enc = ts.ry_mat if cfg.encoding == "ry" else ts.rx_mat
        for l in range(cfg.n_layers):
            if l == 0 or cfg.data_reuploading:
                for w in range(n):
                    state = ts._apply_1q(state, enc(x[:, w]), w)
            for w in range(n):
                p = self.weights[l, w]
                u = ts.rot_mat(p[0], p[1], p[2]) if cfg.rotation == "rot" else ts.ry_mat(p[0])
                state = ts._apply_1q(state, u, w)
            for c, t in self.pairs:
                state = ts.cnot(state, c, t)
        return ts.expval_z(state)

    # ---------------------------------------------------------------- pennylane backend
    def _build_qnode(self):
        import pennylane as qml

        cfg, n = self.cfg, self.cfg.n_qubits
        dev = qml.device(cfg.pennylane_device, wires=n)
        enc = qml.RY if cfg.encoding == "ry" else qml.RX
        pairs = self.pairs

        @qml.qnode(dev, interface="torch", diff_method=cfg.diff_method)
        def qnode(inputs, weights):
            if cfg.hadamard_init:
                for w in range(n):
                    qml.Hadamard(wires=w)
            for l in range(cfg.n_layers):
                if l == 0 or cfg.data_reuploading:
                    for w in range(n):
                        enc(inputs[..., w], wires=w)
                for w in range(n):
                    if cfg.rotation == "rot":
                        qml.Rot(weights[l, w, 0], weights[l, w, 1], weights[l, w, 2], wires=w)
                    else:
                        qml.RY(weights[l, w, 0], wires=w)
                for c, t in pairs:
                    qml.CNOT(wires=[c, t])
            return [qml.expval(qml.PauliZ(w)) for w in range(n)]

        return qnode

    def _forward_pennylane(self, x: torch.Tensor) -> torch.Tensor:
        out = self._qnode(x.double(), self.weights.double())
        out = torch.stack(list(out), dim=-1) if isinstance(out, (list, tuple)) else out
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return out.to(x.dtype)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if angles.shape[-1] != self.cfg.n_qubits:
            raise ValueError(f"expected {self.cfg.n_qubits} angles, got {angles.shape[-1]}")
        t0 = time.perf_counter()
        dev_type = angles.device.type
        with torch.autocast(device_type=dev_type, enabled=False):
            x = angles.float()
            out = self._forward_torch(x) if self.cfg.backend == "torch" else self._forward_pennylane(x)
        self.sim_seconds += time.perf_counter() - t0
        self.sim_calls += 1
        return out.float()

    def resources(self) -> dict:
        return circuit_resources(self.cfg)

    def draw(self) -> str:
        """Text drawing of the circuit (requires PennyLane)."""
        try:
            import pennylane as qml

            if self._qnode is None:
                self._qnode = self._build_qnode()
            return qml.draw(self._qnode)(torch.zeros(1, self.cfg.n_qubits, dtype=torch.float64),
                                         self.weights.detach().double())
        except Exception as e:  # noqa: BLE001
            return f"(circuit drawing unavailable: {e})"
