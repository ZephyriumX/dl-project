"""Quantum-classical components: VQC layer (torch + PennyLane backends) and hybrid helpers."""

from .quantum_hybrid import (angle_saturation, benchmark_simulator, build_quantum_head, build_quantum_hybrid,
                             find_vqc, gradient_norms)
from .vqc import VQCConfig, VQCLayer, circuit_resources, entangling_pairs

__all__ = [n for n in dir() if not n.startswith("_")]
