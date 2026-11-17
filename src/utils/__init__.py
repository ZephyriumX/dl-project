"""Cross-cutting utilities: configuration, dataset discovery, runtime detection, reproducibility, I/O."""

from .config import load_config, deep_merge, config_fingerprint
from .io import atomic_write_json, atomic_write_text, read_json, sha256_file, atomic_save_dataframe
from .repro import set_seed, capture_rng_state, restore_rng_state

__all__ = [
    "load_config",
    "deep_merge",
    "config_fingerprint",
    "atomic_write_json",
    "atomic_write_text",
    "read_json",
    "sha256_file",
    "atomic_save_dataframe",
    "set_seed",
    "capture_rng_state",
    "restore_rng_state",
]
