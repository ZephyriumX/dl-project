"""YAML configuration loading with execution modes (smoke / pilot / full).

Each config file has a ``base`` section and a ``modes`` section::

    base: {...}
    modes:
      smoke: {... overrides ...}
      pilot: {...}
      full:  {...}

``load_config(path, mode, overrides)`` returns ``deep_merge(base, modes[mode], overrides)``.
Changing a setting never requires editing source code: edit the YAML or pass ``overrides``
from a notebook cell.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

VALID_MODES = ("smoke", "pilot", "full")


def deep_merge(base: Mapping[str, Any], *updates: Mapping[str, Any] | None) -> dict:
    """Recursively merge mappings; later updates win. Lists are replaced, not concatenated."""
    out = copy.deepcopy(dict(base))
    for upd in updates:
        if not upd:
            continue
        for k, v in upd.items():
            if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping at top level")
    return data


def load_config(path: str | Path, mode: str = "smoke", overrides: Mapping[str, Any] | None = None) -> dict:
    """Load ``path`` and resolve the given execution mode."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    raw = load_yaml(path)
    base = raw.get("base", {k: v for k, v in raw.items() if k != "modes"})
    mode_cfg = (raw.get("modes") or {}).get(mode, {})
    cfg = deep_merge(base, mode_cfg, overrides)
    cfg["_mode"] = mode
    cfg["_source"] = str(Path(path).name)
    return cfg


def config_fingerprint(cfg: Mapping[str, Any], ignore_prefix: str = "_") -> str:
    """Stable short hash of a config (keys starting with ``ignore_prefix`` are excluded)."""

    def strip(o):
        if isinstance(o, Mapping):
            return {k: strip(v) for k, v in sorted(o.items()) if not str(k).startswith(ignore_prefix)}
        if isinstance(o, (list, tuple)):
            return [strip(v) for v in o]
        return o

    blob = json.dumps(strip(cfg), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
