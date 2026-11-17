"""ProjectContext: one object that every notebook creates first.

It resolves the project root, loads the YAML configs for the chosen mode, locates the RSNA
data and the writable output root, finds artifacts produced by earlier notebooks (in
``/kaggle/working`` or in attached Kaggle datasets / notebook outputs), and owns the
experiment registry.

Layout of the output root (default ``/kaggle/working/rsna_ich_artifacts``)::

    rsna_ich_artifacts/
      shared/header_cache/           DICOM header scan parts (mode-independent, restartable)
      <mode>/data/                   labels, metadata, split manifest, audit reports
      <mode>/experiments/<exp_id>/   checkpoints, history, predictions, metrics, result.json
      <mode>/analysis/ ...           tables and figures
      <mode>/experiment_registry.jsonl

Modes never share manifests or checkpoints, so a smoke run cannot contaminate a full run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .io import atomic_write_json, read_json
from .paths import KAGGLE_INPUT, RSNAPaths, default_output_root, find_artifact, resolve_rsna_paths
from .registry import ExperimentRegistry
from .runtime import RuntimeInfo, detect_runtime

CONFIG_FILES = {
    "data": "data_config.yaml",
    "transformer": "transformer_config.yaml",
    "hybrid": "hybrid_config.yaml",
    "quantum": "quantum_config.yaml",
}


class ProjectContext:
    def __init__(self, project_root: str | os.PathLike, mode: str = "smoke", overrides: dict[str, dict] | None = None,
                 output_root: str | None = None):
        self.root = Path(project_root).resolve()
        self.mode = mode
        overrides = overrides or {}
        self.cfg: dict[str, dict] = {}
        for key, fname in CONFIG_FILES.items():
            p = self.root / "configs" / fname
            if p.exists():
                self.cfg[key] = load_config(p, mode, overrides.get(key))
        data = self.cfg.get("data", {})
        self.base_root = default_output_root(output_root or os.environ.get("RSNA_OUTPUT_ROOT")
                                             or data.get("paths", {}).get("output_root"))
        self.output_root = self.base_root / mode
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.input_artifact_roots = [Path(p) for p in (data.get("paths", {}).get("artifact_input_roots") or [])]
        self.registry = ExperimentRegistry(self.output_root / "experiment_registry.jsonl")
        self._rsna: RSNAPaths | None = None
        self._runtime: RuntimeInfo | None = None

    # ------------------------------------------------------------------ data / runtime
    @property
    def rsna(self) -> RSNAPaths:
        if self._rsna is None:
            p = self.cfg["data"].get("paths", {})
            self._rsna = resolve_rsna_paths(p.get("rsna_root"), p.get("rsna_root_candidates"))
        return self._rsna

    @property
    def runtime(self) -> RuntimeInfo:
        if self._runtime is None:
            self._runtime = detect_runtime(self.base_root)
        return self._runtime

    # ------------------------------------------------------------------ directories
    def dir(self, *parts: str) -> Path:
        d = self.output_root.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def shared_dir(self, *parts: str) -> Path:
        d = self.base_root.joinpath("shared", *parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def find(self, relative: str, required: bool = True) -> Path | None:
        """Find ``<mode>/<relative>`` in the output root, configured roots, or attached Kaggle inputs."""
        p = find_artifact(f"{self.mode}/{relative}", self.base_root, self.input_artifact_roots)
        if p is None and required:
            raise FileNotFoundError(
                f"Artifact '{self.mode}/{relative}' not found in {self.base_root} or attached inputs. Run the producing "
                "notebook first (same MODE), or attach its output (Kaggle: 'Add Input' -> Your Work -> Notebooks) and, "
                "if it is not found automatically, add its mount point to data_config.yaml paths.artifact_input_roots.")
        return p

    def manifest(self):
        from ..data.splitting import load_manifest

        csv = self.find("data/split_manifest.csv")
        return load_manifest(csv, csv.with_suffix(".json"), verify=True)

    # ------------------------------------------------------------------ experiments across sessions
    def _experiment_roots(self) -> list[Path]:
        roots = [self.output_root / "experiments"]
        for r in self.input_artifact_roots:
            roots.append(r / self.mode / "experiments")
        if KAGGLE_INPUT.exists():
            stack = [(KAGGLE_INPUT, 0)]
            while stack:
                d, depth = stack.pop()
                cand = d / self.mode / "experiments"
                if cand.is_dir():
                    roots.append(cand)
                if depth < 4:
                    try:
                        stack += [(p, depth + 1) for p in d.iterdir() if p.is_dir() and not p.name.startswith("stage_2_")]
                    except OSError:
                        pass
        seen, out = set(), []
        for r in roots:
            if r.is_dir() and r.resolve() not in seen:
                seen.add(r.resolve())
                out.append(r)
        return out

    def experiment_dirs(self) -> dict[str, Path]:
        """exp_id -> directory, preferring this session's working directory over attached inputs."""
        found: dict[str, Path] = {}
        for root in self._experiment_roots():
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / "result.json").exists() and d.name not in found:
                    found[d.name] = d
        return found

    def results(self, prefix: str | tuple[str, ...] | None = None) -> pd.DataFrame:
        rows = []
        for exp_id, d in self.experiment_dirs().items():
            if prefix and not exp_id.startswith(prefix):
                continue
            try:
                r = read_json(d / "result.json")
            except Exception:  # noqa: BLE001
                continue
            r["_dir"] = str(d)
            rows.append(r)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ resume across Kaggle sessions
    def restore_experiments(self, prefixes: tuple[str, ...] = ("A_", "B_", "C_", "H_"), include_incomplete: bool = True) -> list[str]:
        """Copy experiment folders found in attached inputs (a previous notebook version's output) into the
        writable output root, so interrupted training resumes from ``last.pt`` instead of restarting.
        Folders that already exist in the working directory are never overwritten."""
        import shutil

        restored = []
        work_root = self.output_root / "experiments"
        for root in self._experiment_roots():
            if root.resolve() == work_root.resolve():
                continue
            for d in root.iterdir():
                if not d.is_dir() or not d.name.startswith(prefixes) or (work_root / d.name).exists():
                    continue
                if not ((d / "result.json").exists() or (include_incomplete and (d / "last.pt").exists())):
                    continue
                shutil.copytree(d, work_root / d.name)
                restored.append(d.name)
        if restored:
            print(f"Restored {len(restored)} experiment folder(s) from attached inputs: {restored}")
        return restored

    def restore_shared_cache(self) -> int:
        """Copy header-scan parts from an attached previous output (``shared/header_cache``) to avoid rescanning."""
        import shutil

        dst = self.shared_dir("header_cache", "header_parts")
        n = 0
        if not KAGGLE_INPUT.exists() and not self.input_artifact_roots:
            return 0
        cands = [r / "shared" / "header_cache" / "header_parts" for r in self.input_artifact_roots]
        if KAGGLE_INPUT.exists():
            cands += list(KAGGLE_INPUT.glob("*/shared/header_cache/header_parts"))
            cands += list(KAGGLE_INPUT.glob("*/*/shared/header_cache/header_parts"))
        for src_dir in cands:
            if src_dir.is_dir() and src_dir.resolve() != dst.resolve():
                for p in src_dir.glob("part_*.parquet"):
                    target = dst / f"part_r_{src_dir.parent.parent.parent.name}_{p.name}"
                    if not target.exists():
                        shutil.copy2(p, target)
                        n += 1
        if n:
            print(f"Restored {n} header-scan part file(s) from attached inputs.")
        return n

    def save_run_info(self, name: str, extra: dict[str, Any] | None = None) -> Path:
        info = {"mode": self.mode, "project_root": str(self.root), "output_root": str(self.output_root),
                "runtime": self.runtime.as_dict(), "configs": self.cfg, **(extra or {})}
        return atomic_write_json(self.dir("run_info") / f"{name}.json", info)

    def describe(self) -> str:
        return (f"ProjectContext(mode={self.mode}, root={self.root}, output={self.output_root})\n"
                f"  {self.runtime.summary()}")
