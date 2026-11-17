"""Dataset discovery, path validation and artifact locations (Kaggle-aware).

Nothing here hard-codes a single dataset path. Resolution order for the RSNA root:

1. explicit argument / config ``data.rsna_root``
2. environment variable ``RSNA_DATA_ROOT``
3. configured candidate list (``data.rsna_root_candidates``)
4. a bounded search under ``/kaggle/input`` for a directory containing ``stage_2_train.csv``
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

REQUIRED_COMPONENTS = {
    "train_dir": ("stage_2_train", "dir"),
    "test_dir": ("stage_2_test", "dir"),
    "train_csv": ("stage_2_train.csv", "file"),
    "sample_submission": ("stage_2_sample_submission.csv", "file"),
}

DEFAULT_CANDIDATES = (
    "/kaggle/input/rsna-intracranial-hemorrhage-detection",
    "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection",
    "/kaggle/input/competitions/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection",
    "/kaggle/input/competitions/rsna-intracranial-hemorrhage-detection",
)


class DatasetNotFoundError(FileNotFoundError):
    pass


@dataclass
class RSNAPaths:
    root: Path
    train_dir: Path
    test_dir: Path
    train_csv: Path
    sample_submission: Path
    missing: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing

    def as_dict(self) -> dict:
        return {k: str(v) for k, v in self.__dict__.items() if k != "missing"} | {"missing": list(self.missing)}


def on_kaggle() -> bool:
    return KAGGLE_INPUT.exists() or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def _has_train_csv(p: Path) -> bool:
    return (p / "stage_2_train.csv").is_file()


def _bounded_search(base: Path, max_depth: int = 4) -> list[Path]:
    hits: list[Path] = []
    if not base.exists():
        return hits
    stack = [(base, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            if _has_train_csv(d):
                hits.append(d)
                continue
            if depth >= max_depth:
                continue
            with os.scandir(d) as it:
                for e in it:
                    # never descend into the DICOM folders themselves (hundreds of thousands of files)
                    if e.is_dir(follow_symlinks=False) and not e.name.startswith("stage_2_"):
                        stack.append((Path(e.path), depth + 1))
        except (PermissionError, OSError):
            continue
    return sorted(hits)


def discover_rsna_root(
    explicit: str | os.PathLike | None = None,
    candidates: Sequence[str] | None = None,
    search_kaggle_input: bool = True,
) -> Path:
    tried: list[str] = []
    for c in [explicit, os.environ.get("RSNA_DATA_ROOT"), *(candidates or DEFAULT_CANDIDATES)]:
        if not c:
            continue
        p = Path(c).expanduser()
        tried.append(str(p))
        if _has_train_csv(p):
            return p.resolve()
    if search_kaggle_input:
        hits = _bounded_search(KAGGLE_INPUT)
        if hits:
            return hits[0].resolve()
        tried.append(f"search under {KAGGLE_INPUT} (depth<=4)")
    raise DatasetNotFoundError(
        "RSNA ICH dataset not found. Attach the competition data "
        "('rsna-intracranial-hemorrhage-detection') to the notebook, or set data.rsna_root in "
        "configs/data_config.yaml, or set the RSNA_DATA_ROOT environment variable.\nLocations tried:\n  - "
        + "\n  - ".join(tried)
    )


def resolve_rsna_paths(root: str | os.PathLike | None = None, candidates: Sequence[str] | None = None,
                       require: Iterable[str] = ("train_dir", "train_csv")) -> RSNAPaths:
    """Locate and validate the four RSNA components. Raises if a *required* component is missing."""
    r = discover_rsna_root(root, candidates)
    resolved, missing = {}, []
    for key, (name, kind) in REQUIRED_COMPONENTS.items():
        p = r / name
        ok = p.is_dir() if kind == "dir" else p.is_file()
        resolved[key] = p
        if not ok:
            missing.append(key)
    paths = RSNAPaths(root=r, missing=missing, **resolved)
    hard = [m for m in missing if m in set(require)]
    if hard:
        raise DatasetNotFoundError(f"RSNA root {r} found but required components are missing: {hard}")
    return paths


def count_files(directory: Path, suffix: str = ".dcm", limit: int | None = None) -> int:
    """Count files with ``suffix`` (stops at ``limit`` if given, to keep it cheap)."""
    n = 0
    with os.scandir(directory) as it:
        for e in it:
            if e.name.endswith(suffix):
                n += 1
                if limit and n >= limit:
                    break
    return n


def default_output_root(configured: str | None = None) -> Path:
    if configured:
        p = Path(configured)
    elif KAGGLE_WORKING.exists():
        p = KAGGLE_WORKING / "rsna_ich_artifacts"
    else:
        p = Path.cwd() / "rsna_ich_artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_artifact(relative: str, output_root: Path, extra_roots: Sequence[str | os.PathLike] = (),
                  search_kaggle_input: bool = True) -> Path | None:
    """Find a previously produced artifact (e.g. ``data/split_manifest.csv``).

    Looks in the current output root first, then in explicitly configured roots, then in any
    attached Kaggle dataset/notebook output under ``/kaggle/input`` (bounded depth).
    """
    roots = [Path(output_root), *[Path(r) for r in extra_roots]]
    for r in roots:
        p = r / relative
        if p.exists():
            return p
    if search_kaggle_input and KAGGLE_INPUT.exists():
        stack = [(KAGGLE_INPUT, 0)]
        while stack:
            d, depth = stack.pop()
            cand = d / relative
            if cand.exists():
                return cand
            if depth >= 4:
                continue
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if e.is_dir(follow_symlinks=False) and not e.name.startswith("stage_2_"):
                            stack.append((Path(e.path), depth + 1))
            except OSError:
                continue
    return None
