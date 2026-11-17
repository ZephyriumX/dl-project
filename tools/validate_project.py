"""Static validation of the project (and optionally of the delivery ZIP).

    python tools/validate_project.py [--zip ../RSNA_ICH_Research_Project.zip] [--report out.json]

Checks: required files exist; every .py file compiles; every src module imports; every YAML config
resolves in every mode; every notebook is valid nbformat JSON, has no stored outputs, contains the
shared setup cell and imports ``src``; the ZIP (if given) contains every required file and no
datasets/checkpoints.
"""

from __future__ import annotations

import argparse
import importlib
import json
import py_compile
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED = [
    "README.md", "IMPLEMENTATION_REPORT.md", "requirements.txt", "environment.yml",
    "configs/data_config.yaml", "configs/transformer_config.yaml", "configs/hybrid_config.yaml", "configs/quantum_config.yaml",
    "src/__init__.py",
    "src/data/__init__.py", "src/data/csv_parser.py", "src/data/dicom_metadata.py", "src/data/dataset.py",
    "src/data/splitting.py", "src/data/preprocessing.py",
    "src/models/__init__.py", "src/models/transformers.py", "src/models/cnn_transformer.py", "src/models/hybrid_classifier.py",
    "src/quantum/__init__.py", "src/quantum/vqc.py", "src/quantum/quantum_hybrid.py", "src/quantum/torch_simulator.py",
    "src/training/__init__.py", "src/training/trainer.py", "src/training/losses.py", "src/training/checkpointing.py",
    "src/training/experiment.py",
    "src/evaluation/__init__.py", "src/evaluation/metrics.py", "src/evaluation/thresholds.py", "src/evaluation/statistics.py",
    "src/evaluation/visualization.py", "src/evaluation/study_aggregation.py", "src/evaluation/explainability.py",
    "src/evaluation/predictions.py", "src/evaluation/submission.py",
    "src/utils/__init__.py", "src/utils/config.py", "src/utils/paths.py", "src/utils/runtime.py", "src/utils/context.py",
    "src/utils/registry.py", "src/utils/repro.py", "src/utils/io.py",
    "notebooks/00_Data_Audit_and_Preprocessing.ipynb", "notebooks/01_Classical_Transformer_Benchmark.ipynb",
    "notebooks/02_CNN_Transformer_Hybrid.ipynb", "notebooks/03_Quantum_Classical_Hybrid.ipynb",
    "notebooks/04_Ablation_and_Comparative_Analysis.ipynb", "notebooks/05_ICH_Triage_and_Visualization.ipynb",
    "tests/test_data_pipeline.py", "tests/test_models.py", "tests/test_metrics.py", "tests/conftest.py",
    "artifacts/.gitkeep",
    "research_documentation/existing_work_audit.md", "research_documentation/experiment_log.md",
    "research_documentation/architecture_notes.md", "research_documentation/limitations.md",
    "research_documentation/progress.md", "research_documentation/implementation_status.md",
    "tools/build_notebooks.py", "tools/nb_common.py", "tools/make_synthetic_rsna.py", "tools/run_notebooks_synthetic.py",
    "tools/validate_project.py", "tools/export_timm_weights.py",
]
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".ckpt", ".dcm", ".npz", ".parquet", ".safetensors", ".h5")


def check_files(rep):
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    rep["required_files_missing"] = missing
    return not missing


def check_syntax(rep):
    bad = []
    files = sorted(p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    for p in files:
        try:
            py_compile.compile(str(p), doraise=True, cfile=None)
        except py_compile.PyCompileError as e:
            bad.append(f"{p.relative_to(ROOT)}: {e.msg}")
    rep["python_files_checked"] = len(files)
    rep["syntax_errors"] = bad
    return not bad


def check_imports(rep):
    mods = sorted({".".join(p.relative_to(ROOT).with_suffix("").parts).replace(".__init__", "")
                   for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts})
    bad = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{m}: {type(e).__name__}: {e}")
    rep["modules_imported"] = len(mods) - len(bad)
    rep["import_errors"] = bad
    return not bad


def check_configs(rep):
    from src.utils.config import load_config

    bad = []
    for f in sorted((ROOT / "configs").glob("*.yaml")):
        for mode in ("smoke", "pilot", "full"):
            try:
                load_config(f, mode)
            except Exception as e:  # noqa: BLE001
                bad.append(f"{f.name}[{mode}]: {e}")
    rep["config_errors"] = bad
    return not bad


def check_notebooks(rep):
    import nbformat

    out = {}
    ok_all = True
    for p in sorted((ROOT / "notebooks").glob("*.ipynb")):
        info = {}
        try:
            json.loads(p.read_text(encoding="utf-8"))
            nb = nbformat.read(p, as_version=4)
            nbformat.validate(nb)
            info["valid_nbformat"] = True
        except Exception as e:  # noqa: BLE001
            info["valid_nbformat"] = False
            info["error"] = str(e)
            out[p.name] = info
            ok_all = False
            continue
        code = [c for c in nb.cells if c.cell_type == "code"]
        info["code_cells"] = len(code)
        info["stored_outputs"] = sum(len(c.get("outputs", [])) for c in code)
        info["execution_counts"] = sum(1 for c in code if c.get("execution_count") is not None)
        src = "\n".join(c.source for c in code)
        info["has_setup_cell"] = "def find_project_root" in code[0].source
        info["imports_src"] = "import src" in src and "from src" in src
        info["has_mode_cell"] = 'MODE = "smoke"' in src
        good = (info["has_setup_cell"] and info["imports_src"] and info["has_mode_cell"]
                and info["stored_outputs"] == 0 and info["execution_counts"] == 0)
        info["ok"] = good
        ok_all &= good
        out[p.name] = info
    rep["notebooks"] = out
    return ok_all and len(out) == 6


def check_zip(rep, zpath: Path):
    z = zipfile.ZipFile(zpath)
    names = z.namelist()
    prefix = "RSNA_ICH_Research/"
    missing = [f for f in REQUIRED if prefix + f not in names]
    forbidden = [n for n in names if n.lower().endswith(FORBIDDEN_SUFFIXES)]
    big = [(i.filename, i.file_size) for i in z.infolist() if i.file_size > 5_000_000]
    bad_crc = z.testzip()
    rep["zip"] = {"path": str(zpath), "n_entries": len(names), "size_mb": round(zpath.stat().st_size / 1e6, 3),
                  "missing_required": missing, "forbidden_data_or_checkpoints": forbidden, "files_over_5MB": big,
                  "crc_error": bad_crc, "reference_notebooks_included": [n for n in names if "reference_inputs" in n]}
    return not missing and not forbidden and not big and bad_crc is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args()
    rep: dict = {}
    results = {"files": check_files(rep), "syntax": check_syntax(rep), "imports": check_imports(rep),
               "configs": check_configs(rep), "notebooks": check_notebooks(rep)}
    if a.zip:
        results["zip"] = check_zip(rep, a.zip)
    rep["results"] = results
    rep["all_ok"] = all(results.values())
    text = json.dumps(rep, indent=2, default=str)
    if a.report:
        a.report.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if rep["all_ok"] else 1)


if __name__ == "__main__":
    main()
