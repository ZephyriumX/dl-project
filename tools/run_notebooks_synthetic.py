"""Execute all six notebooks end-to-end against a SYNTHETIC RSNA-layout dataset (development check).

    python tools/run_notebooks_synthetic.py --work <scratch dir> [--pretrained]

* creates synthetic DICOMs (tools/make_synthetic_rsna.py) - not medical data;
* runs notebooks 00 -> 05 in order with MODE=smoke via nbclient, passing the data/output locations
  through environment variables (RSNA_DATA_ROOT, RSNA_OUTPUT_ROOT, RSNA_ICH_PROJECT_ROOT);
* by default uses random-initialised backbones (no weight downloads) via RSNA_ICH_OVERRIDES_JSON;
* writes executed copies to <work>/executed/ - the notebooks in the project stay output-free.

Numbers produced by this run are meaningless (synthetic phantoms, a few training steps).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_synthetic_rsna import make  # noqa: E402


def overrides(pretrained: bool) -> dict:
    tm = [{"name": n, "kind": "timm", "pretrained": pretrained, "fallback": "random_init", "dropout": 0.2,
           "drop_path_rate": 0.1} for n in ("deit_small", "swin_tiny", "maxvit_tiny")]
    return {
        "data": {"header_scan": {"workers": 4}},
        "transformer": {"models": tm, "loader": {"num_workers": 0}, "ablation_single_window": True},
        "hybrid": {"hybrid": {"pretrained": pretrained}, "loader": {"num_workers": 0}, "fusion_ablation": ["gap_only"],
                   "cnn_reference": {"name": "efficientnet_b0", "kind": "timm", "pretrained": pretrained,
                                     "fallback": "random_init", "dropout": 0.2, "drop_path_rate": 0.1}},
        "quantum": {"backbone": {"feature_batch_size": 32}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    work = Path(a.work)
    data, out, exe = work / "synthetic_rsna", work / "outputs", work / "executed"
    exe.mkdir(parents=True, exist_ok=True)
    if not (data / "stage_2_train.csv").exists():
        print("synthetic data:", make(data, patients=30, slices=10, size=128, test_images=12, seed=0))
    env = {"RSNA_DATA_ROOT": str(data), "RSNA_OUTPUT_ROOT": str(out), "RSNA_ICH_PROJECT_ROOT": str(ROOT),
           "RSNA_ICH_MODE": "smoke", "RSNA_ICH_OVERRIDES_JSON": json.dumps(overrides(a.pretrained)),
           "RSNA_ICH_RUN_COMPETITION_INFERENCE": "1"}
    os.environ.update(env)
    summary = []
    for nb_path in sorted((ROOT / "notebooks").glob("0*.ipynb")):
        if a.only and not any(nb_path.name.startswith(o) for o in a.only):
            continue
        nb = nbformat.read(nb_path, as_version=4)
        client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                                resources={"metadata": {"path": str(ROOT / "notebooks")}})
        t0 = time.time()
        status, err = "ok", ""
        try:
            client.execute()
        except CellExecutionError as e:
            status, err = "error", str(e)[-3000:]
        nbformat.write(nb, exe / nb_path.name)
        n_err = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error")
        summary.append({"notebook": nb_path.name, "status": status, "error_outputs": n_err, "seconds": round(time.time() - t0, 1)})
        print(summary[-1], flush=True)
        if err:
            print(err.encode('ascii', 'replace').decode())
            break
    (work / "notebook_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sys.exit(0 if all(s["status"] == "ok" for s in summary) else 1)


if __name__ == "__main__":
    main()
