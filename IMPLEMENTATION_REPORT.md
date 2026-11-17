# Implementation Report: RSNA ICH Classical and Quantum-Classical Research Framework

## 1. Files created
* **Workspace:** `reference_inputs/` (the three supplied notebooks, byte-identical read-only copies), `RSNA_ICH_Research/` (the project),
  `RSNA_ICH_Research_Project.zip` (the delivery).
* **Project:** README, requirements/environment, 4 YAML configs (smoke/pilot/full), 37 `src/` modules, 6 notebooks (no stored outputs),
  4 test files (60 tests), 6 development tools, 6 research documents, and this report. The full list is in
  `research_documentation/implementation_status.md`.

## 2. Findings from the three supplied notebooks (details: `research_documentation/existing_work_audit.md`)
* **N1 `RSNA_ICH_10CNN_extended.ipynb` (Keras/TF).** Only a one-step CPU **smoke test** was executed (160 images). The 41-cell
  "extended study" was **never executed**. Its executed part splits at **image level**, scales inputs to [0, 0.5] and never applies
  the backbone preprocessing, and its any/subtype network gives the subtype loss **weight 0**. The unexecuted part contains the best
  methodology of the three notebooks: patient-grouped splits, cluster bootstrap, paired tests, the RSNA intercept fix and a uint8 slice cache. It
  also selects the "best" model **on the test set**.
* **N2 `rsna-icch-ann.ipynb` (EfficientNet-B0 + MLP).** Image-level 70/15/15 split (**patient leakage**), backbone **frozen throughout**,
  `pos_weight = neg/pos` (epidural 218), fixed 0.5 thresholds. The saved test macro AUC is 0.885, but the outputs come from a different session than the
  notebook metadata, so it cannot be verified.
* **N3 `rsna-ich-comprehensive-benchmark.ipynb`.** 10-architecture benchmark on 20k images sampled at **image level before grouping**
  (≈1.5 slices/study), **study-level** (not patient-level) split, table sorted by **test** AUC, NASNet failed (recorded), the final comparison
  cell crashed (`KeyError`), and the artifact-saving cell never ran. Its saved ViT-B/16 test macro AUC of 0.925 cannot be verified.
* **Common:** duplicate rows in `stage_2_train.csv` are hidden by `pivot_table` (4,516,842 rows vs 752,803 images × 6). No
  notebook has a Swin/DeiT/MaxViT, CNN-Transformer or quantum component. None persisted reusable splits or predictions.
* **Papers.** Both study knee X-rays. Paper 2 drops KL 1-2 to make the task binary, and its quantum-vs-classical gain is 0.55 pp from one run
  with no CI. Paper 1 has no parameter-matched classical control and partly inconsistent baseline tables. Neither paper's numbers transfer to RSNA ICH.

## 3. Reused vs rewritten
* **Reused (refactored into `src/`):** HU conversion + MONOCHROME1 (N2/N3), three-window channels (N1/N2/N3), RSNA 12-bit intercept fix (N1),
  patient-cluster bootstrap / paired bootstrap / Holm / DeLong concepts (N1), Youden thresholds on development data (N1), study aggregation rules
  (N1), recording failed models instead of hiding them (N3), staged unfreezing (N1/N3, optional).
* **Rewritten:** label parsing (validation, duplicate/conflict reporting, soft-value preservation), header scanning (more fields, orientation-aware
  ordering, chunked restartable cache), splitting (single hash-identified patient-level manifest with leakage checks), datasets (on-demand, bounded
  caches), model factory (weight-status records, offline weights), training loop (AMP, resume, checkpoint guards), metrics (explicit NaN reasons, full
  six-label set), thresholds (val-only guard). Everything for Tracks A/B/C and the notebooks is new.

## 4. Architecture decisions (details: `research_documentation/architecture_notes.md`)
* One manifest, one preprocessing (fingerprinted), one loss, one evaluation for every model. Selection is by validation macro ROC-AUC only.
* **Track A:** DeiT-Small, Swin-Tiny, MaxViT-Tiny (timm), optional ViT-S/BEiT-B.
* **Track B:** EfficientNet-B0 stride-16 stem → 1×1 projection to 14×14 tokens + 2-D positional embedding + [CLS] → 2 pre-LN Transformer blocks →
  concat(CNN GAP, CLS) → MLP. Fusion ablations `gap_only`/`cls_only`, plus a standalone EfficientNet-B0 reference.
* **Track C:** a dressed VQC on the best backbone's frozen features: Linear(d→4) → LayerNorm → π·tanh → RY encoding → L×[Rot on each qubit + cyclic CNOT]
  → ⟨Z⟩ → Linear(4→16) → SiLU → Linear(16→6). Controls share the encoder and readout exactly: bottleneck-identity, bottleneck-MLP (20 vs 24 block params)
  and a linear probe. Predeclared grid (4 primary + 6 ablation heads) × 5 seeds. Paired seed tests and patient-level paired bootstrap. The result text is
  generated from the CI.
* **VQC backends:** an exact batched PyTorch state-vector simulator (default; GPU, autograd) and PennyLane (reference). Agreement is verified to ~1e-6.
  It is a classical simulation, so no quantum speed-up is claimed.
* Triage: max / mean / top-3 / noisy-OR study aggregation, study thresholds chosen on validation studies, and Grad-CAM labelled as non-segmentation.

## 5. Dataset assumptions (to be confirmed by notebook 00 on Kaggle)
* `stage_2_train.csv` is long format `ID_<hex>_<label>,Label`. Image IDs are the DICOM file stems in `stage_2_train/`.
* `PatientID`, `StudyInstanceUID`, `SeriesInstanceUID`, `ImagePositionPatient`, `ImageOrientationPatient` are present in most headers.
  If more than 1% lack PatientID, the split falls back to study level and says so.
* `any` should equal the max of the subtypes. Violations are reported, not corrected.
* `stage_2_test/` is unlabelled and is used for inference only.

## 6. Dependency versions (local verification)
Python 3.12.10 · torch 2.6.0+cu124 · timm 1.0.31 · pennylane 0.45.1 · pydicom 3.0.2 · numpy 2.5.3 · pandas 2.2.3 ·
scikit-learn 1.7.1 · scipy 1.16.1 · opencv 4.12.0 · PyYAML 6.0.3 · matplotlib 3.9.2 · nbformat 5.10.4 · nbclient 0.10.4 · pytest 9.1.1.
GPU: RTX 4060 Laptop 8 GB. Minimum versions are in `requirements.txt`.

## 7. Tests actually performed
| Check | Result |
|---|---|
| Unit tests: `python -m pytest tests -q` (data pipeline, models/VQC, metrics/statistics) | **60 passed, 0 failed, 0 skipped** |
| Static validation: `python tools/validate_project.py` (required files, compile 47 .py files, import 37 modules, configs × 3 modes, notebook structure) | **all_ok = true** |
| End-to-end: `python tools/run_notebooks_synthetic.py` (notebooks 00→05, smoke mode, synthetic DICOMs, random-init backbones, competition inference on) | **6/6 notebooks executed, 0 error outputs** (00: 44 s, 01: 95 s, 02: 79 s, 03: 114 s, 04: 57 s, 05: 55 s on an RTX 4060 Laptop) |
| VQC torch simulator vs PennyLane 0.45.1 (outputs and gradients) | max abs. difference ≤ 1.7e-6 |
| Pretrained weights: DeiT-Small and EfficientNet-B0 via timm/HF download; DeiT-Small from an offline local file | statuses `pretrained` / `pretrained_local_file`; local weights identical |
| ZIP: `python tools/validate_project.py --zip RSNA_ICH_Research_Project.zip` | all required files present, no data/checkpoints, CRC OK |

The metric values printed by the synthetic run come from geometric phantoms and a few training steps. They are **meaningless** and are not reported as results.

## 8. Tests that could not be performed
* No run on the real RSNA data. No Kaggle execution. No full-length training. No measured throughput or epoch times.
* Swin-Tiny/MaxViT-Tiny pretrained downloads (same code path as the verified DeiT-Small download). Compressed-DICOM decoding.

## 9. Known limitations
See `research_documentation/limitations.md`. In particular: 2-D slice models only; internal (not external) test set;
simulated noise-free quantum circuit; frozen-feature protocol for Track C; low-power seed tests; Grad-CAM is qualitative; no clinical validation.

## 10. Kaggle execution order
`00` → `01` → `02` (can run in parallel with 01) → `03` → `04` → `05`, all with the same `MODE`. Start with `smoke`, then `pilot`, then `full`.
Attach each earlier notebook's output as an input to the later ones (README §3).

## 11. Actions still required from you
1. Upload `RSNA_ICH_Research_Project.zip` as a private Kaggle Dataset. Import the six notebooks and attach the code dataset and the RSNA competition data.
2. Run notebook 00 in `smoke` and check the discovery output, the label report (duplicates) and the integrity report. Then run 01-05 in `smoke`.
3. Switch to `pilot` (and later `full`). Save a notebook version after each run and attach outputs downstream. Use `OVERRIDES` to split long runs across sessions.
4. Optional: if your notebooks run offline, create an offline weights dataset (`tools/export_timm_weights.py`) and `pip install pennylane` (or attach wheels).
5. Fill `research_documentation/experiment_log.md` from the generated `result.json` files. Report all models, all labels, and failed or skipped runs.
