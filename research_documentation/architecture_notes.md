# Architecture Notes and Design Decisions

These decisions were made after the audit in `existing_work_audit.md`. Each one names the problem it addresses.

## D1. One shared pipeline, one split manifest
* `src/data/` is the only place where labels are parsed, DICOMs are read, splits are made and images are preprocessed.
  Notebooks 01-05 **load** the manifest written by Notebook 00. None of them split the data again.
* The manifest (`split_manifest.csv` + `split_manifest.json`) carries a SHA-256 hash of the sorted `(image_id, split)` pairs.
  Every checkpoint, prediction file, threshold file and metric file records that hash, and loaders refuse to mix hashes
  (`src/training/checkpointing.py::ManifestMismatchError`).
* *Addresses:* N2/N3 re-split independently per notebook. N3 never saved its splits.

## D2. Patient-level partitioning
* The grouping key is the DICOM `PatientID`. If more than `max_missing_patient_fraction` (default 1%) of scanned images lack a
  PatientID, the split falls back to `StudyInstanceUID`. The manifest then records `grouping_level = "study_fallback"`, and
  every report prints that the split is **study-level, not patient-level**.
* Stratification is done at patient level. Each patient gets a stratum (has any ICH / has epidural / has other subtype / negative),
  and patients are shuffled and allocated per stratum with a fixed seed. All images of a patient go to the same split.
* Subsampling for pilot runs happens **after** grouping, at the patient level (`max_patients`). It never happens at image level.
* Header scanning can be partial (smoke/pilot). This does not leak: only scanned images are ever used, and grouping is applied to
  them. The cost is that unscanned slices of a patient are absent, which is recorded in the manifest JSON.
* *Addresses:* N2 image-level split, N3 study-level split, N3 image-level sampling before grouping.

## D3. Labels are preserved, not "cleaned" silently
* `csv_parser.parse_label_csv` validates the `ID_<hex>_<subtype>` format and drops exact duplicate rows (counted). Conflicting
  duplicates are resolved by a configurable policy (`max` by default) and are always listed in `label_report.json`.
  Non-binary values are kept as they are and reported. It checks `any == max(subtypes)` and reports violations.
* Losses accept soft targets. Metrics binarise targets at 0.5 **only if** non-binary values exist, and flag that in the output.

## D4. Preprocessing
* HU = `pixel * RescaleSlope + RescaleIntercept`, after the RSNA 12-bit intercept fix (from N1-extended, configurable) and
  MONOCHROME1 inversion. Non-square images are zero-padded (in HU, padding value = -1024) to square before resizing.
* Window presets: `brain` (40/80), `subdural` (80/200), `soft_tissue` (40/380), `bone` (600/2800). Channel modes:
  `brain3` (single brain window replicated ×3, the single-window ablation) and `bsb` (brain/subdural/soft, the default
  multi-window input, same as N2/N3). Any list of windows can be configured.
* `preprocessing_fingerprint()` hashes the full preprocessing config. It is stored with checkpoints and checked at inference, so
  training and inference preprocessing are guaranteed identical.
* Caching is optional and bounded: an in-process LRU of decoded `uint8` arrays (`memory_cache_items`) and an optional on-disk `uint8`
  `.npy` cache in `/kaggle/working/cache/<fingerprint>/` with a byte budget (`disk_cache_max_gb`). The disk cache is restartable
  (files are written atomically) and is keyed by the preprocessing fingerprint, so it never mixes configurations.

## D5. Track A — Transformers (timm)
* Candidates: `deit_small_patch16_224`, `swin_tiny_patch4_window7_224`, `maxvit_tiny_tf_224`. Optional extras: `vit_small_patch16_224`,
  `beit_base_patch16_224`, each enabled only if `timm` knows the model.
* `build_backbone()` returns a `ModelBuildRecord` with status `pretrained`, `pretrained_local_file`, `random_init_fallback` or `failed`.
  A random-init fallback is **never** reported as pretrained, and the benchmark table prints the status column.
* Kaggle notebooks often run **offline**. Weights are looked up in this order: (1) a local file in an attached weights dataset
  (`pretrained_weights_dir`), (2) the HF hub / timm download, (3) the fallback policy (`random_init` or `skip`).

## D6. Track B — CNN–Transformer hybrid (`CNNTransformerHybrid`)
```
x ─► CNN stem (timm features_only, EfficientNet-B0 stages 1-4, stride 16, 112 ch @14×14 for 224 input)
       ├─► GAP ─────────────────────────────────────────────┐
       └─► 1×1 conv → d_model tokens + 2-D learned pos-emb   │
             + [CLS] ─► L × TransformerEncoder (pre-LN) ─► CLS ┤
                                                           concat ─► LayerNorm ─► MLP ─► 6 logits
```
* *Motivation:* acute blood is a small, local hyperdensity (≈50-90 HU). Convolutions with their locality bias detect it well.
  Several findings need **global context**, for example comparing hemispheres, relating ventricles to the surrounding parenchyma, and
  separating thin subdural rims from calvarial bone. Self-attention over mid-level CNN tokens gives every location access to the whole
  slice at 14×14 resolution, at a fraction of a full ViT's cost. Fusing the CNN GAP vector (local evidence) with the CLS token (contextual summary) lets the
  head use both, and the fusion ablation (`fusion: cls_only | gap_only | concat`) measures whether the transformer adds anything.
* The attention weights of the last block are exposed for visualisation (not as segmentation).

## D7. Track C — Dressed variational quantum circuit
```
CT ─► backbone (best Track-A model by VALIDATION macro AUC) ─► f ∈ R^d
   ─► Linear(d→n_q) ─► LayerNorm ─► angle scaling (π·tanh(·)) ─► VQC(n_q, L) ─► ⟨Z_i⟩ ∈ [-1,1]^{n_q}
   ─► Linear(n_q→h) ─► SiLU (Swish, as Paper 2) ─► Linear(h→6)
```
* **VQC (default = Paper 2):** n_q=4; encoding `RY(θ_i)` (optionally preceded by `H` as Paper 1); L variational layers of
  `Rot(φ,θ,ω)` per qubit followed by a **cyclic CNOT ring**; Pauli-Z readout per qubit. Configurable options: `n_qubits ∈ {2,4,6,8}`,
  `n_layers ∈ {1,2,4}`, `entanglement ∈ {cyclic, linear, none}`, `data_reuploading`.
* **Two exact simulators:**
  1. `backend="torch"`: a batched state-vector simulator written in PyTorch (complex64). It is exact, differentiable by autograd, and runs
     on GPU. Cost is O(B·2^n) per gate, which is trivial for n ≤ 10. This is what makes training on hundreds of thousands of slices feasible.
  2. `backend="pennylane"`: the same circuit as a PennyLane QNode (`default.qubit` / `lightning.qubit`, torch interface). It serves as the reference
     implementation. `tests/test_models.py` checks that both backends give the same outputs and gradients to 1e-5.
  Both are **classical simulations of an ideal, noise-free circuit**. No quantum hardware is involved, and no speed-up is claimed.
* **Classical controls (same backbone, same features, same projection, same budget):**
  * `bottleneck_identity`: Linear(d→n_q) → LayerNorm → π·tanh → *(no processing)* → same head. Isolates the effect of the 4-dim bottleneck.
  * `bottleneck_mlp`: the same, with the VQC replaced by `Linear(n_q→n_q) → tanh` (parameter count matched to within a few parameters).
  * `linear_probe`: Linear(d→6) on the same frozen features (upper reference).
* **Default protocol:** frozen backbone, features extracted **once** per split and cached. Heads are trained on cached features, with
  `n_seeds` (default 5) per configuration, a fixed epoch budget, and early stopping on validation macro AUC. The configuration
  grid is predeclared in `configs/quantum_config.yaml` and **bounded** (≤ 12 head configurations). End-to-end fine-tuning through
  the VQC is available (`backbone_mode: finetune`) but is not the default.
* *Addresses:* Papers 1-2 had no parameter-matched classical control, a single run, and no paired statistics.

## D8. Training
* `BCEWithLogitsLoss` with configurable `pos_weight` (`none`, `sqrt_inverse` (default, clipped to [1, 20]), `inverse` (clipped)) and
  per-label weights (default RSNA `[1,1,1,1,1,2]` for `any`). Focal loss available. The `neg/pos` weights used in N2/N3 (up to 249) are
  not the default because they cause severe mis-calibration.
* No resampling by default (the evaluation distribution is preserved). An optional **training-only** negative-subsampling ratio exists.
* AdamW, separate backbone/head LRs, warm-up + cosine schedule, gradient clipping, AMP (bf16 if supported, else fp16 with GradScaler),
  gradient accumulation, early stopping on validation macro ROC-AUC, last and best checkpoints, full resume (model/optimiser/scheduler/scaler/RNG).
* `src/utils/runtime.py` detects the GPU and memory. Batch size is chosen from a memory table and then confirmed by an OOM probe
  that halves the batch until a forward/backward pass fits.

## D9. Evaluation and statistics
* All metrics in `src/evaluation/metrics.py` return `NaN` for undefined values, together with a reason (e.g. `single_class`). Macro averages
  report how many labels were defined.
* Thresholds: per label, selected on **validation** (Youden J by default, F1 or target-sensitivity optional), saved to JSON with
  the manifest hash and `selected_on="val"`. `select_thresholds` raises if asked to use the test split.
* CIs: **patient-cluster bootstrap** (default 1,000 resamples). Paired model comparison: paired cluster bootstrap of ΔAUC on the
  intersection of image IDs, plus a two-sided bootstrap p-value. Holm correction across comparisons. DeLong is available
  for reference and documented as anti-conservative with correlated slices.
* Multi-seed comparisons (Track C) report the mean ± SD over seeds, and the per-seed paired bootstrap on the test set.

## D10. Restartability and persistence
* `/kaggle/working/rsna_ich_artifacts/` (configurable) is the output root. Every stage writes atomically (`*.tmp` → rename).
* `experiment_registry.jsonl` records every experiment with status `planned | running | succeeded | failed | skipped`, its configuration,
  fingerprint, timing and error message. Notebook 04 builds its tables from this registry and the saved predictions, and marks missing experiments explicitly.

## Structural changes relative to the requested layout
* Added `src/utils/` (`config.py`, `paths.py`, `runtime.py`, `repro.py`, `registry.py`, `io.py`). Dataset discovery, runtime
  detection, YAML/mode handling, seeding and the experiment registry are needed by every notebook and belong to none of the requested packages.
* Added `src/evaluation/explainability.py` (Grad-CAM / attention maps) and `src/evaluation/submission.py` (competition-format output).
* Added `src/quantum/torch_simulator.py` (the fast exact simulator used by `vqc.py`).
* Added `tools/` (`build_notebooks.py`, `make_synthetic_rsna.py`, `run_notebooks_synthetic.py`, `validate_project.py`). These are development tools
  for generating notebooks and testing them end-to-end against a small **synthetic** DICOM dataset. They are never used for research results.
