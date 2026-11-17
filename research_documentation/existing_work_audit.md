# Existing-Work Audit

**Scope.** This audit covers exactly the three notebooks that were supplied as reference material, plus the two
research papers. Everything below comes from reading the notebook JSON (code, markdown, and saved outputs). None
of the three notebooks was re-executed for this audit: the RSNA data is not in the development environment, and
re-execution was out of scope. Saved outputs are reported as **saved outputs**. They are not independently verified results.

| # | File (copied read-only to `reference_inputs/`) | SHA-256 (first 16 hex) | Cells | Framework |
|---|---|---|---|---|
| N1 | `RSNA_ICH_10CNN_extended.ipynb` | `a51d901d098b09e6` | 119 (cells 0-77 original baseline, 78-118 "extended study") | TensorFlow 2.17.1 / Keras 3.12.0 |
| N2 | `rsna-icch-ann.ipynb` | `89c911ad7c875653` | 26 | PyTorch 2.10.0+cu128 / torchvision |
| N3 | `rsna-ich-comprehensive-benchmark.ipynb` | `8503bf369f73f793` | 41 | PyTorch 2.10.0+cu128 / torchvision 0.25 / timm |

The copies in `reference_inputs/` are byte-identical to the originals (checked by hash). The originals were not modified.

> **Note on the ANN notebook.** The task brief treated the ANN/EfficientNet-B0 baseline as "background only unless supplied".
> `rsna-icch-ann.ipynb` (N2) **is** one of the three supplied notebooks and contains that EfficientNet-B0 + MLP ("ANN") baseline,
> so it was inspected as source code. No other ANN source was assumed.

---

## 1. Notebook N1 — `RSNA_ICH_10CNN_extended.ipynb`

### 1.1 Provenance and execution state
* Kaggle metadata lists the RSNA competition (`sourceId 13451`) plus one other dataset. The saved outputs came from a
  **local Windows CPU run** (`Python 3.12.10 ... MSC v.1943 64 bit (AMD64)`, `TensorFlow GPUs: []`, `is_cuda_build False`).
  Data was read directly from `C:\Users\...\Downloads\rsna-intracranial-hemorrhage-detection.zip`.
* **Cells 0-77** ("original notebook", an adaptation of a public Keras baseline) have execution counts 1-60 and outputs.
  They ran in **smoke mode only** (`RUN_SMOKE_TEST = True`): 160 training images (80 positive / 80 negative, sampled at
  image level), 20 test images, batch size 2, **one training step** per model. `train_brute_force = False` and
  `train_anysubtype_network = False`, so no real training happened. The saved losses (e.g. `loss: 1.6026`) come from a single step and say nothing about model quality.
* **Cells 78-118** (the "Extended study": 10-architecture comparison, CV, statistics, 2.5D, 3D, ablations, external validation)
  all have `execution_count = None` and **no outputs**. They were never executed. None of that section's results exist.

### 1.2 Data loading and labels
* `stage_2_train.csv` is split with `ID.str.rsplit("_", n=1)` into `id` / `subtype`, then pivoted.
  * Original section: `pd.pivot_table(..., values="label")` with the **default `aggfunc="mean"`**. Any conflicting duplicate
    rows would silently become fractional labels (e.g. 0.5). Duplicates are not checked or reported.
  * Extended section: `aggfunc="first"`. This silently keeps the first duplicate.
* Competition test IDs come from `stage_2_sample_submission.csv`, with labels set to 0 (inference only, which is correct).
* Label columns end up in alphabetical order (`any` first), which differs from N2/N3.

### 1.3 DICOM preprocessing
* Original: `pixel_array * RescaleSlope + RescaleIntercept`, clip below -1024 HU, a **single 0-90 HU window** replicated to
  3 channels, `skimage.resize` to 224x224, then `normalize()` maps to **[0, 0.5]** (`... * 0.5`). The backbone-specific
  `preprocess_fun` (ResNet/VGG caffe preprocessing) is stored but **never applied**, so ImageNet weights receive out-of-distribution inputs.
* Extended: three windows (brain 0-80, subdural -20-180, bone -800-2000 HU). It also includes the **known RSNA rescale-intercept fix**
  (`BitsStored == 12`, `PixelRepresentation == 0`, intercept > -100 → shift by +1000 and wrap at 4096). This fix is sound and is
  reused in the new project. Slices are decoded **once** into a `uint8` memmap cache, keyed by an MD5 hash of the subset IDs. This is also a good, reusable idea.
* MONOCHROME1 is not handled in N1.

### 1.4 Splitting
* Original: `StratifiedKFold` on `any` at **image level**. Slices from the same patient/study end up in train and dev (**leakage**).
  The extended section's own markdown says so.
* Extended: header scan (PatientID, StudyInstanceUID, z from `ImagePositionPatient[2]`), cached and resumable. **Patient-grouped**
  `StratifiedGroupKFold` on `any`, with an explicit patient-overlap printout. Sampling happens at **study level** (`N_STUDIES=300` in
  "full" mode, `MIN_SLICES=20`). This is the most methodologically sound split logic of the three notebooks, but it never ran.
* z-ordering uses `ImagePositionPatient[2]` only. It ignores `ImageOrientationPatient`, which matters for gantry-tilted or oblique acquisitions.

### 1.5 Models
* Original: Keras ResNet50 / VGG16 (frozen, `include_top=False`) → GAP → Dropout → Dense(100) → Dropout → Dense(6, sigmoid) with
  L2(0.05) on the output. Also an "any → subtype" two-output network. Its compiled loss has
  `loss_weights={"any": 1.0, "subtype_pred": 0.0}`, so **the subtype output is never trained**. This bug is not commented on in the notebook.
* Extended: a zoo of 10 backbones (LeNet, AlexNet, GoogLeNet, ViT-Tiny from scratch; VGG16, ResNet50, DenseNet121, Xception,
  NASNetMobile, EfficientNetB0 ImageNet-pretrained) behind one shared head, with per-backbone input normalisation layers. The notebook
  states that from-scratch and pretrained models are confounded and provides a "random init" ablation for that.

### 1.6 Training
* Original: Adam 1e-4, `ModelCheckpoint`/`ReduceLROnPlateau`/`EarlyStopping` on `val_loss`. Per-epoch **undersampling** to 1:2
  positive:negative in train **and dev** loaders (`dev_dataloader` is built with `steps=STEPS`). This changes the validation
  distribution, so val metrics are not representative.
* Extended: two-stage fine-tuning (frozen backbone, then top `unfreeze_frac` layers, BatchNorm kept frozen), random hyper-parameter
  search on a patient-grouped tuning split, `neg_ratio` negative undersampling of training batches only, and losses BCE / weighted
  BCE (positive weights `sqrt(inverse prevalence)` clipped to [1, 10]) / focal. Class weights `[2,1,1,1,1,1]` follow the RSNA metric.

### 1.7 Metrics and statistics (extended section, unexecuted)
* Primary metric is **`any` AUC only**. Subtype macro-AUC and RSNA weighted log-loss are secondary columns.
* Patient-cluster bootstrap CIs, paired cluster-bootstrap AUC differences, DeLong (flagged as anti-conservative), McNemar,
  Friedman, Nadeau-Bengio corrected t-test, Holm correction, Youden thresholds chosen on dev OOF predictions and frozen for test.
  The statistical design is strong and is the main methodological input to the new `src/evaluation/statistics.py`.
* **Selection-on-test problem:** `compare_to_best(MAIN, "test")` picks `BEST_NAME` as the model with the highest **held-out test**
  AUC. `BEST_NAME` then sets `CTX_MODEL` (the backbone for context/ablation studies) and the BiGRU base model. So the held-out test
  set influences later experimental design.

### 1.8 Other components
* Study-level aggregation (max / mean / top-3 / noisy-OR) and a BiGRU over slice-probability sequences: reusable ideas.
* 3D Conv on sub-volumes (label = any positive slice in window), hand-crafted HU/texture features with LogReg/RF/HGB, and a
  synthetic "scanner-shift" stress test (correctly labelled as *not* external validation).

### 1.9 Bugs / risks summary (N1)
1. Image-level split in the executed part (leakage). 2. Mean-aggregated pivot can create fractional labels silently.
3. Backbone preprocessing not applied; inputs scaled to [0, 0.5]. 4. Subtype head weight 0 in the any/subtype network.
5. Dev loader undersampled. 6. Best model chosen on test in the extended section. 7. Extended section never executed.
8. Uses the deprecated `skimage` resize and Keras-specific code that cannot be reused in a PyTorch project.

---

## 2. Notebook N2 — `rsna-icch-ann.ipynb` (EfficientNet-B0 + "ANN" head)

### 2.1 Execution state
* Kaggle metadata says `accelerator: none`, `isGpuEnabled: false`, but the saved outputs show `Device: cuda`, `GPU: Tesla T4`,
  `PyTorch 2.10.0+cu128`. The saved outputs and the notebook metadata therefore come from **different sessions**.
  The outputs (10 epochs, test macro ROC-AUC **0.885**) look like a complete run, but **cannot be independently verified**.

### 2.2 Pipeline
* Labels: correct `rsplit("_", 1)` + `pivot_table(aggfunc="max")` → **752,803 unique images from 4,516,842 rows**.
  4,516,842 / 6 = 752,807, so the CSV contains duplicate rows. `max` hides them silently, with no report.
* Subset: `MAX_IMAGES = 20000` drawn by **image-level** stratified sampling on the 6-label string (positive rate `any` = 0.143,
  epidural = 0.0042).
* Preprocessing: HU via slope/intercept, MONOCHROME1 inversion, three windows: "brain" (40/80), "blood" (80/200, which is the
  standard *subdural* window despite the name), "soft_tissue" (40/380). Bilinear resize to 224, ImageNet normalisation. No RSNA
  intercept fix. No handling of non-512 or non-square images beyond plain resize.
* Split: `train_test_split` 70/15/15 at **image level**, no grouping → **patient and study leakage between train/val/test.**
  The reported test metrics are therefore optimistic by an unknown amount.
* Model: torchvision EfficientNet-B0 (ImageNet) with the **feature extractor frozen for the entire run** (comment says "Freeze CNN
  initially" but it is never unfrozen) → AvgPool → MLP 1280-512-256-6 with BatchNorm + Dropout. 790,278 trainable parameters out of 4.8 M.
* Loss: `BCEWithLogitsLoss(pos_weight = neg/pos)` → epidural weight **217.75**. Such extreme weights push probabilities far
  above the true prevalence (poorly calibrated). The reports show this: epidural precision 0.02, recall 0.75 at threshold 0.5.
* Optimiser AdamW 1e-4, ReduceLROnPlateau on val macro AUC, 10 epochs, no AMP, no early stopping. Best `state_dict` saved by val AUC.
* Evaluation: per-label ROC-AUC / PR-AUC / F1 at **fixed threshold 0.5**, plus classification reports. The validation macro AUC
  silently **drops** labels with a single class (`except ValueError: pass`) rather than reporting them as undefined.
* Saved test results (unverified): ROC-AUC epidural 0.874, IPH 0.895, IVH 0.925, SAH 0.867, SDH 0.864, any 0.884. Macro PR-AUC 0.372,
  macro F1 0.258.

### 2.3 Issues
1. Image-level split → leakage. 2. Backbone permanently frozen. 3. Extreme `pos_weight` → poor calibration and threshold
mismatch. 4. Threshold 0.5 not tuned on validation. 5. No CIs. 6. Metadata/outputs inconsistency. 7. Hard-coded Kaggle path
`/kaggle/input/competitions/...` (this layout differs between Kaggle versions).

---

## 3. Notebook N3 — `rsna-ich-comprehensive-benchmark.ipynb`

### 3.1 Execution state
* Cells 0-37 have execution counts 1-23 and outputs, from a Tesla T4 session (PyTorch 2.10.0+cu128). **Cell 37 raised
  `KeyError: 'macro_roc_auc'`** (the comparison table expects a column the benchmark never produces). **Cell 39 (artifact saving) never ran**,
  so splits, predictions and configs were not persisted.
* Several "experiments" listed in the introduction are only templates or print statements: bootstrap CIs (commented example),
  CV (`RUN_CROSS_VALIDATION=False`), DeLong (example), classical ML (example), 2.5D (dataset constructed, never trained),
  3D (dataset constructed, never trained), ensemble (prints instructions), ablations (list only), external validation (adapter only).

### 3.2 Pipeline
* Labels: same as N2 (`aggfunc="max"`). 752,803 images.
* Subset: `MAX_IMAGES=20000` image-level stratified sample **before** reading DICOM headers. The result has 20,000 images from
  **13,030 studies / 11,784 patients**, about 1.5 slices per study. Neighbouring slices are therefore almost never present, which
  makes the 2.5D/3D datasets meaningless here (the "3D volume clips: 9120" are mostly 1-2 real slices with clamped padding).
* Grouping: `group_id = StudyInstanceUID` (fallback PatientID, then image_id). `GroupShuffleSplit` 70/15/15 by **study**.
  The leakage assertion checks study disjointness only. Patients with several studies can appear in more than one split.
  So this is a **study-level split, not a patient-level split.**
* Preprocessing identical to N2, plus augmentation (h-flip, ±7° rotation, 3% translation, 0.95-1.05 scale) applied to the windowed tensor.
* Class imbalance: `pos_weight = neg/pos` (epidural 249).
* Model factory: LeNet (LazyLinear), AlexNet, VGG16, GoogLeNet (aux logits), ResNet50, DenseNet121, EfficientNet-B0 (torchvision,
  ImageNet), Xception, NASNet-A-Mobile, ViT-B/16 (timm). **`nasnetamobile` failed** ("Unknown model") and was recorded as an error row.
  This is good practice: the failure was not hidden.
* Training: 2 head-only epochs then full fine-tuning at 1e-5 (AdamW), gradient clipping 5.0, 10 epochs, best epoch by val macro AUC
  kept **in memory** (`copy.deepcopy`). No AMP, no disk checkpoints, no resumability. `trained_models[arch] = m` keeps every model
  alive, so `del m` does not free GPU memory. With 10 models this can exhaust memory.
* Metrics: `macro_metrics` returns NaN for single-class labels and uses `nanmean` (correct). F1 at a fixed 0.5 threshold.

### 3.3 Saved results (unverified, study-level split, 20k images, single seed)
| model | val macro AUC | test macro AUC | test macro PR-AUC |
|---|---|---|---|
| vit_base (B/16) | 0.943 | 0.925 | 0.588 |
| densenet121 | 0.938 | 0.913 | 0.488 |
| vgg16 | 0.933 | 0.909 | 0.454 |
| resnet50 | 0.916 | 0.908 | 0.449 |
| efficientnet_b0 | 0.913 | 0.899 | 0.454 |
| xception | 0.916 | 0.895 | 0.433 |
| googlenet | 0.913 | 0.887 | 0.426 |
| alexnet | 0.900 | 0.873 | — (truncated in output) |
| lenet | 0.779 | — (truncated) | — |
| nasnet_mobile | FAILED | — | — |

Also saved: frozen EfficientNet-B0 test macro AUC 0.837 vs fine-tuned 0.900. Spatial-attention EfficientNet test macro AUC 0.898.

### 3.4 Issues
1. Study-level, not patient-level split. 2. Image-level sampling before grouping (≈1.5 slices/study), which invalidates 2.5D/3D
context. 3. **Benchmark table sorted by test AUC**, which invites model selection on test. 4. Bootstrap CI implementation
resamples **images**, not patients (the function exists but is unused). 5. DeLong implementation computes an O(n_pos × n_neg) matrix and
contains dead code (`v01 = (ty[:, None] if False else None)`). 6. Frozen-vs-fine-tuned comparison is not budget-matched (the frozen
model trains 10 epochs without model selection, the fine-tuned model uses best-epoch selection). 7. The final comparison cell crashes. 8. No
artifacts saved. 9. `2.5D` averages the windowed slices back to 3 channels, which mostly removes the context it was meant to add.
10. timm name `xception` is deprecated (renamed `legacy_xception`). It still worked in that environment.

---

## 4. Cross-notebook comparison

### 4.1 Genuinely different components
| Component | N1 | N2 | N3 |
|---|---|---|---|
| Framework | Keras/TF | PyTorch | PyTorch + timm |
| Split | image-level (executed); patient-grouped (unexecuted) | image-level | study-level |
| Windows | 0-90 single (exec); brain/subdural/bone (unexec) | brain/subdural/soft | brain/subdural/soft |
| RSNA intercept fix | yes (unexec) | no | no |
| Caching | uint8 memmap (unexec) | none | none |
| Statistics | cluster bootstrap, paired tests, Holm (unexec) | none | image-level bootstrap (unused), DeLong (unused) |
| Study aggregation | max/mean/top3/noisy-OR + BiGRU (unexec) | none | none |

### 4.2 Duplicates / near-duplicates
* N2 and N3 share almost identical label parsing, `dicom_to_hu`, `window_image`, three-window channel construction, ImageNet
  normalisation, and the `neg/pos` positive weighting. N3 is essentially N2 plus a model factory and templates.
* Training/validation loops in N2 and N3 are near-duplicates (N3 adds clipping and staged unfreezing).

### 4.3 Most reusable components (and what was done with them)
| Component | Source | Decision in new project |
|---|---|---|
| `rsplit("_",1)` + pivot label parsing | N2/N3 | **Rewritten** in `src/data/csv_parser.py`, with malformed-ID detection, duplicate/conflict reporting, `any`-consistency checks and preservation of fractional values. |
| HU conversion + MONOCHROME1 | N2/N3 | **Reused (refactored)** in `src/data/preprocessing.py`. |
| RSNA rescale-intercept fix | N1 (extended) | **Reused (refactored)**, configurable, counted in the audit. |
| Multi-window channels | N1/N2/N3 | **Reused**, windows made configurable (presets for single-window vs multi-window ablation H). |
| Header scan with resumable cache | N1 (extended) | **Rewritten** for more fields (series, IPP, IOP, spacing) with chunked restartable parquet/CSV cache. |
| Patient-grouped stratified split | N1 (extended) | **Rewritten** as a single manifest generator with hashing and overlap verification. |
| Cluster bootstrap / paired bootstrap / Holm / DeLong | N1 (extended) | **Reused conceptually, rewritten** in `src/evaluation/statistics.py`, vectorised and multi-label. |
| Youden thresholds from dev, frozen for test | N1 (extended) | **Reused**, generalised to per-label threshold sets with a hard guard against selecting on test. |
| Study aggregation (max/mean/top-k/noisy-OR) | N1 (extended) | **Reused** in `src/evaluation/study_aggregation.py`. |
| Spatial attention over CNN features | N3 | Not reused directly. Motivated the attention map output of the CNN-Transformer hybrid. |
| torchvision/timm model factory with failure recording | N3 | **Rewritten** as `src/models/transformers.py` with explicit status records (pretrained / fallback / failed). |
| Staged unfreezing | N1/N3 | **Reused** as an optional `freeze_backbone_epochs` setting in the trainer. |

### 4.4 Most reliable approaches
1. Patient-grouped splitting with explicit overlap checks (N1 extended).
2. Patient-cluster bootstrap for CIs and paired comparisons (N1 extended).
3. Thresholds chosen on development predictions and frozen (N1 extended).
4. Three CT windows plus the RSNA intercept fix (N1 extended + N2/N3).
5. Explicit failure rows for models that fail to build (N3).

### 4.5 Incorrect or questionable choices (all notebooks)
* Image-level or study-level splitting presented as leakage-free (N1-exec, N2, N3).
* Model ranking or selection using held-out test results (N3 table ordering; N1 `BEST_NAME`).
* Extreme `neg/pos` positive weights (N2, N3) without calibration analysis.
* Fixed 0.5 thresholds (N2, N3).
* Silent duplicate handling in label pivot (all).
* Image-level subsampling before grouping, which destroys volumetric context (N3).
* Frozen backbone presented as a full baseline (N2).

### 4.6 Missing experiments / incomplete functionality
* No notebook trains a modern vision Transformer other than ViT-B/16 (N3). No Swin, DeiT or MaxViT.
* No CNN-Transformer hybrid. No quantum component of any kind.
* No persisted split manifests, checkpoints with metadata, or prediction files (N3 artifact cell never ran).
* No study-level evaluation actually executed. No calibration analysis. No Brier score. No specificity/TP/FP tables (except N1-unexec).
* No competition-test inference pipeline producing a valid submission (only N1 smoke with 20 images).

### 4.7 Results that cannot be independently verified
* All saved metrics in N2 and N3 (no saved predictions, splits, or checkpoints; N2 metadata inconsistent with outputs).
* None of N1's extended-study results exist. Its executed part is a one-step smoke test.
* None of these numbers may be compared with the new project's results as if they came from the same protocol. They use different
  splits (image/study-level), different subsets (20k image-level samples), and no patient grouping.

---

## 5. Research papers

### 5.1 Paper 1 — Dong et al., *Classification of knee osteoarthritis based on quantum-to-classical transfer learning*, Front. Phys. 11:1212373 (2023)
**Access:** full text (14 pages) read from the supplied PDF via text extraction. Figures (architecture diagrams, ROC plots,
confusion matrices) are images and were not reviewed in detail. Their captions were read.

* **Architecture (HQCNN):** a *quanvolutional* layer (after Henderson et al.) slides over the image. Each n×n patch is angle-encoded
  (Hadamard then `RY(π·z_i)` on 4 qubits), passed through a variational circuit of interleaved single-qubit `RY` rotation layers and
  nearest-neighbour CNOT entanglement with **depth 4**, then decoded by **Pauli-Z** measurement into a 4-channel quantum feature map.
  The quantum feature maps are **precomputed and serialised to disk**, then fed to a classical **Xception** network (GAP + dense).
* **Transfer scheme:** HQCNN is pre-trained on a 4-class brain-tumour MRI dataset (3,264 images, 233 patients) and reused as a feature extractor (512 features) for 5-class KL grading on OAI X-rays (9,516 images): `L_{512→64} ∘ K ∘ L_{64→5}`.
  The paper's Table 1 also describes the "dressed quantum circuit" of Mari et al.: `L_{4→2} ∘ Q ∘ L_{512→4}` (512 classical features → 4 qubits → 2 outputs).
* **Training:** Glorot-uniform init, categorical cross-entropy, Adam lr 1e-3, images 128×128×1 scaled to [0,1], stratified 10-fold
  CV (Table 4 per-fold accuracy 0.948-0.999), plus "adversarial validation" to *repartition* the MRI train/test sets.
* **Reported:** HQCNN 97.85% (MRI), QCTL 98.36% (OAI), a "1.08%" gain from quantum-to-classical transfer.
* **Limitations observed (our assessment):** splits are image-level; patient grouping is not described for either dataset. Test sets
  were repartitioned after looking at distribution shift. No confidence intervals or significance tests. Table 5 and Table 6 report very
  different numbers for the same classical models (e.g. EfficientNet 90.82 vs 56.38), so the classical baselines were not tuned
  under comparable conditions. Per-fold losses vary from 0.0017 to 0.91. The quantum circuit is simulated, and no ablation separates
  the quantum layer from a parameter-matched classical layer. Notation errors in the text ("Bubbleley spin matrices", "Adama gate").
* **What we take from it:** the *dressed quantum circuit* pattern `L_{d→n} ∘ Q ∘ L_{n→out}` with angle encoding and Pauli-Z readout,
  and the idea of precomputing classical features before the quantum stage.

### 5.2 Paper 2 — Tikariha et al., *Osteoarthritis Classification Using Hybrid Quantum Convolutional Neural Network*, IEEE Access 13 (2025), DOI 10.1109/ACCESS.2025.3599679
**Access:** full text (11 pages) read via text extraction. Pages 4, 6 and 10 (Tables 1-3, 6, 7 and Figure 1) were rendered
to images and read, because their tables are images.

* **Architecture:** ImageNet VGG16 as a **frozen** feature extractor → features reduced to **4 features** → **4-qubit** circuit
  (PennyLane `default.qubit`): angle embedding `RY(x_i)` with `θ_i = π·f_i` (Table 3, step 7) → parameterised `RX, RY, RZ` rotations
  ("strongly entangling layers") → **cyclic CNOT** chain `CNOT(0,1), CNOT(1,2), CNOT(2,3), CNOT(3,0)` → **Pauli-Z expectation** per
  qubit → dense layers with **Swish** activation → softmax (2 classes).
* **What is not specified:** how the VGG16 output (512 after GAP) is reduced to 4 features (Table 3 only says "extracted feature
  space"), the number of variational layers, the dense-layer widths, and the number of trainable quantum parameters.
* **Training (as stated):** 224×224, /255 normalisation, SGD momentum 0.9, lr 1e-3, ReduceLROnPlateau (0.2, patience 5), early
  stopping (patience 15), class weights, sparse categorical cross-entropy, ≤50 epochs, batch 64, Kaggle 2×T4.
  **Inconsistency:** Table 3 says "Binary Cross entropy loss and Adam optimizer".
* **Task simplification:** binary KL 0 vs KL 3+4. KL 1-2 (the hard, ambiguous grades) were **removed**. Train/val/test 3,216/461/913.
* **Reported:** hybrid+Swish 97.26% test accuracy, AUC 0.9928. Ablation (Table 7): VGG16+ReLU 96.06, VGG16+Swish 96.71,
  hybrid+ReLU 96.39, hybrid+Swish 97.26.
* **Limitations observed:** the quantum-vs-classical gain in Table 7 is **0.55 percentage points** (about 5 of 913 test images). It comes
  from a single run with no seeds, CIs or significance tests. The hybrid takes about 2× longer to train (10.43 vs 5.41 min). Patient-level
  separation of knee images is not described. The task was made easier by dropping KL 1-2. All quantum computation is simulated.
* **What we take from it:** the concrete 4-qubit circuit (angle `RY` encoding of π-scaled features, trainable `Rot` rotations, cyclic CNOT,
  Pauli-Z readout) as the starting configuration of the VQC. We also take its own ablation logic (hybrid vs. same backbone without the quantum layer),
  run here with matched feature dimensions, multiple seeds and paired statistics.

### 5.3 Transferability
Both papers study **knee X-ray** classification (single-label, 2 or 5 classes) on different datasets. Their accuracies **must not**
be used as expectations for six-label, highly imbalanced, slice-level ICH detection on CT. Neither paper shows that a quantum
layer outperforms a parameter-matched classical layer under a controlled protocol, and that is the question this project's Track C tests.
