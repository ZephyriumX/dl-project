# Limitations

## A. Verification status (what has *not* been shown)
1. **No real-data run.** The RSNA dataset (~459 GB) was not available in the development environment. Nothing has been trained
   or evaluated on real RSNA images. Every number produced so far came from a **synthetic** phantom dataset and is meaningless.
2. **No Kaggle GPU run.** The code was run locally (Windows, Python 3.12, PyTorch 2.6 + CUDA, RTX 4060 Laptop 8 GB). The Kaggle
   environment (Linux, different package versions, T4/P100 GPUs, read-only `/kaggle/input`, ~12 h sessions) has not been exercised.
   Kaggle-specific behaviour is implemented but unverified. This includes dataset mount discovery, DataLoader worker forking, throughput of the
   header scan on the Kaggle filesystem, and session-time budgets.
3. **Runtime/throughput estimates are not measured.** The batch-size memory table in `src/utils/runtime.py` is a starting point
   only (an OOM probe then confirms the batch). Epoch times on the full dataset are unknown.
4. Pretrained-weight loading was verified locally for DeiT-Small and the EfficientNet-B0 stem (download), and the offline
   local-file path was verified for DeiT-Small. Swin-Tiny and MaxViT-Tiny pretrained weights were not downloaded locally; only random-init
   builds and forward passes were tested.

## B. Data and labels
1. Labels are **slice-level** labels from the RSNA annotation process. They contain label noise and inter-reader variability that is
   not modelled here.
2. The patient-level split relies on DICOM `PatientID`. In the de-identified RSNA data these are pseudonymous IDs. If one real patient
   received several IDs, the split cannot detect it.
3. In `smoke` mode the header scan is partial. Studies are incomplete, so study-level (triage) results in that mode are not meaningful.
   The manifest records this.
4. Pilot mode subsamples patients. Its prevalence and difficulty can differ from the full dataset.
5. The held-out test set is an **internal** split of one dataset (single competition, limited scanner and site diversity). It is not external
   validation. No external dataset (e.g. CQ500) is integrated.
6. The competition `stage_2_test` set has no labels and is used only for inference.

## C. Modelling
1. All models are 2-D slice classifiers (one slice, 3 CT windows). Inter-slice context (2.5-D/3-D, sequence models) is out of scope,
   apart from the study-level aggregation of slice probabilities.
2. The CNN-Transformer hybrid is one technically motivated design. Its hyper-parameters (depth 2, d=256) were not tuned.
3. The default loss weighting (`sqrt` inverse prevalence, clipped) is a design decision taken before seeing results. It is not tuned.
4. Multi-GPU (2×T4) data parallelism is not implemented. One GPU is used.

## D. Quantum component
1. The VQC is **simulated exactly on classical hardware** (PyTorch state vector; PennyLane as reference). No quantum hardware, noise,
   shot sampling or hardware-specific compilation is involved. No computational quantum advantage can be inferred.
2. With 4 qubits the circuit has 24 trainable parameters and a 16-dimensional state. Any effect it has is that of a small, structured
   non-linear function of 4 projected features. The parameter-matched classical controls exist precisely to test whether that structure
   helps.
3. The default protocol trains heads on **frozen** backbone features. The backbone was fine-tuned with its own linear head, so its
   features are already well suited to a linear read-out. A 4-dimensional bottleneck (quantum or classical) is therefore expected to lose
   information relative to the linear probe. The primary comparison is VQC vs the matched classical bottleneck, not VQC vs the full model.
4. Seed-level tests use few seeds (3-5), so their power is low. Patient-level paired bootstrap tests are reported alongside.
5. The two source papers studied binary or 5-class knee X-ray classification. Their accuracies are not transferable to this task (see the audit).

## E. Statistics
1. The patient-cluster percentile bootstrap assumes patients are independent draws. Intervals can be unstable for labels with very few
   positive patients (epidural).
2. DeLong's test is included for reference only. It assumes independent slices and is anti-conservative here.
3. Multiple comparisons are Holm-corrected within each table, not across the whole study.

## F. Explainability and triage
1. Grad-CAM and attention maps are qualitative saliency. They are **not segmentations**. The RSNA data has no pixel annotations, so
   no localisation accuracy can be, or is, claimed.
2. Study-level "triage" is an aggregation of slice predictions for research purposes. It has no clinical validation, no prospective
   evaluation and no regulatory status, and must not be used for patient care.

## G. Reference material
1. The three supplied notebooks were inspected statically. They were not re-executed. Their saved results could not be verified.
2. Paper tables and figures that exist only as images were read from rendered pages. Fine-grained figure content (ROC plots,
   confusion matrices, training curves) was not digitised.
