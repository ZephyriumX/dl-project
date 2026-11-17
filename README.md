# RSNA Intracranial Hemorrhage (ICH) Research Framework

A research framework for multi-label intracranial hemorrhage classification from head CT slices using classical vision transformers, a CNN–Transformer hybrid, and a quantum–classical hybrid model.

> **Research status:** The project has been validated locally with static checks, unit tests, and end-to-end notebook runs on a **synthetic DICOM dataset**. It has **not yet been trained or evaluated on the real RSNA competition dataset**. No clinical performance claims are made. This is an experimental research project, not a diagnostic or clinical decision-support system.

## Overview

The framework studies six image-level labels from the RSNA dataset:

- Epidural hemorrhage
- Intraparenchymal hemorrhage
- Intraventricular hemorrhage
- Subarachnoid hemorrhage
- Subdural hemorrhage
- Any hemorrhage

It is organized into three modeling tracks:

| Track | Approach | Models / components |
|---|---|---|
| A | Classical vision-transformer benchmark | DeiT-Small, Swin-Tiny, MaxViT-Tiny; ViT/BEiT are optional where configured |
| B | CNN–Transformer hybrid | EfficientNet-B0 feature stem, Transformer encoder and fusion head; standalone CNN reference and fusion ablations |
| C | Quantum–classical hybrid | Variational quantum circuit (VQC) head on frozen classical features, compared with parameter-matched classical controls |

Additional components include patient-/study-aware data splitting, DICOM metadata processing, CT preprocessing, checkpointing and experiment tracking, multi-label evaluation, statistical comparisons, study-level aggregation, Grad-CAM visualizations, and submission formatting.

The VQC's default backend is an exact **classical PyTorch state-vector simulator**. It is not execution on quantum hardware, and the project makes no quantum-speedup claim.

## Repository layout

```text
RSNA_ICH_Research/
├── README.md
├── IMPLEMENTATION_REPORT.md
├── requirements.txt
├── environment.yml
├── configs/                 # Data, transformer, hybrid and quantum settings
├── notebooks/               # Six ordered research / analysis notebooks
├── src/
│   ├── data/                # Labels, DICOM metadata, preprocessing, splits, datasets
│   ├── models/              # Transformer and CNN–Transformer models
│   ├── quantum/             # VQC and PyTorch state-vector simulator
│   ├── training/             # Training, losses, checkpoints, experiment runners
│   ├── evaluation/           # Metrics, statistics, thresholds, triage, explainability
│   └── utils/                # Configuration, paths, runtime, reproducibility, I/O
├── tests/                   # Unit tests
├── tools/                   # Validation, synthetic data and notebook utilities
├── artifacts/               # Placeholder only; generated outputs are not committed
└── research_documentation/  # Architecture, audit, experiment and limitation notes
```

## Dataset

This project is designed for the **RSNA Intracranial Hemorrhage Detection** dataset. The dataset contains de-identified head CT DICOM images and image-level annotations for five hemorrhage subtypes plus an `any` label.

- **Kaggle competition and data page:** https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data
- **Competition rules:** https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/rules
- **RSNA challenge overview:** https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019
- **RSNA dataset description:** https://www.rsna.org/-/media/Files/RSNA/Education/AI%20resources%20and%20training/AI%20image%20challenge/RSNA-2019-Brain-Hemorrhage-Detection-Challenge-Dataset-Description.ashx
- **RSNA dataset documentation:** https://github.com/RSNA/AI-Challenge-Data
- **AWS Open Data Registry entry:** https://registry.opendata.aws/rsna-intracranial-hemorrhage-detection/

### Dataset access and restrictions

The dataset is **not included** in this repository. Obtain it directly through Kaggle or the official RSNA-approved access route and accept the applicable terms before use. The dataset is restricted to non-commercial use under its stated terms. Do not upload, mirror, redistribute, or commit the DICOM images, annotation files, or any portion of the competition data to this repository. Do not attempt to re-identify any person represented in the data.

The project expects the stage-2 layout used by the competition, including `stage_2_train.csv` and the corresponding DICOM image directory. Notebook 00 performs the data audit and checks the discovered data layout. See `configs/data_config.yaml` for path discovery and override settings.

## Research references

### Dataset and clinical-imaging task

1. Flanders, A. E. et al. **Construction of a Machine Learning Dataset through Collaboration: The RSNA 2019 Brain CT Hemorrhage Challenge.** *Radiology: Artificial Intelligence*, 2020. https://doi.org/10.1148/ryai.2020190211
2. RSNA. **Intracranial Hemorrhage Detection Challenge (2019).** https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019

### Model architecture references

3. Dosovitskiy, A. et al. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** https://arxiv.org/abs/2010.11929
4. Touvron, H. et al. **Training data-efficient image transformers & distillation through attention (DeiT).** https://arxiv.org/abs/2012.12877
5. Liu, Z. et al. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.** https://arxiv.org/abs/2103.14030
6. Tu, Z. et al. **MaxViT: Multi-Axis Vision Transformer.** https://arxiv.org/abs/2204.01697
7. Tan, M. and Le, Q. **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.** https://arxiv.org/abs/1905.11946

### Quantum–classical design background

8. Dong, Y. et al. **Classification of knee osteoarthritis based on quantum-to-classical transfer learning.** *Frontiers in Physics* (2023). https://doi.org/10.3389/fphy.2023.1212373
9. Tikariha, A. et al. **Osteoarthritis Classification Using Hybrid Quantum Convolutional Neural Network.** *IEEE Access* (2025). https://doi.org/10.1109/ACCESS.2025.3599679

These papers concern knee osteoarthritis imaging, not intracranial hemorrhage. They are included only as background for the hybrid quantum–classical design choices; their reported metrics are not transferable to this project's task. The project's `research_documentation/existing_work_audit.md` discusses their limitations and relevance.

### Implementation documentation

- PyTorch: https://pytorch.org/docs/stable/index.html
- `timm` model library: https://huggingface.co/docs/timm/index
- PennyLane: https://docs.pennylane.ai/
- PennyLane + PyTorch interface: https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html
- pydicom: https://pydicom.github.io/pydicom/stable/
- Kaggle notebooks: https://www.kaggle.com/docs/notebooks

These references provide background for the dataset and architectures. They do not imply that the models in this repository reproduce the results reported in those papers.

## Environment setup

Python 3.10+ is recommended. A CUDA-enabled PyTorch build and a compatible NVIDIA driver are recommended for GPU training. The exact PyTorch/CUDA installation command depends on the machine; use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) rather than assuming one CUDA wheel fits every system.

```bash
# Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RSNA_ICH_Research

# Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux alternative:
# python -m venv .venv
# source .venv/bin/activate

# Install PyTorch separately using the official selector, then:
pip install -r requirements.txt
```

PennyLane is used as an optional reference backend; the default VQC implementation uses the included PyTorch simulator. `requirements.txt` also includes development and notebook-validation dependencies.

## Running the notebooks on Kaggle

The supplied workflow is designed around Kaggle notebooks and Kaggle's competition dataset access.

1. Create a **private Kaggle Dataset** containing this repository's project files (or attach the repository files through your preferred Kaggle workflow). Do not include the RSNA competition data in that code package.
2. Create/import the six notebooks in `notebooks/` into Kaggle.
3. Attach the project-code dataset to each notebook.
4. Attach the RSNA Intracranial Hemorrhage Detection competition data to the notebooks that read images. Accept the competition rules first.
5. Enable a GPU accelerator for model-training notebooks. Enable internet only if you need to download pretrained weights or optional packages; for offline use, see the existing README history / implementation notes and `tools/export_timm_weights.py`.
6. Start with `MODE = "smoke"`. The notebooks support `smoke`, `pilot`, and `full` modes. Keep the mode consistent across the notebook sequence.
7. Run the notebooks in this order:

   `00_Data_Audit_and_Preprocessing` → `01_Classical_Transformer_Benchmark` → `02_CNN_Transformer_Hybrid` → `03_Quantum_Classical_Hybrid` → `04_Ablation_and_Comparative_Analysis` → `05_ICH_Triage_and_Visualization`

   Notebooks 01 and 02 can be run independently after notebook 00. Notebook 03 uses the selected classical backbone/features; notebook 04 compares experiment outputs; notebook 05 performs study-level triage and visualization.

8. Outputs are written to `/kaggle/working/rsna_ich_artifacts/<MODE>/`. Save notebook versions and attach earlier notebook outputs as inputs to downstream notebooks where required.

For path discovery and run overrides, see `configs/data_config.yaml` and the configuration cells at the start of each notebook. The project attempts to discover the RSNA data mount automatically; set the `rsna_root` override if discovery fails.

## Local validation

From the repository root:

```bash
# Run the unit tests
python -m pytest tests -q

# Validate required files, imports, configurations and notebook structure
python tools/validate_project.py

# Execute the notebook workflow against generated synthetic DICOM data
python tools/run_notebooks_synthetic.py
```

The synthetic workflow is intended to test software integration only. Its metrics are not meaningful model-performance results and must not be presented as such.

## Evaluation and reporting

The framework includes multi-label metrics, validation-based threshold selection, patient-level bootstrap utilities, paired comparisons, and study-level aggregation methods. Use the held-out test split only for final evaluation. Report the split strategy, class-wise metrics, confidence intervals, random seeds, model initialization/weight status, failed or skipped experiments, and compute environment. Do not select thresholds or tune models on the test set.

No real-data benchmark results are currently reported. Update `research_documentation/experiment_log.md` only after actual experiments have been run and verified.

## Limitations and responsible use

- This is a 2-D CT-slice classification research framework, not a validated clinical system.
- The current verification uses synthetic DICOM data; real-data training and Kaggle GPU execution remain to be completed.
- Internal validation does not establish external generalization across hospitals, scanners, or populations.
- Grad-CAM outputs are qualitative explanations, not pixel-level hemorrhage segmentations.
- The quantum circuit is simulated classically; no quantum hardware execution or quantum advantage is claimed.
- Clinical deployment would require rigorous external validation, clinical oversight, regulatory review, privacy/security controls, and workflow evaluation.

See `research_documentation/limitations.md` and `research_documentation/implementation_status.md` for the detailed status and limitations.

## Citation

If you use the RSNA dataset in academic or other permitted work, cite the dataset as required by its terms:

> Flanders AF, et al. The RSNA Brain CT Hemorrhage Dataset. *Radiology: Artificial Intelligence*. 2020;2(3). https://doi.org/10.1148/ryai.2020190211

Also cite the relevant architecture papers and software libraries used in your experiments.

## License

No license for this repository's source code is specified yet. Unless a license is added by the repository owner, do not assume that the code is available for reuse, modification, or redistribution. The RSNA dataset has separate terms and is not relicensed by this repository.
