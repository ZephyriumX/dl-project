# Progress

## Completed stages (development environment)
| # | Stage | Result |
|---|---|---|
| 1 | Workspace inspection; the three supplied notebooks located and copied read-only to `reference_inputs/` (hash-verified identical) | done |
| 2 | Static inspection of all three notebooks (code, markdown, saved outputs) | done → `existing_work_audit.md` §1-4 |
| 3 | Inspection of both papers (full text; image-only tables rendered and read) | done → `existing_work_audit.md` §5 |
| 4 | Architecture decisions recorded | done → `architecture_notes.md` |
| 5 | Shared modules `src/` (data, models, quantum, training, evaluation, utils) | done |
| 6 | Configs with smoke / pilot / full modes | done |
| 7 | Six notebooks generated (no stored outputs) | done |
| 8 | Unit tests (60) | all passing locally |
| 9 | End-to-end execution of notebooks 00-05 on a synthetic DICOM dataset (smoke mode) | passing locally |
| 10 | Pretrained-weight download path (DeiT-Small, EfficientNet-B0 stem) and offline local-weights path (DeiT-Small) | verified locally |
| 11 | Documentation (README, audit, architecture, limitations, experiment-log template, status) | done |
| 12 | ZIP packaging and content verification | done (see implementation_status.md) |

## Remaining work (requires Kaggle + the real RSNA data; not done)
1. Run notebook 00 in `smoke`, then `pilot`/`full` on Kaggle with the competition data attached. Check discovery, the header-scan
   throughput and the label report (expected: duplicate rows in `stage_2_train.csv`, since 4,516,842 rows ≠ 6 × 752,803 images).
2. Run notebooks 01-02 (Track A/B) in `pilot`, then `full`. Several Kaggle sessions per mode will likely be needed, using the resume path.
3. Run notebook 03 (Track C) once a backbone exists. Then notebook 04 (analysis) and 05 (triage).
4. Fill `experiment_log.md` from the produced `result.json` files.
5. Optional: external validation set, 2.5-D context, multi-GPU training.

## Resumption note (for whoever continues this work)
Before continuing, inspect `implementation_status.md` and the workspace. Do not regenerate notebooks by hand: edit
`tools/build_notebooks.py` and run it. Do not regenerate the split manifest once models have been trained on it.
