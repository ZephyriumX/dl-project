# Experiment Log (template)

> **This file is a template. It contains no results.** Fill one entry per experiment *after* it has run. Copy numbers from
> `rsna_ich_artifacts/<MODE>/experiments/<exp_id>/result.json` and the notebook 04 tables. Do not round optimistically. Do not
> omit failed or skipped runs.

## Conventions
* Experiment IDs: `A_<transformer>`, `B_cnn_transformer_hybrid`, `B_cnn_reference_<cnn>`, `B_hybrid_fusion_<variant>`,
  `H_<model>_brain3`, `C_<head>_s<seed>`.
* Selection uses **validation** macro ROC-AUC only. Test metrics use thresholds frozen on validation.
* Always record the split-manifest hash (first 12 hex digits) and the config fingerprint.

## Study-level settings (fill once per MODE)
| field | value |
|---|---|
| MODE | |
| Split manifest hash | |
| Grouping level (patient / study_fallback) | |
| # patients / studies / images (train / val / test) | |
| Header scan complete? | |
| Kaggle GPU type / session count | |
| Package versions (torch / timm / pennylane / pydicom) | |

## Entry template
```
### <exp_id>  (<date>, MODE=<mode>)
- Status: succeeded | failed | skipped | not run
- Config fingerprint: <...>   Manifest hash: <...>
- Model / weights status: <timm name>, <pretrained | pretrained_local_file | random_init_fallback>
- Preprocessing: <channel_mode, image_size, fingerprint>
- Training: epochs run <n>/<max>, best epoch <k>, early stopped <yes/no>, batch <b>, AMP <dtype>, wall time <min>
- Validation: macro ROC-AUC <...>, per-label AUC <...>
- Held-out test (frozen val thresholds): macro ROC-AUC <...> [95% patient-bootstrap CI ...], macro AP <...>,
  any-AUC <...>, macro F1 <...>, Brier <...>, Hamming <...>
- Failures / warnings: <decode failures, OOM, fallbacks, NaN metrics and why>
- Notes: <anything a reader needs to interpret the result>
```

## Planned experiments (predeclared)
| ID | Question | Comparator | Status |
|---|---|---|---|
| A_deit_small / A_swin_tiny / A_maxvit_tiny | best standalone Transformer (A) | each other | |
| H_<best>_brain3 | single vs multi-window input (H) | A_<best> | |
| B_cnn_transformer_hybrid | CNN-Transformer hybrid (G) | A_<best>, B_cnn_reference | |
| B_hybrid_fusion_gap_only / cls_only | does the Transformer branch / CNN branch contribute (feature-module ablation) | B_cnn_transformer_hybrid | |
| C_linear_probe_s* | linear read-out of frozen features (reference) | — | |
| C_bottleneck_identity_s* | 4-d bottleneck without processing (B-control) | C_vqc | |
| C_bottleneck_mlp_s* | parameter-matched classical block (B) | C_vqc | |
| C_vqc_q4_l2_cyclic_s* | quantum head (C) | controls above | |
| C_vqc_q4_l2_noent_s* | no entanglement (D) | C_vqc_q4_l2_cyclic | |
| C_vqc_q4_l1/l4_cyclic_s* | circuit depth (E) | C_vqc_q4_l2_cyclic | |
| C_vqc_q2/q8_l2_cyclic_s*, C_bottleneck_mlp_k8_s* | qubit count (F) | k-matched controls | |

## Log
<!-- add entries below; newest last -->
