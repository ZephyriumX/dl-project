"""Shared data pipeline (labels, DICOM metadata, splitting, preprocessing, datasets)."""

from .csv_parser import (LabelReport, binarize_targets, label_distribution_table, parse_label_csv,
                         parse_submission_ids)
from .dataset import (CTAugment, FeatureDataset, RSNASliceDataset, SyntheticCTDataset, build_augment, collate,
                      make_loader)
from .dicom_metadata import (add_slice_order, load_metadata, metadata_integrity_report, read_header, scan_headers,
                             slice_position)
from .preprocessing import (CHANNEL_MODES, WINDOW_PRESETS, PreprocessConfig, dicom_to_hu, hu_to_channels,
                            load_dicom_channels, preprocessing_fingerprint)
from .splitting import (LeakageError, ManifestHashError, load_manifest, make_patient_split, manifest_hash,
                        overlap_report, save_manifest, split_summary, subset, verify_no_leakage)

__all__ = [n for n in dir() if not n.startswith("_")]
