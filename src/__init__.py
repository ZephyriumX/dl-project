"""RSNA ICH research framework: classical, CNN-Transformer and quantum-classical models.

All notebooks import from this package; no notebook re-implements data loading, splitting,
preprocessing, metrics or checkpointing.
"""

__version__ = "1.0.0"

#: Canonical label order used everywhere (predictions, targets, metrics, submissions).
LABELS = (
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
    "any",
)
SUBTYPES = LABELS[:5]
NUM_LABELS = len(LABELS)
