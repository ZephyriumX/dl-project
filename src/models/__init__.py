"""Model factories for the three tracks."""

from __future__ import annotations

from .cnn_transformer import CNNTransformerHybrid, build_cnn_transformer
from .hybrid_classifier import HEAD_KINDS, BackboneWithHead, FeatureHead
from .transformers import (TRANSFORMER_CANDIDATES, BackboneClassifier, ModelBuildRecord, build_backbone,
                           build_transformer_classifier, count_params)


def create_model(cfg: dict):
    """Create a model from a config dict. Returns (model, ModelBuildRecord) or (None, record) on failure.

    ``cfg['kind']``: ``"timm"`` (Track A / CNN references) or ``"cnn_transformer"`` (Track B).
    """
    kind = cfg.get("kind", "timm")
    if kind == "timm":
        return build_transformer_classifier(
            cfg["name"], pretrained=cfg.get("pretrained", True), weights_dir=cfg.get("weights_dir"),
            fallback=cfg.get("fallback", "random_init"), dropout=cfg.get("dropout", 0.2),
            drop_path_rate=cfg.get("drop_path_rate", 0.1), in_chans=cfg.get("in_chans", 3))
    if kind == "cnn_transformer":
        return build_cnn_transformer(cfg)
    raise ValueError(f"unknown model kind {kind!r}")


__all__ = [n for n in dir() if not n.startswith("_")]
