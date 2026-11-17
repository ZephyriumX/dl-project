"""Models, VQC (both backends), heads, checkpointing, trainer, explainability."""

import math

import numpy as np
import pytest
import torch

from src.models import FeatureHead, build_backbone, build_transformer_classifier
from src.models.cnn_transformer import CNNTransformerHybrid
from src.quantum import VQCConfig, VQCLayer, circuit_resources, entangling_pairs
from src.quantum import torch_simulator as ts

try:
    import pennylane  # noqa: F401
    HAS_PL = True
except ImportError:
    HAS_PL = False


# ------------------------------------------------------------------------------------------ simulator
def test_cnot_truth_table():
    s = ts.zero_state(1, 2)
    s = ts._apply_1q(s, ts.rx_mat(torch.tensor(math.pi)), 0)  # |10> (up to phase)
    s = ts.cnot(s, 0, 1)
    p = (s.abs() ** 2).reshape(-1)
    assert torch.allclose(p, torch.tensor([0, 0, 0, 1.0]), atol=1e-6)


def test_single_qubit_analytic_expectation():
    cfg = VQCConfig(n_qubits=1, n_layers=1, rotation="ry", entanglement="none")
    layer = VQCLayer(cfg)
    layer.weights.data.fill_(0.3)
    x = torch.linspace(-2, 2, 9)[:, None]
    assert torch.allclose(layer(x)[:, 0], torch.cos(x[:, 0] + 0.3), atol=1e-5)


@pytest.mark.parametrize("ent", ["cyclic", "linear", "none"])
@pytest.mark.parametrize("reup", [False, True])
@pytest.mark.skipif(not HAS_PL, reason="PennyLane not installed")
def test_torch_backend_matches_pennylane(ent, reup):
    torch.manual_seed(0)
    c = dict(n_qubits=4, n_layers=2, entanglement=ent, data_reuploading=reup, hadamard_init=reup)
    a, b = VQCLayer(VQCConfig(**c)), VQCLayer(VQCConfig(**c, backend="pennylane"))
    b.weights.data = a.weights.data.clone()
    x1 = torch.randn(6, 4, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    ya, yb = a(x1), b(x2)
    assert torch.allclose(ya, yb, atol=1e-5)
    ya.pow(2).sum().backward()
    yb.pow(2).sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(a.weights.grad, b.weights.grad, atol=1e-5)


def test_vqc_forward_backward_and_resources():
    layer = VQCLayer(VQCConfig(n_qubits=4, n_layers=2))
    x = torch.randn(5, 4, requires_grad=True)
    y = layer(x)
    assert y.shape == (5, 4) and (y.abs() <= 1 + 1e-5).all()
    y.sum().backward()
    assert layer.weights.grad is not None and torch.isfinite(layer.weights.grad).all() and x.grad.abs().sum() > 0
    r = circuit_resources(layer.cfg)
    assert r["trainable_params"] == 24 and r["cnot_gates"] == 8
    assert entangling_pairs(4, "cyclic") == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert entangling_pairs(4, "none") == []


def test_vqc_ignores_autocast():
    layer = VQCLayer(VQCConfig(n_qubits=3, n_layers=1))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = layer(torch.randn(2, 3))
    assert y.dtype == torch.float32


def test_invalid_vqc_config():
    with pytest.raises(ValueError):
        VQCConfig(entanglement="full")


# ------------------------------------------------------------------------------------------ heads
@pytest.mark.parametrize("kind", ["linear_probe", "mlp", "bottleneck_identity", "bottleneck_mlp", "vqc"])
def test_feature_heads(kind):
    h = FeatureHead(kind, in_dim=32, k=4, vqc={"n_layers": 2})
    f = torch.randn(7, 32)
    out = h(f)
    assert out.shape == (7, 6)
    out.sum().backward()
    assert all(p.grad is not None for p in h.parameters() if p.requires_grad)
    pb = h.param_breakdown()
    assert pb["total"] > 0


def test_classical_control_is_parameter_matched():
    q = FeatureHead("vqc", in_dim=64, k=4, vqc={"n_layers": 2}).param_breakdown()
    c = FeatureHead("bottleneck_mlp", in_dim=64, k=4).param_breakdown()
    assert q["encoder"] == c["encoder"] and q["readout"] == c["readout"]
    assert abs(q["block"] - c["block"]) <= 4  # 24 vs 20


# ------------------------------------------------------------------------------------------ backbones
@pytest.mark.parametrize("name", ["deit_small", "swin_tiny", "maxvit_tiny"])
def test_transformer_forward_random_init(name):
    m, rec = build_transformer_classifier(name, pretrained=False)
    assert rec.status == "random_init" and not rec.is_pretrained and rec.num_features > 0
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(1, 3, 224, 224))
    assert y.shape == (1, 6)
    assert m.forward_features(torch.randn(1, 3, 224, 224)).shape == (1, rec.num_features)


def test_backbone_failure_and_fallback_records(monkeypatch):
    m, rec = build_backbone("definitely_not_a_model", pretrained=True)
    assert m is None and rec.status == "failed" and rec.error
    import timm

    real = timm.create_model

    def fake(name, pretrained=False, **kw):
        if pretrained:
            raise RuntimeError("offline")
        return real(name, pretrained=False, **kw)

    monkeypatch.setattr(timm, "create_model", fake)
    m, rec = build_backbone("deit_small", pretrained=True, fallback="random_init")
    assert m is not None and rec.status == "random_init_fallback" and not rec.is_pretrained
    m, rec = build_backbone("deit_small", pretrained=True, fallback="fail")
    assert m is None and rec.status == "failed"


@pytest.mark.parametrize("fusion", ["concat", "cls_only", "gap_only"])
def test_cnn_transformer_hybrid(fusion):
    m = CNNTransformerHybrid(pretrained=False, fusion=fusion, d_model=64, depth=1, heads=4)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    assert y.shape == (2, 6)
    f, att = m.forward_features(x, return_attention=True)
    assert f.shape[1] == m.num_features
    if fusion != "gap_only":
        assert att.shape == (2, 14, 14)
    y.sum().backward()


# ------------------------------------------------------------------------------------------ checkpoint + trainer
def test_checkpoint_roundtrip_and_guards(tmp_path):
    from src.training import (CheckpointConfigMismatchError, ManifestMismatchError, load_checkpoint,
                              save_checkpoint)

    m = FeatureHead("bottleneck_mlp", in_dim=8)
    opt = torch.optim.AdamW(m.parameters())
    meta = {"manifest_hash": "abc", "config_fingerprint": "f1"}
    p = save_checkpoint(tmp_path / "c.pt", model=m, meta=meta, optimizer=opt, epoch=3, best_metric=0.7)
    m2 = FeatureHead("bottleneck_mlp", in_dim=8)
    ck = load_checkpoint(p, m2, expected_manifest_hash="abc")
    assert ck["epoch"] == 3 and all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
    with pytest.raises(ManifestMismatchError):
        load_checkpoint(p, m2, expected_manifest_hash="other")
    with pytest.raises(CheckpointConfigMismatchError):
        save_checkpoint(p, model=m, meta={"manifest_hash": "abc", "config_fingerprint": "f2"})


def test_trainer_fit_resume_and_predictions(tmp_path):
    from src.data import FeatureDataset, make_loader
    from src.evaluation.predictions import load_predictions, predictions_frame, save_predictions
    from src.training import TrainConfig, Trainer, build_loss, predict

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16)).astype(np.float32)
    Y = (rng.random((200, 6)) < 0.3).astype(np.float32)
    X[:, :6] += 2 * Y  # learnable signal
    Y[:, 5] = Y[:, :5].max(1)
    ids = [f"I{i}" for i in range(200)]
    tr = make_loader(FeatureDataset(X[:150], Y[:150], ids[:150]), 32, True, 0, pin_memory=False)
    va = make_loader(FeatureDataset(X[150:], Y[150:], ids[150:]), 64, False, 0, pin_memory=False)
    meta = {"manifest_hash": "h", "config_fingerprint": "f"}
    cfg = TrainConfig(epochs=3, lr=1e-2, amp=False, early_stopping_patience=5, backbone_lr=None)
    loss, desc = build_loss({"kind": "bce", "pos_weight": "sqrt_inverse"}, Y[:150])
    head = FeatureHead("vqc", in_dim=16, k=4)
    res = Trainer(head, loss, cfg, tmp_path, meta, device="cpu").fit(tr, va)
    assert res.epochs_run == 3 and res.best_metric > 0.5 and (tmp_path / "best.pt").exists()
    # second call: completed run is detected and not retrained
    res2 = Trainer(FeatureHead("vqc", in_dim=16, k=4), loss, cfg, tmp_path, meta, device="cpu").fit(tr, va)
    assert res2.best_epoch == res.best_epoch
    out = predict(head, va, torch.device("cpu"), amp=False)
    df = predictions_frame(out["image_ids"], out["probs"], out["targets"])
    p = save_predictions(df, tmp_path / "pred.parquet", {"manifest_hash": "h", "experiment_id": "x", "split": "val"})
    back, meta2 = load_predictions(p, "h")
    assert back.shape == df.shape and meta2["split"] == "val"
    with pytest.raises(ValueError):
        load_predictions(p, "other")


def test_trainer_resumes_from_last_checkpoint(tmp_path):
    from src.data import FeatureDataset, make_loader
    from src.training import TrainConfig, Trainer, build_loss

    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(64, 8)).astype(np.float32), (rng.random((64, 6)) < 0.4).astype(np.float32)
    dl = make_loader(FeatureDataset(X, Y, [str(i) for i in range(64)]), 16, True, 0, pin_memory=False)
    meta = {"manifest_hash": "h", "config_fingerprint": "f"}
    loss, _ = build_loss({"kind": "bce", "pos_weight": "none"})
    Trainer(FeatureHead("mlp", in_dim=8), loss, TrainConfig(epochs=1, amp=False, backbone_lr=None), tmp_path, meta,
            device="cpu").fit(dl, dl)
    (tmp_path / "done.json").unlink()  # simulate interruption after epoch 1 of a 3-epoch run
    res = Trainer(FeatureHead("mlp", in_dim=8), loss, TrainConfig(epochs=3, amp=False, backbone_lr=None,
                                                                  early_stopping_patience=10), tmp_path,
                  meta, device="cpu").fit(dl, dl)
    assert res.resumed_from_epoch == 0 and res.epochs_run == 3


def test_gradcam_shapes():
    from src.evaluation.explainability import GradCAM, default_cam_target

    m = CNNTransformerHybrid(pretrained=False, d_model=32, depth=1, heads=4)
    layer, layout, prefix = default_cam_target(m)
    cam = GradCAM(m, layer, layout, prefix)
    h = cam(torch.randn(2, 3, 224, 224), 5)
    cam.remove()
    assert h.shape == (2, 224, 224) and float(h.max()) <= 1.0 + 1e-6

    vit, _ = build_transformer_classifier("deit_small", pretrained=False)
    layer, layout, prefix = default_cam_target(vit)
    assert layout == "tokens" and prefix >= 1
    cam = GradCAM(vit, layer, layout, prefix)
    assert cam(torch.randn(1, 3, 224, 224), 0).shape == (1, 224, 224)
    cam.remove()


def test_runtime_detection_and_settings():
    from src.utils.runtime import detect_runtime, suggest_settings

    rt = detect_runtime()
    st = suggest_settings(rt, "deit_small_patch16_224", requested_batch=8)
    assert st.batch_size <= 8 and st.device in ("cpu", "cuda")
    assert rt.cpu_count >= 1 and "torch" in rt.packages
