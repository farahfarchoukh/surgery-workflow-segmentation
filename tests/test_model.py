"""Tests for src/model.py: shape/interface contracts, the causality
guarantee the report leans on, the parameter budget, and the view-dropout
safety net (never let softmax attention over cameras go all -inf)."""

import torch

from src.config import NUM_CLASSES, ModelConfig
from src.model import (
    CameraFusion,
    PhaseSegmentationModel,
    compute_receptive_field_frames,
    count_params,
)


def tiny_model_config(**overrides) -> ModelConfig:
    base = dict(fusion_dim=16, hidden_dim=16, kernel_size=3, stage1_layers=3, refine_layers=2, num_refine_stages=1)
    base.update(overrides)
    return ModelConfig(**base)


def test_forward_pass_shapes_for_each_camera_count():
    feature_dim, seq_len = 8, 40
    model = PhaseSegmentationModel(tiny_model_config(), feature_dim, NUM_CLASSES)
    model.eval()
    for num_cameras in (1, 2, 3):
        features = torch.randn(2, 3, seq_len, feature_dim)
        camera_mask = torch.zeros(2, 3, seq_len)  # per-timestep, see src/sync.py
        camera_mask[:, :num_cameras, :] = 1.0
        with torch.no_grad():
            all_logits = model(features, camera_mask)
        assert len(all_logits) == tiny_model_config().num_refine_stages + 1
        for logits in all_logits:
            assert logits.shape == (2, seq_len, NUM_CLASSES)


def test_param_budget_under_500k():
    """Enforces the <500K parameter claim made in the report - a shipped
    default-config assertion, not just a one-off print."""
    from src.config import ModelConfig as DefaultModelConfig

    model = PhaseSegmentationModel(DefaultModelConfig(), feature_dim=128, num_classes=NUM_CLASSES)
    assert count_params(model) < 500_000


def test_causality_future_frames_do_not_affect_past_outputs():
    """Directly verifies the 'causal/online-deployable' claim: perturbing
    only the tail of the sequence must leave every earlier-frame output
    bit-identical, since every conv in the stack is left-padded only."""
    feature_dim, seq_len = 8, 50
    model = PhaseSegmentationModel(tiny_model_config(), feature_dim, NUM_CLASSES)
    model.eval()

    features = torch.randn(1, 2, seq_len, feature_dim)
    camera_mask = torch.ones(1, 2, seq_len)

    with torch.no_grad():
        out1 = model(features, camera_mask)[-1]

        perturbed = features.clone()
        cutoff = seq_len - 10
        perturbed[:, :, cutoff:, :] += 50.0  # large perturbation, future frames only
        out2 = model(perturbed, camera_mask)[-1]

    assert torch.equal(out1[:, :cutoff, :], out2[:, :cutoff, :])


def test_receptive_field_formula_matches_empirical_behavior():
    """compute_receptive_field_frames is a claim about exactly how many
    input frames the model's output at time t depends on - don't just trust
    the arithmetic, find the TRUE farthest-back position that still affects
    output[query_t] and assert it matches the formula exactly.

    Deliberately a LINEAR scan, not a binary search: a first version of
    this test used binary search and got a wrong (too-small) empirical
    answer, because a dilated conv's dependency on its input is a SPARSE
    comb of positions {t, t-d, t-2d, ...}, not a contiguous range - the
    "does perturbing position p change the output" property is NOT
    monotonic in p, so binary search silently converges to the wrong
    boundary. A full scan has no such assumption to violate."""
    cfg = tiny_model_config()
    feature_dim = 8
    rf = compute_receptive_field_frames(cfg)

    seq_len = rf + 30
    query_t = rf + 15  # comfortably inside the sequence with margin on both sides
    model = PhaseSegmentationModel(cfg, feature_dim, NUM_CLASSES)
    model.eval()

    torch.manual_seed(0)
    features = torch.randn(1, 2, seq_len, feature_dim)
    camera_mask = torch.ones(1, 2, seq_len)

    with torch.no_grad():
        baseline = model(features, camera_mask)[-1][0, query_t]

    farthest_affecting_offset = None
    for offset in range(0, rf + 5):  # small margin past the claimed RF to also catch a formula UNDER-count
        perturbed = features.clone()
        perturbed[:, :, query_t - offset, :] += 5.0
        with torch.no_grad():
            out = model(perturbed, camera_mask)[-1][0, query_t]
        if not torch.equal(out, baseline):
            farthest_affecting_offset = offset

    empirical_rf = farthest_affecting_offset + 1
    assert empirical_rf == rf, f"formula says RF={rf} but empirical farthest-affecting offset gives RF={empirical_rf}"


def test_view_dropout_never_drops_every_camera():
    """With view_dropout_prob=1.0 every real camera would be masked absent
    the safety fallback - softmax over an all -inf row produces NaN, which
    is exactly what this test would catch if the fallback were removed."""
    torch.manual_seed(0)
    fusion = CameraFusion(feature_dim=8, fusion_dim=8, view_dropout_prob=1.0)
    fusion.train()
    features = torch.randn(4, 3, 5, 8)
    camera_mask = torch.ones(4, 3, 5)
    out = fusion(features, camera_mask)
    assert torch.isfinite(out).all()


def test_fusion_handles_fully_missing_timestep():
    """A real, not hypothetical, edge case with per-timestep sync masks
    (src/sync.py): every camera can genuinely lack data at the same instant
    (independent per-camera sync losses coinciding, or a single-camera case
    hitting one dropped frame). Without the fully-missing-timestep fallback
    in CameraFusion.forward, softmax over an all -inf row at that position
    would produce NaN and poison every downstream timestep through the
    causal convolutions."""
    torch.manual_seed(0)
    fusion = CameraFusion(feature_dim=8, fusion_dim=8, view_dropout_prob=0.0)
    fusion.eval()
    features = torch.randn(2, 3, 5, 8)
    camera_mask = torch.ones(2, 3, 5)
    camera_mask[0, :, 2] = 0.0  # every camera missing at t=2, batch item 0 only
    out = fusion(features, camera_mask)
    assert torch.isfinite(out).all()
    # unaffected timesteps/batch items should be untouched by the fallback
    camera_mask_all_ones = torch.ones(2, 3, 5)
    out_baseline = fusion(features, camera_mask_all_ones)
    assert torch.equal(out[1], out_baseline[1])  # batch item 1 had no missing timestep


def test_eval_mode_fusion_is_deterministic():
    """view_dropout only applies when self.training is True - eval mode
    must be bit-identical across repeated calls on the same input."""
    torch.manual_seed(0)
    fusion = CameraFusion(feature_dim=8, fusion_dim=8, view_dropout_prob=0.9)
    fusion.eval()
    features = torch.randn(2, 3, 5, 8)
    camera_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]).unsqueeze(-1).expand(2, 3, 5).contiguous()
    out1 = fusion(features, camera_mask)
    out2 = fusion(features, camera_mask)
    assert torch.equal(out1, out2)
