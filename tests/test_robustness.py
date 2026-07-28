"""Production-readiness / adversarial tests: behavior under conditions the
happy-path tests in the other files don't exercise - config misuse,
corrupted artifacts, numerical divergence, extreme sync loss, and a
correctness check on the batched-inference refactor (evaluate.py). Every
test here asserts either "fails loudly with a clear error" or "degrades
gracefully (stays finite)" - never "silently produces something wrong."
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import DataConfig, EvalConfig, ExperimentConfig, ModelConfig, NUM_CLASSES, PHASE_LABELS, TrainConfig
from src.data import SurgeryPhaseDataset, collate_cases
from src.evaluate import CorruptedCheckpointError, compute_batch_logits, load_model
from src.model import PhaseSegmentationModel
from src.train import TrainingDivergedError, train

# ---------------------------------------------------------------------------
# Config validation actually fires (not just implemented, but wired up)
# ---------------------------------------------------------------------------


def test_config_rejects_inconsistent_camera_bounds():
    with pytest.raises(ValueError, match="max_cameras"):
        DataConfig(min_cameras=3, max_cameras=1)


def test_config_rejects_wrong_length_duration_weights():
    with pytest.raises(ValueError, match="phase_duration_weights"):
        DataConfig(phase_duration_weights=[1.0, 2.0])  # wrong length for NUM_CLASSES=5


def test_config_rejects_min_duration_missing_a_class():
    incomplete = {name: 10 for name in PHASE_LABELS if name != "operation"}
    with pytest.raises(ValueError, match="min_duration_frames"):
        EvalConfig(min_duration_frames=incomplete)


def test_config_rejects_invalid_dropout_probability():
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=1.5)


def test_config_rejects_zero_epochs():
    with pytest.raises(ValueError, match="epochs"):
        TrainConfig(epochs=0)


# ---------------------------------------------------------------------------
# Training divergence: fail loudly, never checkpoint garbage
# ---------------------------------------------------------------------------


def _tiny_config(**train_overrides) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(seed=1, feature_dim=8, seq_len=30, num_train_sequences=4, num_val_sequences=2),
        model=ModelConfig(fusion_dim=8, hidden_dim=8, stage1_layers=2, refine_layers=1, num_refine_stages=1),
        train=TrainConfig(epochs=3, batch_size=2, num_threads=1, **train_overrides),
        eval=EvalConfig(),
    )


def test_training_divergence_raises_and_writes_no_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "should_not_exist.pt"
    cfg = _tiny_config(lr=1e8)  # absurd LR forces a non-finite loss

    with pytest.raises(TrainingDivergedError, match="Non-finite loss"):
        train(cfg, checkpoint_path)

    assert not checkpoint_path.exists()


def test_normal_training_does_not_raise_and_writes_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    cfg = _tiny_config(lr=1e-3)
    train(cfg, checkpoint_path)
    assert checkpoint_path.exists()


# ---------------------------------------------------------------------------
# Checkpoint corruption: fail loudly with an actionable message
# ---------------------------------------------------------------------------


def test_corrupted_checkpoint_file_raises_clear_error(tmp_path):
    bad_path = tmp_path / "corrupted.pt"
    bad_path.write_text("not a real checkpoint")

    cfg = _tiny_config()
    with pytest.raises(CorruptedCheckpointError, match="Could not load checkpoint"):
        load_model(cfg, bad_path)


def test_structurally_malformed_checkpoint_raises_clear_error(tmp_path):
    """A validly-pickled file that's simply missing the keys this codebase
    expects (e.g. written by an incompatible/older version) - a different
    failure mode than outright corruption, should still be caught."""
    bad_path = tmp_path / "malformed.pt"
    torch.save({"unexpected_key": 123}, bad_path)

    cfg = _tiny_config()
    with pytest.raises(CorruptedCheckpointError, match="model_state"):
        load_model(cfg, bad_path)


# ---------------------------------------------------------------------------
# Batched inference correctness: chunking must not change results
# ---------------------------------------------------------------------------


def test_batch_chunking_does_not_change_results():
    """The whole point of compute_batch_logits's batch_size parameter is
    memory-bounded chunking without changing behavior - verify a small
    chunk size and one giant single-chunk call agree exactly."""
    cfg = _tiny_config()
    model = PhaseSegmentationModel(cfg.model, cfg.data.feature_dim, NUM_CLASSES)
    model.eval()

    ds = SurgeryPhaseDataset(cfg.data, num_sequences=10, base_seed=5)
    cases = [ds[i] for i in range(10)]

    logits_small_chunks = compute_batch_logits(model, cases, batch_size=3)
    logits_one_chunk = compute_batch_logits(model, cases, batch_size=100)

    assert np.array_equal(logits_small_chunks, logits_one_chunk)


def test_batched_and_sequential_single_case_inference_agree():
    """A case processed inside a batch of many must produce the same
    logits as when processed alone - i.e. no cross-contamination between
    batch items (a classic real bug class for any batched inference path)."""
    cfg = _tiny_config()
    model = PhaseSegmentationModel(cfg.model, cfg.data.feature_dim, NUM_CLASSES)
    model.eval()

    ds = SurgeryPhaseDataset(cfg.data, num_sequences=5, base_seed=6)
    cases = [ds[i] for i in range(5)]

    batched = compute_batch_logits(model, cases, batch_size=100)
    for i, case in enumerate(cases):
        alone = compute_batch_logits(model, [case], batch_size=100)[0]
        assert np.array_equal(batched[i], alone), f"case {i} differs when run alone vs. in a batch"


def test_compute_batch_logits_handles_empty_case_list():
    cfg = _tiny_config()
    model = PhaseSegmentationModel(cfg.model, cfg.data.feature_dim, NUM_CLASSES)
    model.eval()
    result = compute_batch_logits(model, [])
    assert result.shape == (0,)


# ---------------------------------------------------------------------------
# Extreme sync loss / degenerate data: must degrade gracefully, never crash
# ---------------------------------------------------------------------------


def test_near_total_sync_loss_stays_finite():
    """95% per-frame drop probability, 3x the tolerance window in jitter -
    a near-worst-case sync environment. The pipeline must still produce
    finite output end to end (data -> model -> loss), never NaN/crash,
    even though most frames will be genuinely unavailable."""
    cfg = DataConfig(
        seed=2,
        feature_dim=8,
        seq_len=50,
        min_cameras=1,
        max_cameras=2,
        camera_frame_drop_prob=0.95,
        camera_jitter_std_seconds=30.0,
        sync_tolerance_seconds=5.0,
    )
    ds = SurgeryPhaseDataset(cfg, num_sequences=5, base_seed=2)
    cases = [ds[i] for i in range(5)]
    batch = collate_cases(cases)

    assert torch.isfinite(batch["features"]).all()
    availability = batch["camera_mask"].mean().item()
    assert availability < 0.3, "expected severe (but not necessarily zero) availability at these parameters"

    model_cfg = ModelConfig(fusion_dim=8, hidden_dim=8, stage1_layers=2, refine_layers=1, num_refine_stages=1)
    model = PhaseSegmentationModel(model_cfg, cfg.feature_dim, NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        logits = model(batch["features"], batch["camera_mask"])[-1]
    assert torch.isfinite(logits).all()


def test_sequence_shorter_than_receptive_field_still_works():
    """The default config's receptive field (321 frames) exceeds its own
    240-frame sequence length already (Sec 4.2.2) - push further with a
    deliberately tiny seq_len to confirm the model has no hidden assumption
    that seq_len >= receptive_field (it shouldn't: causal convs just use
    whatever context is available, capped by the padding)."""
    cfg = DataConfig(seed=3, feature_dim=8, seq_len=5, min_cameras=1, max_cameras=2)
    ds = SurgeryPhaseDataset(cfg, num_sequences=3, base_seed=3)
    cases = [ds[i] for i in range(3)]
    batch = collate_cases(cases)

    model_cfg = ModelConfig(fusion_dim=8, hidden_dim=8, stage1_layers=4, refine_layers=3, num_refine_stages=1)
    model = PhaseSegmentationModel(model_cfg, cfg.feature_dim, NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        all_logits = model(batch["features"], batch["camera_mask"])
    for logits in all_logits:
        assert logits.shape == (3, 5, NUM_CLASSES)
        assert torch.isfinite(logits).all()


def test_single_camera_case_with_frequent_full_dropout_never_nans():
    """A single-camera case (no redundancy at all) combined with a high
    drop rate means many timesteps will have their only camera missing -
    i.e. CameraFusion's fully-missing-timestep fallback (tested in
    isolation in test_model.py) fires repeatedly here under realistic data
    generation, not just a hand-constructed mask."""
    cfg = DataConfig(
        seed=4, feature_dim=8, seq_len=60, min_cameras=1, max_cameras=1, camera_frame_drop_prob=0.6
    )
    ds = SurgeryPhaseDataset(cfg, num_sequences=8, base_seed=4)
    cases = [ds[i] for i in range(8)]
    batch = collate_cases(cases)

    model_cfg = ModelConfig(fusion_dim=8, hidden_dim=8, stage1_layers=2, refine_layers=1, num_refine_stages=1)
    model = PhaseSegmentationModel(model_cfg, cfg.feature_dim, NUM_CLASSES)
    model.train()  # also exercise view-dropout composing with sync loss
    logits = model(batch["features"], batch["camera_mask"])[-1]
    assert torch.isfinite(logits).all()
