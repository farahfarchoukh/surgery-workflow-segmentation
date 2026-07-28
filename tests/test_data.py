"""Tests for src/data.py: shape/interface contracts every downstream module
relies on, plus the two structural properties the mock generator claims to
have (reproducibility, real-but-noisy class signal)."""

import numpy as np

from src.config import NUM_CLASSES, DataConfig
from src.data import SurgeryPhaseDataset, collate_cases, generate_class_prototypes


def small_config(**overrides) -> DataConfig:
    base = dict(seed=7, feature_dim=16, seq_len=100, min_cameras=1, max_cameras=3)
    base.update(overrides)
    return DataConfig(**base)


def test_shapes_match_config():
    cfg = small_config()
    ds = SurgeryPhaseDataset(cfg, num_sequences=5)
    case = ds[0]
    assert case.features.shape == (cfg.max_cameras, cfg.seq_len, cfg.feature_dim)
    assert case.camera_mask.shape == (cfg.max_cameras,)
    assert case.labels.shape == (cfg.seq_len,)


def test_label_range_and_segment_sum():
    cfg = small_config()
    ds = SurgeryPhaseDataset(cfg, num_sequences=5)
    case = ds[0]
    assert case.labels.min().item() >= 0
    assert case.labels.max().item() < NUM_CLASSES
    assert case.labels.shape[0] == cfg.seq_len  # durations must sum exactly to seq_len


def test_camera_count_within_configured_range():
    cfg = small_config(min_cameras=1, max_cameras=3)
    ds = SurgeryPhaseDataset(cfg, num_sequences=20)
    counts = [int(ds[i].camera_mask.sum().item()) for i in range(len(ds))]
    assert min(counts) >= 1
    assert max(counts) <= 3
    assert len(set(counts)) > 1, "expected camera count to vary across cases"


def test_reproducibility_same_index_same_case():
    cfg = small_config()
    ds = SurgeryPhaseDataset(cfg, num_sequences=5)
    case_a = ds[2]
    case_b = ds[2]
    assert (case_a.features == case_b.features).all()
    assert (case_a.labels == case_b.labels).all()
    assert (case_a.camera_mask == case_b.camera_mask).all()


def test_different_base_seed_gives_different_cases():
    cfg = small_config()
    ds_a = SurgeryPhaseDataset(cfg, num_sequences=5, base_seed=1)
    ds_b = SurgeryPhaseDataset(cfg, num_sequences=5, base_seed=2)
    assert not (ds_a[0].labels == ds_b[0].labels).all().item() or not (
        ds_a[0].features == ds_b[0].features
    ).all().item()


def test_classes_are_statistically_distinguishable():
    """Sanity check that the mock isn't pure noise: mean feature vectors
    for two different classes should be measurably different, i.e. a model
    has real signal to learn from (see report Sec 1.1)."""
    prototypes = generate_class_prototypes(feature_dim=16, seed=7)
    dists = []
    for i in range(NUM_CLASSES):
        for j in range(i + 1, NUM_CLASSES):
            dists.append(np.linalg.norm(prototypes[i] - prototypes[j]))
    assert min(dists) > 1.0, "class prototypes should be well-separated in feature space"


def test_collate_cases_batches_cleanly():
    cfg = small_config()
    ds = SurgeryPhaseDataset(cfg, num_sequences=4)
    batch = collate_cases([ds[i] for i in range(4)])
    assert batch["features"].shape == (4, cfg.max_cameras, cfg.seq_len, cfg.feature_dim)
    assert batch["camera_mask"].shape == (4, cfg.max_cameras)
    assert batch["labels"].shape == (4, cfg.seq_len)
