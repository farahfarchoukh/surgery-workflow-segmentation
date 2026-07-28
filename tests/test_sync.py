"""Tests for src/sync.py: the jitter/frame-loss simulation and the
alignment layer that recovers a per-timestep availability mask from it."""

import numpy as np

from src.sync import generate_jittered_camera_stream, synchronize_streams


def test_zero_jitter_zero_drop_recovers_everything_exactly():
    """With no jitter and no drop probability, every nominal frame should
    survive synchronization and land on the exact matching feature -
    the degenerate case that proves the alignment logic itself is correct
    before jitter/loss are layered on top."""
    rng = np.random.default_rng(0)
    seq_len, feature_dim = 30, 4
    nominal = np.arange(seq_len, dtype=np.float64) * 10.0
    features = np.stack([np.full(feature_dim, i, dtype=np.float32) for i in range(seq_len)])

    stream = generate_jittered_camera_stream(nominal, features, jitter_std_seconds=0.0, drop_prob=0.0, rng=rng)
    assert len(stream) == seq_len

    aligned, available = synchronize_streams([stream], nominal, tolerance_seconds=5.0, feature_dim=feature_dim)
    assert available.shape == (1, seq_len)
    assert available.sum() == seq_len
    assert np.allclose(aligned[0], features)


def test_full_drop_probability_yields_no_available_frames():
    rng = np.random.default_rng(1)
    seq_len, feature_dim = 20, 4
    nominal = np.arange(seq_len, dtype=np.float64) * 10.0
    features = np.random.default_rng(2).normal(size=(seq_len, feature_dim)).astype(np.float32)

    stream = generate_jittered_camera_stream(nominal, features, jitter_std_seconds=1.0, drop_prob=1.0, rng=rng)
    assert stream == []

    aligned, available = synchronize_streams([stream], nominal, tolerance_seconds=5.0, feature_dim=feature_dim)
    assert available.sum() == 0
    assert (aligned == 0).all()


def test_availability_rate_roughly_matches_drop_probability():
    """A statistical, not exact, check - jitter can additionally push a
    non-dropped sample outside the tolerance window, so availability is
    expected to be SOMEWHAT below (1 - drop_prob), never above it."""
    rng = np.random.default_rng(3)
    seq_len, feature_dim = 500, 4
    nominal = np.arange(seq_len, dtype=np.float64) * 10.0
    features = rng.normal(size=(seq_len, feature_dim)).astype(np.float32)
    drop_prob = 0.2

    stream = generate_jittered_camera_stream(nominal, features, jitter_std_seconds=1.0, drop_prob=drop_prob, rng=rng)
    _, available = synchronize_streams([stream], nominal, tolerance_seconds=5.0, feature_dim=feature_dim)

    rate = available.mean()
    assert rate < (1 - drop_prob) + 0.02  # never above the drop-free ceiling (plus float slack)
    assert rate > (1 - drop_prob) - 0.25  # jitter loss shouldn't be implausibly large at these parameters


def test_alignment_picks_the_nearest_sample_within_tolerance():
    """Two candidate samples straddle a target timestamp; synchronize_streams
    must pick the nearer one, and must reject a sample outside tolerance
    even if it's the only candidate."""
    from src.sync import RawSample

    feature_dim = 2
    near = RawSample(timestamp=9.0, feature=np.array([1.0, 1.0], dtype=np.float32))
    far = RawSample(timestamp=3.0, feature=np.array([9.0, 9.0], dtype=np.float32))
    stream = [far, near]  # unsorted on purpose - synchronize_streams shouldn't assume caller pre-sorted

    stream.sort(key=lambda s: s.timestamp)  # (generate_jittered_camera_stream sorts; do it here explicitly too)
    aligned, available = synchronize_streams([stream], np.array([10.0]), tolerance_seconds=2.0, feature_dim=feature_dim)
    assert available[0, 0] == 1.0
    assert np.allclose(aligned[0, 0], [1.0, 1.0])  # picked `near` (dist=1), not `far` (dist=7, also out of tolerance)

    aligned2, available2 = synchronize_streams([stream], np.array([50.0]), tolerance_seconds=2.0, feature_dim=feature_dim)
    assert available2[0, 0] == 0.0  # nothing within tolerance of t=50
    assert (aligned2[0, 0] == 0).all()


def test_multi_camera_streams_are_independent():
    rng = np.random.default_rng(4)
    seq_len, feature_dim = 50, 3
    nominal = np.arange(seq_len, dtype=np.float64) * 10.0
    features = rng.normal(size=(seq_len, feature_dim)).astype(np.float32)

    reliable = generate_jittered_camera_stream(nominal, features, jitter_std_seconds=0.5, drop_prob=0.01, rng=rng)
    unreliable = generate_jittered_camera_stream(nominal, features, jitter_std_seconds=0.5, drop_prob=0.6, rng=rng)

    aligned, available = synchronize_streams([reliable, unreliable], nominal, tolerance_seconds=3.0, feature_dim=feature_dim)
    assert available[0].mean() > available[1].mean(), "the low-drop camera should be available more often"
    assert aligned.shape == (2, seq_len, feature_dim)
