"""Tests for src/postprocess.py: each pass's specific guarantee, checked in
isolation and then as a full pipeline, using hand-built label sequences
with known correct answers (not model output, so these are unaffected by
any future retraining)."""

import numpy as np

from src.config import PHASE_LABELS, ExperimentConfig, build_allowed_transition_matrix
from src.postprocess import (
    Segment,
    coalesce_adjacent,
    enforce_transition_prior,
    frames_to_segments,
    generate_timeline,
    majority_filter,
    merge_short_segments,
    segments_to_frames,
)

ALLOWED = build_allowed_transition_matrix()
PP = PHASE_LABELS.index("patient_present")
PREP = PHASE_LABELS.index("preparation")
OP = PHASE_LABELS.index("operation")
CLOSE = PHASE_LABELS.index("closing")
LEAVE = PHASE_LABELS.index("patient_leave")


def full_config():
    return ExperimentConfig.from_yaml("config/default.yaml")


def test_frames_to_segments_and_back_roundtrip():
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    segs = frames_to_segments(labels)
    assert [s.label for s in segs] == [0, 1, 2]
    assert [s.duration for s in segs] == [3, 2, 4]
    restored = segments_to_frames(segs, len(labels))
    assert (restored == labels).all()


def test_majority_filter_removes_single_frame_flip():
    labels = np.zeros(21, dtype=int)
    labels[10] = 1  # isolated 1-frame flip in the middle
    filtered = majority_filter(labels, window=9)
    assert filtered[10] == 0, "a single-frame flip should be outvoted by its neighborhood"
    assert (filtered == 0).all()


def test_min_duration_merge_eliminates_short_segment():
    # patient_present(50) -> operation(3, below its floor of 15) -> closing(50)
    segments = [Segment(PP, 0, 50), Segment(OP, 50, 53), Segment(CLOSE, 53, 103)]
    merged = merge_short_segments(segments, {"patient_present": 15, "operation": 15, "closing": 15})
    assert all(s.duration >= 15 for s in merged)
    assert OP not in [s.label for s in merged], "the short operation blip should have been absorbed"


def test_transition_prior_rejects_illegal_transition():
    # operation -> patient_present is not in ALLOWED (a case never reverts)
    segments = [Segment(OP, 0, 50), Segment(PP, 50, 100)]
    cleaned = enforce_transition_prior(segments, ALLOWED)
    assert len(cleaned) == 1
    assert cleaned[0].label == OP
    assert cleaned[0].end == 100


def test_coalesce_adjacent_merges_same_label():
    segments = [Segment(0, 0, 10), Segment(0, 10, 20), Segment(1, 20, 30)]
    coalesced = coalesce_adjacent(segments)
    assert len(coalesced) == 2
    assert coalesced[0].start == 0 and coalesced[0].end == 20


def test_full_pipeline_cleans_flicker_blip_and_illegal_reversion():
    seq_len, num_classes = 200, len(PHASE_LABELS)
    labels = np.zeros(seq_len, dtype=int)
    labels[0:60], labels[60:90], labels[90:150], labels[150:180], labels[180:200] = PP, PREP, OP, CLOSE, LEAVE
    labels[15] = OP  # 1-frame flicker
    labels[110:113] = CLOSE  # 3-frame spurious blip mid-operation

    logits = np.zeros((seq_len, num_classes))
    for t, label in enumerate(labels):
        logits[t, label] = 10.0

    segments = generate_timeline(logits, full_config().eval, ALLOWED)
    assert [PHASE_LABELS[s.label] for s in segments] == [
        "patient_present",
        "preparation",
        "operation",
        "closing",
        "patient_leave",
    ]
    for a, b in zip(segments, segments[1:]):
        assert ALLOWED[a.label, b.label]


def test_pipeline_is_idempotent():
    cfg = full_config()
    rng = np.random.default_rng(0)
    seq_len = cfg.data.seq_len
    labels = rng.integers(0, len(PHASE_LABELS), size=seq_len)
    logits = np.zeros((seq_len, len(PHASE_LABELS)))
    for t, label in enumerate(labels):
        logits[t, label] = 10.0

    first_pass = generate_timeline(logits, cfg.eval, ALLOWED)
    cleaned_frames = segments_to_frames(first_pass, seq_len)
    logits2 = np.zeros((seq_len, len(PHASE_LABELS)))
    for t, label in enumerate(cleaned_frames):
        logits2[t, label] = 10.0
    second_pass = generate_timeline(logits2, cfg.eval, ALLOWED)

    as_tuples = lambda segs: [(s.label, s.start, s.end) for s in segs]  # noqa: E731
    assert as_tuples(first_pass) == as_tuples(second_pass)
