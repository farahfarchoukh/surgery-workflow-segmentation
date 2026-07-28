"""Tests for src/metrics.py against hand-built toy sequences with a known,
hand-computed answer - the single most reviewer-trust-critical test file,
since it proves the metrics aren't just "run and eyeball" (see plan)."""

from src.config import ExperimentConfig
from src.metrics import compute_all_metrics, edit_score, frame_accuracy_by_class, segmental_f1
from src.postprocess import Segment, segments_to_frames


def eval_cfg():
    return ExperimentConfig.from_yaml("config/default.yaml").eval


def test_perfect_match_scores_maximally_on_every_metric():
    segments = [Segment(0, 0, 60), Segment(1, 60, 90), Segment(2, 90, 150), Segment(3, 150, 180), Segment(4, 180, 200)]
    gt_frames = segments_to_frames(segments, 200)
    pred_frames = segments_to_frames(segments, 200)

    report = compute_all_metrics(pred_frames, gt_frames, segments, segments, eval_cfg(), seconds_per_frame=10.0)

    assert report.frame_acc == 1.0
    assert report.segmental_f1["f1@50"]["f1"] == 1.0
    assert report.edit_score == 100.0
    assert all(v == 0 for v in report.phase_duration_error_seconds.values())
    assert report.boundary_latency_median_frames == 0.0
    assert report.boundary_latency_missed == 0
    assert report.cost_weighted["cost"] == 0.0


def test_segmental_f1_known_answer_for_a_split_segment():
    """gt: one 100-frame class-0 segment, one 100-frame class-2 segment.
    pred: class-0 segment split into two (over-segmentation).
    First split half has 50% IoU with gt (a match at threshold<=0.5);
    the second half is then an unmatched FP since gt-0 is already claimed.
    tp=2, fp=1, fn=0 -> precision=2/3, recall=1.0, f1=0.8 (hand-computed)."""
    gt = [Segment(0, 0, 100), Segment(2, 100, 200)]
    pred = [Segment(0, 0, 50), Segment(0, 50, 100), Segment(2, 100, 200)]

    result = segmental_f1(pred, gt, threshold=0.5)

    assert result == {"precision": 2 / 3, "recall": 1.0, "f1": 0.8, "tp": 2, "fp": 1, "fn": 0}


def test_edit_score_known_answer():
    """pred label sequence [0,0,2] vs gt [0,2]: Levenshtein distance = 1
    (delete one 0). Normalized: 1 - 1/max(3,2) = 1 - 1/3 -> 66.67."""
    gt = [Segment(0, 0, 100), Segment(2, 100, 200)]
    pred = [Segment(0, 0, 50), Segment(0, 50, 100), Segment(2, 100, 200)]

    score = edit_score(pred, gt)

    assert abs(score - (1 - 1 / 3) * 100) < 1e-9


def test_frame_accuracy_by_class_isolates_a_single_wrong_class():
    import numpy as np

    gt = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 0, 1, 0, 0])  # class 1 predicted wrong for 2/3 of its frames; class 0 perfect

    per_class = frame_accuracy_by_class(pred, gt)

    assert per_class["patient_present"] == 1.0
    assert abs(per_class["preparation"] - (1 / 3)) < 1e-9


def test_boundary_latency_only_counts_causal_delay():
    """A predicted transition that fires BEFORE the true boundary must not
    count as a (negative) latency - it should be treated as a miss, since a
    live system can't act on a transition it hasn't seen yet."""
    gt = [Segment(0, 0, 100), Segment(1, 100, 200)]
    pred_late = [Segment(0, 0, 105), Segment(1, 105, 200)]  # fires 5 frames late
    pred_early = [Segment(0, 0, 90), Segment(1, 90, 200)]  # fires 10 frames early

    from src.metrics import boundary_detection_latency

    delays_late, missed_late = boundary_detection_latency(pred_late, gt)
    assert delays_late == [5]
    assert missed_late == 0

    delays_early, missed_early = boundary_detection_latency(pred_early, gt)
    assert delays_early == []
    assert missed_early == 1
