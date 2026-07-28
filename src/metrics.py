"""Dual Metric Stack (Deliverable 1, component 4).

MODEL-QUALITY metrics answer "is the segmentation topologically correct?":
frame accuracy is included but explicitly flagged as insufficient ALONE - a
model predicting the majority class for an entire long "patient_present"
video scores high frame accuracy while being clinically useless. Segmental
F1@{10,25,50} IoU and the edit score are the standard action-segmentation
metrics (Cholec80/Breakfast/50Salads convention, also used by MS-TCN) that
actually expose over-segmentation/fragmentation - frame accuracy can't see
it.

PRODUCT/CLINICAL-QUALITY metrics answer "does this matter to the OR?":
phase-duration error (feeds scheduling/turnover analytics), boundary
detection latency (does an alert fire fast enough to be useful), and a
cost-weighted false-positive/false-negative score with hysteresis debounce
(a false "operation started" alert is operationally worse than a
2-second-late true one - see report Sec 1.4 for the OR Black Box(R)
precedent this framing is based on).

Every function here operates on `Segment` objects from postprocess.py or
plain frame-label arrays - no coupling to the model or training code, so
this file is independently unit-testable against hand-built toy sequences
(see tests/test_metrics.py) with a KNOWN, hand-computed answer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from src.config import NUM_CLASSES, PHASE_LABELS, EvalConfig
from src.postprocess import Segment

# ---------------------------------------------------------------------------
# Model-quality metrics
# ---------------------------------------------------------------------------


def frame_accuracy(pred_frames: np.ndarray, gt_frames: np.ndarray) -> float:
    return float((pred_frames == gt_frames).mean())


def frame_accuracy_by_class(pred_frames: np.ndarray, gt_frames: np.ndarray) -> dict:
    """Per-class frame accuracy - more locally sensitive than segmental F1,
    which only requires overall segment-level IoU overlap and can therefore
    stay high even when a chunk of frames INSIDE a segment are wrong (see
    error_analysis.py, which reports both: frame accuracy is what reveals
    the raw model's degradation under injected noise, segmental F1 is what
    shows how much of that degradation postprocessing recovers)."""
    result = {}
    for idx, name in enumerate(PHASE_LABELS):
        class_mask = gt_frames == idx
        result[name] = float((pred_frames[class_mask] == gt_frames[class_mask]).mean()) if class_mask.sum() > 0 else None
    return result


def segment_iou(a: Segment, b: Segment) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def segmental_f1(pred_segments: list[Segment], gt_segments: list[Segment], threshold: float) -> dict:
    """Standard action-segmentation F1@IoU: greedily match each predicted
    segment to the best not-yet-matched ground-truth segment of the SAME
    class; a match counts as TP only if IoU >= threshold. Unmatched preds
    are FP, unmatched ground truth is FN."""
    gt_matched = [False] * len(gt_segments)
    tp = 0
    fp = 0
    for p in pred_segments:
        best_iou, best_idx = 0.0, -1
        for i, g in enumerate(gt_segments):
            if gt_matched[i] or g.label != p.label:
                continue
            iou = segment_iou(p, g)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx >= 0 and best_iou >= threshold:
            tp += 1
            gt_matched[best_idx] = True
        else:
            fp += 1
    fn = gt_matched.count(False)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def segmental_f1_by_class(pred_segments: list[Segment], gt_segments: list[Segment], threshold: float) -> dict:
    """Per-class F1@threshold - the building block error_analysis.py uses
    to show that 'patient_present'/'operation' degrade more than other
    classes under injected noise."""
    result = {}
    for c in range(NUM_CLASSES):
        pred_c = [s for s in pred_segments if s.label == c]
        gt_c = [s for s in gt_segments if s.label == c]
        result[PHASE_LABELS[c]] = segmental_f1(pred_c, gt_c, threshold)
    return result


def _levenshtein(a: list[int], b: list[int]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def edit_score(pred_segments: list[Segment], gt_segments: list[Segment]) -> float:
    """Levenshtein distance over the SEQUENCE OF SEGMENT LABELS (not frames),
    normalized to 0-100. This is the metric that most directly exposes
    over-segmentation: a model that's frame-accurate but produces spurious
    extra segments scores well on frame accuracy yet poorly here, because
    the segment-label sequence itself is wrong length/order."""
    pred_labels = [s.label for s in pred_segments]
    gt_labels = [s.label for s in gt_segments]
    dist = _levenshtein(pred_labels, gt_labels)
    max_len = max(len(pred_labels), len(gt_labels), 1)
    return (1.0 - dist / max_len) * 100.0


def segment_count_by_class(segments: list[Segment]) -> dict:
    counts = {name: 0 for name in PHASE_LABELS}
    for s in segments:
        counts[PHASE_LABELS[s.label]] += 1
    return counts


# ---------------------------------------------------------------------------
# Product / clinical-quality metrics
# ---------------------------------------------------------------------------


def phase_duration_error_seconds(
    pred_segments: list[Segment], gt_segments: list[Segment], seconds_per_frame: float
) -> dict:
    """|predicted total duration - ground truth total duration| per class,
    in seconds. This is what OR scheduling/turnover-time analytics would
    actually consume - not frame accuracy - so it's reported as its own
    first-class metric, not derived after the fact."""
    pred_totals: dict = defaultdict(int)
    gt_totals: dict = defaultdict(int)
    for s in pred_segments:
        pred_totals[s.label] += s.duration
    for s in gt_segments:
        gt_totals[s.label] += s.duration
    return {
        name: abs(pred_totals.get(idx, 0) - gt_totals.get(idx, 0)) * seconds_per_frame
        for idx, name in enumerate(PHASE_LABELS)
    }


def boundary_detection_latency(
    pred_segments: list[Segment], gt_segments: list[Segment]
) -> tuple[list[int], int]:
    """For each true phase transition, find the nearest PREDICTED transition
    of the same class-pair at or after the true boundary (causal delay only
    - a prediction that fires BEFORE the true transition doesn't count,
    since a live system can't know the future either). Returns
    (delays_in_frames, num_missed) - missed transitions get no delay value
    and are reported separately rather than silently dropped."""
    gt_transitions = [
        (gt_segments[i].end, gt_segments[i].label, gt_segments[i + 1].label) for i in range(len(gt_segments) - 1)
    ]
    pred_transitions = [
        (pred_segments[i].end, pred_segments[i].label, pred_segments[i + 1].label)
        for i in range(len(pred_segments) - 1)
    ]
    delays: list[int] = []
    missed = 0
    for t_frame, from_c, to_c in gt_transitions:
        candidates = [pt for pt in pred_transitions if pt[1] == from_c and pt[2] == to_c and pt[0] >= t_frame]
        if candidates:
            nearest = min(candidates, key=lambda pt: pt[0] - t_frame)
            delays.append(nearest[0] - t_frame)
        else:
            missed += 1
    return delays, missed


def cost_weighted_transition_score(
    pred_segments: list[Segment],
    gt_segments: list[Segment],
    hysteresis_frames: int,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict:
    """Models the real cost asymmetry: a spurious "operation started" alert
    (false positive) is operationally more disruptive than a late/missed one
    (false negative) - see report Sec 1.4. Hysteresis: a predicted segment
    only "fires" an alert if it's at least `hysteresis_frames` long (shorter
    ones are suppressed as noise, trading a little latency for far fewer
    false alerts - the same debounce idea a live system would implement)."""
    tolerance = 2 * hysteresis_frames
    gt_events = [(g.label, g.start) for g in gt_segments[1:]]
    pred_events = [(p.label, p.start) for p in pred_segments[1:] if p.duration >= hysteresis_frames]

    matched_pred = [False] * len(pred_events)
    fn = 0
    for label, t in gt_events:
        found = False
        for i, (pl, pt) in enumerate(pred_events):
            if matched_pred[i] or pl != label or abs(pt - t) > tolerance:
                continue
            matched_pred[i] = True
            found = True
            break
        if not found:
            fn += 1
    fp = matched_pred.count(False)
    cost = fp * false_positive_cost + fn * false_negative_cost
    return {"false_positive_events": fp, "false_negative_events": fn, "cost": cost}


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


@dataclass
class MetricsReport:
    frame_acc: float
    segmental_f1: dict = field(default_factory=dict)  # {threshold_str: {precision,recall,f1,...}}
    edit_score: float = 0.0
    phase_duration_error_seconds: dict = field(default_factory=dict)
    boundary_latency_median_frames: float | None = None
    boundary_latency_p95_frames: float | None = None
    boundary_latency_missed: int = 0
    cost_weighted: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_all_metrics(
    pred_frames: np.ndarray,
    gt_frames: np.ndarray,
    pred_segments: list[Segment],
    gt_segments: list[Segment],
    eval_cfg: EvalConfig,
    seconds_per_frame: float,
) -> MetricsReport:
    """One call, one structured result - evaluate.py and error_analysis.py
    both call exactly this, so they can never accidentally compute metrics
    two different ways."""
    seg_f1 = {
        f"f1@{int(t * 100)}": segmental_f1(pred_segments, gt_segments, t) for t in eval_cfg.iou_thresholds
    }
    delays, missed = boundary_detection_latency(pred_segments, gt_segments)
    median_delay = float(np.median(delays)) if delays else None
    p95_delay = float(np.percentile(delays, 95)) if delays else None

    return MetricsReport(
        frame_acc=frame_accuracy(pred_frames, gt_frames),
        segmental_f1=seg_f1,
        edit_score=edit_score(pred_segments, gt_segments),
        phase_duration_error_seconds=phase_duration_error_seconds(pred_segments, gt_segments, seconds_per_frame),
        boundary_latency_median_frames=median_delay,
        boundary_latency_p95_frames=p95_delay,
        boundary_latency_missed=missed,
        cost_weighted=cost_weighted_transition_score(
            pred_segments, gt_segments, eval_cfg.hysteresis_frames, eval_cfg.false_positive_cost, eval_cfg.false_negative_cost
        ),
    )
