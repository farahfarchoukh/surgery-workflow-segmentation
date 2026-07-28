"""Segmentation Timeline Generator (Deliverable 1, component 3).

Turns noisy frame-level model output into a clean, contiguous timeline of
`Segment(label, start, end)` objects. Three deterministic, inspectable
passes, each targeting a specific named failure mode:

  1. Majority filter - removes isolated single/few-frame label flips (the
     literal flicker that fragments the long "patient_present" phase).
  2. Minimum-duration-per-class merge - any segment shorter than its class's
     floor gets absorbed into its longer neighbor (kills residual
     "patient_present" fragmentation the filter alone doesn't catch).
  3. Transition-prior masking (PKI-style, cf. Czempiel et al./Cholec80
     practice) - a real OR case flows forward through phases and never
     legitimately reverts (see config.ALLOWED_TRANSITIONS); a predicted
     segment that would create an illegal transition (e.g.
     "operation" -> "patient_present") is absorbed into the preceding,
     already-established phase instead.

`viterbi_decode` is also implemented (globally-optimal DP smoothing using
the same transition prior as a soft cost) but is OFF by default
(`EvalConfig.use_viterbi=False`): it requires the full sequence (or a large
lookahead buffer) to backtrack from, which is a poor fit for a live/online
system that must emit a decision as frames arrive - the majority-filter +
min-duration + transition-mask pipeline above only ever looks at a small,
bounded window (`majority_filter_window // 2` frames of lookahead), a much
better match for the "AWS live inference" story in the report. Viterbi is
kept available as the deliberate, explainable alternative for an offline
reprocessing job (e.g. nightly hard-example re-scoring, see report Sec 4.1.1)
where full-sequence latency is fine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import PHASE_LABELS, EvalConfig


@dataclass
class Segment:
    label: int  # class index into PHASE_LABELS
    start: int  # inclusive frame index
    end: int  # exclusive frame index

    @property
    def duration(self) -> int:
        return self.end - self.start


def majority_filter(labels: np.ndarray, window: int) -> np.ndarray:
    """Centered sliding-window majority vote. Centered (not causal) on
    purpose: it introduces a small, BOUNDED lookahead of `window // 2`
    frames, which we then measure honestly via the boundary-latency metric
    in metrics.py rather than pretending it's zero-latency - a few frames
    of smoothing delay in exchange for removing flicker is the correct
    trade for this system (see report Sec 4.2.2)."""
    n = len(labels)
    half = window // 2
    num_classes = len(PHASE_LABELS)
    out = np.empty_like(labels)
    for t in range(n):
        lo, hi = max(0, t - half), min(n, t + half + 1)
        counts = np.bincount(labels[lo:hi], minlength=num_classes)
        out[t] = int(np.argmax(counts))
    return out


def frames_to_segments(labels: np.ndarray) -> list[Segment]:
    segments: list[Segment] = []
    start = 0
    n = len(labels)
    for t in range(1, n + 1):
        if t == n or labels[t] != labels[start]:
            segments.append(Segment(label=int(labels[start]), start=start, end=t))
            start = t
    return segments


def segments_to_frames(segments: list[Segment], seq_len: int) -> np.ndarray:
    out = np.zeros(seq_len, dtype=np.int64)
    for seg in segments:
        out[seg.start : seg.end] = seg.label
    return out


def coalesce_adjacent(segments: list[Segment]) -> list[Segment]:
    """Merge back-to-back segments that ended up with the same label after
    a merge/absorb pass - keeps the segment list a true minimal
    representation of the timeline."""
    if not segments:
        return segments
    out = [segments[0]]
    for seg in segments[1:]:
        if seg.label == out[-1].label and seg.start == out[-1].end:
            out[-1] = Segment(label=out[-1].label, start=out[-1].start, end=seg.end)
        else:
            out.append(seg)
    return out


def merge_short_segments(segments: list[Segment], min_duration_frames: dict) -> list[Segment]:
    """Any segment under its class's minimum duration is absorbed into the
    LONGER of its two neighbors (first/last segments only have one
    neighbor). Iterates to a fixed point since a merge can occasionally
    leave a newly-adjacent pair that itself needs coalescing."""
    segments = list(segments)
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for i, seg in enumerate(segments):
            floor = min_duration_frames.get(PHASE_LABELS[seg.label], 1)
            if seg.duration >= floor:
                continue
            left = segments[i - 1] if i > 0 else None
            right = segments[i + 1] if i < len(segments) - 1 else None
            if left is None:
                target, target_idx = right, i + 1
            elif right is None:
                target, target_idx = left, i - 1
            else:
                target, target_idx = (left, i - 1) if left.duration >= right.duration else (right, i + 1)

            merged = Segment(
                label=target.label,
                start=min(seg.start, target.start),
                end=max(seg.end, target.end),
            )
            drop = {i, target_idx}
            segments = sorted(
                [s for j, s in enumerate(segments) if j not in drop] + [merged],
                key=lambda s: s.start,
            )
            changed = True
            break  # segment list mutated - restart the scan
    return coalesce_adjacent(segments)


def enforce_transition_prior(segments: list[Segment], allowed: np.ndarray) -> list[Segment]:
    """Walk segments in order; a transition the domain forbids (see
    config.build_allowed_transition_matrix) gets absorbed into the
    preceding, already-accepted phase rather than starting a new one."""
    if not segments:
        return segments
    result = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        if allowed[prev.label, seg.label]:
            result.append(seg)
        else:
            result[-1] = Segment(label=prev.label, start=prev.start, end=seg.end)
    return coalesce_adjacent(result)


def _log_softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def build_transition_log_prior(allowed: np.ndarray, disallowed_penalty: float = -1e4) -> np.ndarray:
    prior = np.where(allowed, 0.0, disallowed_penalty)
    return prior.astype(np.float64)


def viterbi_decode(log_probs: np.ndarray, transition_log_prior: np.ndarray) -> np.ndarray:
    """Standard max-sum Viterbi DP: globally-optimal label sequence under
    per-frame emission log-probabilities plus a fixed transition log-cost.
    See module docstring for why this is implemented but off by default."""
    seq_len, num_classes = log_probs.shape
    dp = np.full((seq_len, num_classes), -np.inf)
    backptr = np.zeros((seq_len, num_classes), dtype=int)
    dp[0] = log_probs[0]
    for t in range(1, seq_len):
        scores = dp[t - 1][:, None] + transition_log_prior  # [prev_class, curr_class]
        backptr[t] = np.argmax(scores, axis=0)
        dp[t] = scores[backptr[t], np.arange(num_classes)] + log_probs[t]
    path = np.zeros(seq_len, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(seq_len - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path


def generate_timeline(
    frame_logits: np.ndarray,
    eval_cfg: EvalConfig,
    allowed_transitions: np.ndarray,
) -> list[Segment]:
    """Full pipeline: frame_logits [T, num_classes] -> clean Segment list.
    This is what evaluate.py and error_analysis.py call - the single entry
    point tying the three (or, with use_viterbi, two) passes together."""
    if eval_cfg.use_viterbi:
        log_probs = _log_softmax_np(frame_logits)
        prior = build_transition_log_prior(allowed_transitions)
        path = viterbi_decode(log_probs, prior)
        segments = frames_to_segments(path)
    else:
        labels = frame_logits.argmax(axis=-1)
        smoothed = majority_filter(labels, eval_cfg.majority_filter_window)
        segments = frames_to_segments(smoothed)

    segments = merge_short_segments(segments, eval_cfg.min_duration_frames)
    segments = enforce_transition_prior(segments, allowed_transitions)
    return segments
