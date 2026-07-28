"""Multi-camera stream synchronization (part of Deliverable 1, component 1).

Directly answers a gap in an earlier version of this prototype: a mock
generator that hands every camera identical frame indices demonstrates
multi-camera DATA, but not the actual frame-synchronization/jitter problem
the assignment names (Sec 4.3.1: "frame synchronization/jitter management
across multi-camera streams"). This module simulates that problem for real:

    Camera 1 (raw):  t=0.3,  t=10.4,          t=29.8, t=40.1, ...
    Camera 2 (raw):  t=-0.2, t=9.9,   t=19.7,  t=30.3,         ...  (frame at 40 lost)
    Camera 3 (raw):  t=0.1,           t=20.2,  t=29.9, t=40.4, ...  (frame at 10 lost)
                              |
                    synchronize_streams()
                              |
              aligned [camera, target_frame, feature] tensor
              + per-(camera, target_frame) availability mask

Two effects are modeled independently and compose:
  - **Jitter**: a camera's actual capture instant drifts around the nominal
    grid time (clock drift, encode/network latency) - `camera_jitter_std_seconds`.
  - **Frame loss**: a camera occasionally produces no sample at all for a
    given interval (dropped packet, decode failure, momentary link outage) -
    `camera_frame_drop_prob`.

`synchronize_streams` is the alignment layer: for each camera and each
target (nominal) timestamp, it takes the nearest raw sample within
`sync_tolerance_seconds`, exactly the "buffer a short rolling window per
camera stream, select nearest frame within a tolerance window" design
described in the report's AWS architecture section (Sec 4.3.1) - this
module is that design, implemented and tested, not just described.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RawSample:
    """One observation actually produced by a camera - may be jittered off
    the nominal grid. A dropped frame simply has no RawSample at all."""

    timestamp: float
    feature: np.ndarray


def generate_jittered_camera_stream(
    nominal_timestamps: np.ndarray,
    features_at_nominal: np.ndarray,
    jitter_std_seconds: float,
    drop_prob: float,
    rng: np.random.Generator,
) -> list[RawSample]:
    """Simulate one camera's raw, irregular stream from a clean nominal
    grid: each nominal slot is independently either lost entirely
    (probability `drop_prob`) or emitted at its nominal time plus Gaussian
    jitter. Returned sorted by timestamp (jitter can reorder samples near
    the drop-probability/jitter-magnitude boundary, so sort defensively -
    `synchronize_streams` assumes a time-ordered stream)."""
    stream: list[RawSample] = []
    for t_nominal, feature in zip(nominal_timestamps, features_at_nominal):
        if rng.random() < drop_prob:
            continue  # packet never arrives - no sample for this interval
        jitter = rng.normal(0.0, jitter_std_seconds)
        stream.append(RawSample(timestamp=float(t_nominal + jitter), feature=feature))
    stream.sort(key=lambda s: s.timestamp)
    return stream


def synchronize_streams(
    camera_streams: list[list[RawSample]],
    target_timestamps: np.ndarray,
    tolerance_seconds: float,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align `len(camera_streams)` independent, irregular raw streams onto a
    common `target_timestamps` grid.

    For each camera and each target time, picks the nearest raw sample
    within `tolerance_seconds`; if none exists (jitter pushed every nearby
    sample outside the tolerance window, or the frame was dropped), that
    (camera, target-frame) slot is marked unavailable and its feature slot
    is left zeroed - the same "absent -> zero features, excluded via mask"
    convention data.py already uses for whole-camera absence, now applied
    per-timestep too.

    Returns (aligned_features[num_cameras, num_targets, feature_dim],
    availability_mask[num_cameras, num_targets]).
    """
    num_cameras = len(camera_streams)
    num_targets = len(target_timestamps)
    aligned = np.zeros((num_cameras, num_targets, feature_dim), dtype=np.float32)
    available = np.zeros((num_cameras, num_targets), dtype=np.float32)

    for cam, stream in enumerate(camera_streams):
        if not stream:
            continue
        stream_times = np.array([s.timestamp for s in stream])
        # searchsorted gives the insertion point; the nearest sample is one
        # of the two neighbors around it, so check both. O(T log S) total
        # per camera - fine at this prototype's sequence lengths, and easy
        # to follow without a specialized nearest-neighbor structure.
        insert_idx = np.searchsorted(stream_times, target_timestamps)
        for j, t_target in enumerate(target_timestamps):
            candidates = [k for k in (insert_idx[j] - 1, insert_idx[j]) if 0 <= k < len(stream)]
            if not candidates:
                continue
            best = min(candidates, key=lambda k: abs(stream_times[k] - t_target))
            if abs(stream_times[best] - t_target) <= tolerance_seconds:
                aligned[cam, j, :] = stream[best].feature
                available[cam, j] = 1.0

    return aligned, available
