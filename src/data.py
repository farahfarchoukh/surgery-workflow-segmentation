"""Ingestion & Feature Stubs (Deliverable 1, component 1).

Simulates the output of a per-frame CNN feature-extraction backbone running
on 1-3 corner-mounted OR cameras, WITHOUT touching real video, real codecs,
or real file I/O. This is a deliberate stand-in for a real feature-extraction
service - the same posture TeCNO (Czempiel et al., MICCAI 2020) takes:
operate on pre-extracted embeddings, not raw pixels, since end-to-end video
training over hour-long surgeries is intractable in practice and explicitly
out of scope for this "mock features" assignment.

Two structural properties are deliberately reproduced (not accuracy - per
the assignment's "not aiming for accuracy" framing - but *structure*):
  1. Segment-duration priors: "patient_present" and "operation" are sampled
     as the two long phases (matching the assignment's framing of both as
     long/high-duration bottleneck classes), not uniform-random label noise.
  2. Class-conditional-but-noisy features: each phase has a fixed prototype
     feature vector, but observed features are that prototype plus per-camera
     and cross-camera noise - a lightweight model can learn real signal, but
     it isn't a trivial lookup, mirroring the real gap between raw CNN
     embeddings and clean phase labels.

Design choice - fixed sequence length: durations are sampled per-phase, then
rescaled to sum to exactly `seq_len`. Every generated case therefore has the
identical [max_cameras, seq_len, feature_dim] shape, so batching is a plain
`torch.stack` with no time-axis padding/masking machinery. That machinery is
real-world-necessary but adds complexity orthogonal to what this prototype
is meant to demonstrate; camera-axis padding (1-3 real cameras vs.
max_cameras) is kept, via `camera_mask`, because that IS the multi-view
occlusion problem the assignment asks the model to handle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import NUM_CLASSES, DataConfig


def generate_class_prototypes(feature_dim: int, seed: int) -> np.ndarray:
    """One fixed feature-space "anchor" per phase, shared across every
    generated case. Represents "what a real CNN embedding for this phase
    tends to look like" - held constant so the synthetic task has a stable
    signal to learn, rather than being redrawn (and hence unlearnable) for
    every case."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=2.0, size=(NUM_CLASSES, feature_dim))


@dataclass
class SyntheticCase:
    features: torch.Tensor  # [max_cameras, seq_len, feature_dim], float32
    camera_mask: torch.Tensor  # [max_cameras], 1.0 = real camera, 0.0 = absent/occluded-out
    labels: torch.Tensor  # [seq_len], int64 class indices


class SyntheticSurgeryGenerator:
    """Generates one synthetic OR case per call to `.generate(case_seed)`."""

    def __init__(self, config: DataConfig):
        self.config = config
        # Prototypes are tied to the *dataset* seed, not the per-case seed,
        # so every case in a dataset shares the same underlying class
        # semantics - this is what makes the task learnable at all rather
        # than pure per-case noise.
        self.prototypes = generate_class_prototypes(config.feature_dim, config.seed)

    def _sample_segment_lengths(self, rng: np.random.Generator) -> list[int]:
        """Sample per-phase durations from a Gamma prior parameterized by
        the config's relative duration weights, then rescale so they sum
        exactly to `seq_len` (see module docstring for why fixed-length)."""
        weights = np.array(self.config.phase_duration_weights, dtype=np.float64)
        shape = self.config.phase_duration_gamma_shape
        # Gamma(shape, scale) has mean = shape * scale; choosing scale so
        # the mean equals each phase's relative weight, then rescaling the
        # whole vector to sum to seq_len, preserves relative proportions
        # (long "patient_present"/"operation", short others) while still
        # injecting per-case variability around them.
        raw = rng.gamma(shape=shape, scale=weights / shape)
        raw = np.clip(raw, a_min=1e-3, a_max=None)
        scaled = raw / raw.sum() * self.config.seq_len
        lengths = np.floor(scaled).astype(int)
        lengths = np.clip(lengths, a_min=1, a_max=None)  # every phase gets >=1 frame
        remainder = self.config.seq_len - int(lengths.sum())
        lengths[int(np.argmax(lengths))] += remainder  # absorb rounding drift into the longest segment
        return lengths.tolist()

    def generate(self, case_seed: int) -> SyntheticCase:
        rng = np.random.default_rng(case_seed)
        cfg = self.config

        num_cameras = int(rng.integers(cfg.min_cameras, cfg.max_cameras + 1))

        lengths = self._sample_segment_lengths(rng)
        labels_np = np.concatenate(
            [np.full(length, class_idx, dtype=np.int64) for class_idx, length in enumerate(lengths)]
        )
        assert labels_np.shape[0] == cfg.seq_len, "segment lengths must sum to seq_len exactly"

        features = np.zeros((cfg.max_cameras, cfg.seq_len, cfg.feature_dim), dtype=np.float32)

        # Cross-camera correlated noise: one noise trajectory shared by
        # every camera in this case (scene-level nuisance - lighting,
        # ambient motion - all cameras observe together), distinct from
        # each camera's own independent per-sensor/viewpoint noise below.
        shared_noise = rng.normal(0.0, cfg.cross_camera_noise_std, size=(cfg.seq_len, cfg.feature_dim))
        class_signal = self.prototypes[labels_np]  # [seq_len, feature_dim]

        for cam in range(num_cameras):
            per_camera_noise = rng.normal(0.0, cfg.feature_noise_std, size=(cfg.seq_len, cfg.feature_dim))
            features[cam] = class_signal + shared_noise + per_camera_noise
        # Cameras beyond num_cameras stay all-zero and are excluded via
        # camera_mask - the fusion layer in model.py must never rely on
        # their (meaningless) zero values, only on the mask.

        camera_mask = np.zeros(cfg.max_cameras, dtype=np.float32)
        camera_mask[:num_cameras] = 1.0

        return SyntheticCase(
            features=torch.from_numpy(features),
            camera_mask=torch.from_numpy(camera_mask),
            labels=torch.from_numpy(labels_np),
        )


class SurgeryPhaseDataset(Dataset):
    """Deterministic-per-index synthetic dataset: `__getitem__(i)` always
    regenerates the exact same case for a given `base_seed`, so re-running
    train.py/evaluate.py with the same config reproduces an identical
    dataset without persisting anything to disk."""

    def __init__(self, config: DataConfig, num_sequences: int, base_seed: int | None = None):
        self.generator = SyntheticSurgeryGenerator(config)
        self.num_sequences = num_sequences
        # Offset so train/val splits (constructed with different base_seed
        # values, see train.py) never sample overlapping cases.
        self.base_seed = base_seed if base_seed is not None else config.seed

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> SyntheticCase:
        case_seed = self.base_seed * 100_003 + idx  # large odd multiplier avoids trivial seed collisions
        return self.generator.generate(case_seed)


def collate_cases(batch: list[SyntheticCase]) -> dict[str, torch.Tensor]:
    """Every case already has identical shape, so batching is a plain
    stack - no time-axis padding/masking required (see module docstring)."""
    return {
        "features": torch.stack([c.features for c in batch]),
        "camera_mask": torch.stack([c.camera_mask for c in batch]),
        "labels": torch.stack([c.labels for c in batch]),
    }
