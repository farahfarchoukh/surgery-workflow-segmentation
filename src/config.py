"""Single source of truth for pipeline configuration.

Design choice: four small dataclasses + one YAML file, no Hydra/OmegaConf.
A config-composition framework would be one more thing to defend line-by-line
in the follow-up interview; a plain dataclass hierarchy is fully sufficient
for a repo this size and every field is greppable back to where it's used.

PHASE_LABELS and ALLOWED_TRANSITIONS live here (not in data.py) because they
are shared vocabulary across every module: data.py samples segments from
this label set, model.py sizes its output layer from it, postprocess.py
enforces ALLOWED_TRANSITIONS, metrics.py reports per-class breakdowns using
these names, and error_analysis.py targets "patient_present"/"operation" by
name.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# Ordered class list. Index position IS the model's class index everywhere
# (data.py, model.py output layer, metrics.py confusion accounting) - keep
# this list as the one place that ordering is defined.
PHASE_LABELS: list[str] = [
    "patient_present",  # long, static, background-noise-driven false positives
    "preparation",
    "operation",  # high-intensity, occlusion-heavy, imprecise boundaries
    "closing",
    "patient_leave",
]

NUM_CLASSES = len(PHASE_LABELS)

# A single OR case flows forward through phases; it does not revisit an
# earlier phase. Encoding this as a strict adjacency (self-loop + one
# forward step) lets postprocess.py reject nonsensical predicted
# transitions like "operation -> patient_present" (PKI-style prior masking,
# see report Sec 1.3 / Czempiel et al. TeCNO 2020 precedent).
_FORWARD_ADJACENCY: dict[str, list[str]] = {
    "patient_present": ["patient_present", "preparation"],
    "preparation": ["preparation", "operation"],
    "operation": ["operation", "closing"],
    "closing": ["closing", "patient_leave"],
    "patient_leave": ["patient_leave"],
}


def build_allowed_transition_matrix() -> np.ndarray:
    """Boolean [NUM_CLASSES, NUM_CLASSES] matrix; [i, j] = True iff a
    transition from class i to class j is clinically plausible."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=bool)
    for src, dsts in _FORWARD_ADJACENCY.items():
        i = PHASE_LABELS.index(src)
        for dst in dsts:
            j = PHASE_LABELS.index(dst)
            matrix[i, j] = True
    return matrix


@dataclass
class DataConfig:
    seed: int = 42
    feature_dim: int = 128
    # 240 frames at seconds_per_frame=10 below = a ~40-minute synthetic case.
    # Sized (together with ModelConfig/TrainConfig below) so the full training
    # run finishes in ~20-30s on a 4-core CPU with no GPU - measured directly
    # on this dev box, not assumed (see report Sec 4.2.2 for the FPS/scale
    # rationale, and note this is a *lightweight prototype* sizing choice,
    # not a claim about the receptive field a production model would need).
    seq_len: int = 240  # fixed-length synthetic case; see data.py docstring for why fixed-length
    min_cameras: int = 1
    max_cameras: int = 3
    # Relative mean-duration weights per phase (index-aligned with PHASE_LABELS),
    # normalized to sum to seq_len per sampled case - see data.py. Ratios matter,
    # not absolute values: "patient_present" and "operation" are the two long
    # phases, matching the assignment's framing of both as long/static.
    phase_duration_weights: list[float] = field(
        default_factory=lambda: [90.0, 70.0, 180.0, 60.0, 30.0]
    )
    phase_duration_gamma_shape: float = 8.0  # higher = less relative variance around the mean
    feature_noise_std: float = 1.0  # per-camera independent noise
    cross_camera_noise_std: float = 0.4  # shared noise correlated across a case's cameras
    seconds_per_frame: float = 10.0  # low-Hz feature extraction, see report Sec 4.2.2
    num_train_sequences: int = 32
    num_val_sequences: int = 8


@dataclass
class ModelConfig:
    fusion_dim: int = 48
    hidden_dim: int = 48
    kernel_size: int = 3
    stage1_layers: int = 6  # dual-dilated layers in the prediction-generation stage (max dilation 32)
    refine_layers: int = 4  # dual-dilated layers in each refinement stage
    num_refine_stages: int = 2  # MS-TCN-style iterative refinement stages
    dropout: float = 0.15
    view_dropout_prob: float = 0.3  # P(a given camera is zeroed) during training, see model.py


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 8
    smoothing_loss_weight: float = 0.15  # MS-TCN truncated-MSE flicker penalty, see model.py
    device: str = "cpu"
    num_threads: int = 4  # this dev box has 4 cores; torch defaults to fewer, see train.py


@dataclass
class EvalConfig:
    iou_thresholds: list[float] = field(default_factory=lambda: [0.10, 0.25, 0.50])
    majority_filter_window: int = 9  # frames, odd, see postprocess.py
    min_duration_frames: dict = field(
        default_factory=lambda: {
            "patient_present": 15,
            "preparation": 10,
            "operation": 15,
            "closing": 10,
            "patient_leave": 8,
        }
    )
    hysteresis_frames: int = 5  # consecutive frames required before "firing" a phase-change event
    false_positive_cost: float = 2.0  # cost of a spurious phase-start alert
    false_negative_cost: float = 1.0  # cost of a late/missed phase-start alert
    use_viterbi: bool = False  # off by default: non-causal, worse fit for the online story


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            eval=EvalConfig(**raw.get("eval", {})),
        )


def set_seed(seed: int) -> None:
    """Centralized seeding so reproducibility is explicit at each entrypoint
    (train.py / evaluate.py / error_analysis.py all call this once), not
    implicitly buried inside a config object's __post_init__."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
