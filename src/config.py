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
    # Multi-camera synchronization (src/sync.py): each camera's raw stream is
    # independently jittered and lossy before the alignment layer recovers a
    # per-timestep availability mask - see sync.py's module docstring.
    camera_jitter_std_seconds: float = 3.0  # ~0.3x seconds_per_frame: clock drift / encode+network latency
    camera_frame_drop_prob: float = 0.05  # per-frame packet loss / decode failure rate, independent per camera
    sync_tolerance_seconds: float = 5.0  # ~0.5x seconds_per_frame: matches the report's "nearest frame within tolerance" design

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.min_cameras < 1:
            raise ValueError(f"min_cameras must be >= 1, got {self.min_cameras}")
        if self.max_cameras < self.min_cameras:
            raise ValueError(f"max_cameras ({self.max_cameras}) must be >= min_cameras ({self.min_cameras})")
        if len(self.phase_duration_weights) != NUM_CLASSES:
            raise ValueError(
                f"phase_duration_weights must have {NUM_CLASSES} entries (one per PHASE_LABELS), "
                f"got {len(self.phase_duration_weights)}"
            )
        if any(w <= 0 for w in self.phase_duration_weights):
            raise ValueError(f"phase_duration_weights must all be positive, got {self.phase_duration_weights}")
        if self.phase_duration_gamma_shape <= 0:
            raise ValueError(f"phase_duration_gamma_shape must be positive, got {self.phase_duration_gamma_shape}")
        if self.feature_noise_std < 0 or self.cross_camera_noise_std < 0:
            raise ValueError("noise std values must be non-negative")
        if self.seconds_per_frame <= 0:
            raise ValueError(f"seconds_per_frame must be positive, got {self.seconds_per_frame}")
        if self.num_train_sequences <= 0 or self.num_val_sequences <= 0:
            raise ValueError("num_train_sequences and num_val_sequences must be positive")
        if self.camera_jitter_std_seconds < 0:
            raise ValueError(f"camera_jitter_std_seconds must be non-negative, got {self.camera_jitter_std_seconds}")
        if not (0.0 <= self.camera_frame_drop_prob < 1.0):
            raise ValueError(f"camera_frame_drop_prob must be in [0, 1), got {self.camera_frame_drop_prob}")
        if self.sync_tolerance_seconds <= 0:
            raise ValueError(f"sync_tolerance_seconds must be positive, got {self.sync_tolerance_seconds}")


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

    def __post_init__(self) -> None:
        if self.fusion_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("fusion_dim and hidden_dim must be positive")
        if self.kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {self.kernel_size}")
        if self.stage1_layers < 1:
            raise ValueError(f"stage1_layers must be >= 1, got {self.stage1_layers}")
        if self.num_refine_stages < 0:
            raise ValueError(f"num_refine_stages must be >= 0, got {self.num_refine_stages}")
        if self.num_refine_stages > 0 and self.refine_layers < 1:
            raise ValueError("refine_layers must be >= 1 when num_refine_stages > 0")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not (0.0 <= self.view_dropout_prob <= 1.0):
            raise ValueError(f"view_dropout_prob must be in [0, 1], got {self.view_dropout_prob}")


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 8
    smoothing_loss_weight: float = 0.15  # MS-TCN truncated-MSE flicker penalty, see model.py
    device: str = "cpu"
    num_threads: int = 4  # this dev box has 4 cores; torch defaults to fewer, see train.py

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.smoothing_loss_weight < 0:
            raise ValueError(f"smoothing_loss_weight must be non-negative, got {self.smoothing_loss_weight}")
        if self.num_threads <= 0:
            raise ValueError(f"num_threads must be positive, got {self.num_threads}")


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

    def __post_init__(self) -> None:
        if not self.iou_thresholds or any(not (0.0 < t <= 1.0) for t in self.iou_thresholds):
            raise ValueError(f"iou_thresholds must be non-empty with all values in (0, 1], got {self.iou_thresholds}")
        if self.majority_filter_window < 1:
            raise ValueError(f"majority_filter_window must be >= 1, got {self.majority_filter_window}")
        missing = set(PHASE_LABELS) - set(self.min_duration_frames)
        if missing:
            raise ValueError(f"min_duration_frames is missing entries for classes: {sorted(missing)}")
        if any(v <= 0 for v in self.min_duration_frames.values()):
            raise ValueError(f"min_duration_frames values must all be positive, got {self.min_duration_frames}")
        if self.hysteresis_frames < 1:
            raise ValueError(f"hysteresis_frames must be >= 1, got {self.hysteresis_frames}")
        if self.false_positive_cost < 0 or self.false_negative_cost < 0:
            raise ValueError("false_positive_cost and false_negative_cost must be non-negative")


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
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
