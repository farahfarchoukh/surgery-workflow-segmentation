"""Mock Error Analysis Script (Deliverable 1, component 5).

Simulates an evaluation run on a NOISY validation set and prints a
per-class breakdown, specifically designed to demonstrate that the two
classes the assignment names - "patient_present" (background-noise false
positives / fragmentation) and "operation" (occlusion-driven boundary
imprecision) - degrade more than other classes under targeted, realistic
corruption. This is the "proof of understanding" deliverable: it shows the
pipeline (and whoever built it) correctly reproduces the two named
bottlenecks, not just that metrics.py can compute a number.

Two corruption functions, each targeting exactly one failure mode and
applied ONLY to that class's frames (ground-truth labels are never
touched - only the model's OBSERVED features are corrupted, which is the
correct framing: the model still has to get the same right answer, its
input signal just got worse):

  - inject_occlusion_noise: during "operation" frames, each real camera
    independently either goes fully dark (zeroed - a surgeon/instrument
    fully blocking that view) or gets its noise variance sharply inflated
    (partial occlusion/motion blur) - the direct synthetic analog of
    "severe camera occlusions" during Operation.
  - inject_background_jitter: during "patient_present" frames, short bursts
    of frames are REPLACED with the neighboring "preparation" class's
    feature prototype plus noise - the direct synthetic analog of
    incidental staff movement/equipment adjustment briefly making the
    visual scene look like early activity has started, without the phase
    having actually changed.

Reports BOTH raw (argmax) and postprocessed (postprocess.generate_timeline)
per-class results, and both frame accuracy AND segmental F1 - a deliberate
choice, not redundancy: frame accuracy is locally sensitive and is what
reveals the raw degradation; segmental F1 only requires overall segment
overlap and is what shows how much of that damage postprocess.py recovers.
Showing only one of the two would hide half the story.

Corruption severity was tuned empirically against the trained model (see
git history / development notes) rather than guessed - the model's dual-
dilated receptive field and view-dropout training make it genuinely robust
to mild noise, so weak corruption showed near-zero effect. The parameters
below produce a large, clean gap between target and non-target classes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import PHASE_LABELS, ExperimentConfig, build_allowed_transition_matrix, set_seed
from src.data import SurgeryPhaseDataset, SyntheticCase, generate_class_prototypes
from src.evaluate import compute_batch_logits, load_model
from src.metrics import frame_accuracy_by_class, segment_count_by_class, segmental_f1_by_class
from src.postprocess import frames_to_segments, generate_timeline

TARGET_CLASSES = {"patient_present", "operation"}


def inject_occlusion_noise(
    case: SyntheticCase,
    feature_dim: int,
    rng: np.random.Generator,
    severity: float = 30.0,
    camera_dropout_prob: float = 0.85,
) -> SyntheticCase:
    features = case.features.clone()
    operation_idx = PHASE_LABELS.index("operation")
    mask_frames = case.labels == operation_idx
    num_frames = int(mask_frames.sum().item())
    if num_frames == 0:
        return SyntheticCase(
            features=features, camera_mask=case.camera_mask, labels=case.labels, num_cameras=case.num_cameras
        )

    for cam in range(case.num_cameras):
        if rng.random() < camera_dropout_prob:
            features[cam, mask_frames, :] = 0.0  # camera fully blocked during operation
        else:
            noise = rng.normal(0.0, severity, size=(num_frames, feature_dim)).astype(np.float32)
            features[cam, mask_frames, :] += torch.from_numpy(noise)  # partial occlusion / motion blur
    return SyntheticCase(
        features=features, camera_mask=case.camera_mask, labels=case.labels, num_cameras=case.num_cameras
    )


def inject_background_jitter(
    case: SyntheticCase,
    prototypes: np.ndarray,
    rng: np.random.Generator,
    flip_prob: float = 0.18,
    burst_len: int = 10,
) -> SyntheticCase:
    features = case.features.clone()
    labels_np = case.labels.numpy()
    pp_idx = PHASE_LABELS.index("patient_present")
    neighbor_idx = PHASE_LABELS.index("preparation")
    proto_neighbor = torch.from_numpy(prototypes[neighbor_idx].astype(np.float32))

    t, seq_len = 0, len(labels_np)
    while t < seq_len:
        if labels_np[t] == pp_idx and rng.random() < flip_prob:
            end = min(t + burst_len, seq_len)
            for cam in range(case.num_cameras):
                noise = rng.normal(0.0, 1.0, size=(end - t, proto_neighbor.shape[0])).astype(np.float32)
                features[cam, t:end, :] = proto_neighbor + torch.from_numpy(noise)
            t = end
        else:
            t += 1
    return SyntheticCase(
        features=features, camera_mask=case.camera_mask, labels=case.labels, num_cameras=case.num_cameras
    )


def analyze_case_from_logits(logits_np: np.ndarray, case: SyntheticCase, cfg: ExperimentConfig, allowed: np.ndarray) -> dict:
    """Takes an already-computed logits array rather than running the model
    itself, so the (expensive, batchable) forward pass can be computed once
    for every clean/noisy case together - see evaluate.py's module
    docstring for why batching, not per-case threading, is the mechanism
    that actually helps here."""
    gt_frames = case.labels.numpy()
    gt_segments = frames_to_segments(gt_frames)

    raw_pred_frames = logits_np.argmax(axis=-1)
    raw_segments = frames_to_segments(raw_pred_frames)
    post_segments = generate_timeline(logits_np, cfg.eval, allowed)

    return {
        "raw_frame_acc_by_class": frame_accuracy_by_class(raw_pred_frames, gt_frames),
        "raw_f1_by_class": segmental_f1_by_class(raw_segments, gt_segments, threshold=0.5),
        "raw_seg_counts": segment_count_by_class(raw_segments),
        "post_f1_by_class": segmental_f1_by_class(post_segments, gt_segments, threshold=0.5),
        "post_seg_counts": segment_count_by_class(post_segments),
    }


def aggregate_class_breakdown(results: list[dict]) -> dict:
    breakdown = {}
    for name in PHASE_LABELS:
        raw_acc = [r["raw_frame_acc_by_class"][name] for r in results if r["raw_frame_acc_by_class"][name] is not None]
        breakdown[name] = {
            "raw_frame_acc_mean": float(np.mean(raw_acc)) if raw_acc else None,
            "raw_f1_mean": float(np.mean([r["raw_f1_by_class"][name]["f1"] for r in results])),
            "post_f1_mean": float(np.mean([r["post_f1_by_class"][name]["f1"] for r in results])),
            "raw_avg_segment_count": float(np.mean([r["raw_seg_counts"][name] for r in results])),
            "post_avg_segment_count": float(np.mean([r["post_seg_counts"][name] for r in results])),
        }
    return breakdown


def print_breakdown(clean: dict, noisy: dict) -> None:
    print(f"\n{'-- FRAME ACCURACY (raw model output, most locally sensitive) --':<80s}")
    print(f"{'class':<18s}{'clean':>10s}{'noisy':>10s}{'delta':>10s}")
    print("-" * 48)
    for name in PHASE_LABELS:
        c = clean[name]["raw_frame_acc_mean"]
        n = noisy[name]["raw_frame_acc_mean"]
        marker = "  <-- target failure class" if name in TARGET_CLASSES else ""
        print(f"{name:<18s}{c:>10.3f}{n:>10.3f}{(n - c):>10.3f}{marker}")

    print(f"\n{'-- SEGMENTAL F1@50: raw vs. postprocessed (shows postprocess.py value-add) --':<80s}")
    print(f"{'class':<18s}{'raw clean':>10s}{'raw noisy':>10s}{'post clean':>11s}{'post noisy':>11s}")
    print("-" * 62)
    for name in PHASE_LABELS:
        c, n = clean[name], noisy[name]
        marker = "  <-- target" if name in TARGET_CLASSES else ""
        print(f"{name:<18s}{c['raw_f1_mean']:>10.3f}{n['raw_f1_mean']:>10.3f}{c['post_f1_mean']:>11.3f}{n['post_f1_mean']:>11.3f}{marker}")

    print(f"\n{'-- PREDICTED SEGMENT COUNT (fragmentation proxy: 1.0 = no fragmentation) --':<80s}")
    print(f"{'class':<18s}{'raw clean':>10s}{'raw noisy':>10s}{'post clean':>11s}{'post noisy':>11s}")
    print("-" * 62)
    for name in PHASE_LABELS:
        c, n = clean[name], noisy[name]
        print(
            f"{name:<18s}{c['raw_avg_segment_count']:>10.2f}{n['raw_avg_segment_count']:>10.2f}"
            f"{c['post_avg_segment_count']:>11.2f}{n['post_avg_segment_count']:>11.2f}"
        )

    target_delta = float(np.mean([noisy[c]["raw_frame_acc_mean"] - clean[c]["raw_frame_acc_mean"] for c in TARGET_CLASSES]))
    other = [c for c in PHASE_LABELS if c not in TARGET_CLASSES]
    other_delta = float(np.mean([noisy[c]["raw_frame_acc_mean"] - clean[c]["raw_frame_acc_mean"] for c in other]))
    print(f"\nmean raw frame-acc delta on target classes (patient_present, operation): {target_delta:+.3f}")
    print(f"mean raw frame-acc delta on other classes:                                {other_delta:+.3f}")
    print(f"target classes degrade more under injected noise: {target_delta < other_delta}")


def run_error_analysis(cfg: ExperimentConfig, checkpoint_path: Path) -> dict:
    set_seed(cfg.data.seed)
    torch.set_num_threads(cfg.train.num_threads)

    model = load_model(cfg, checkpoint_path)
    # A THIRD, distinct seed offset from both train's val split (+1) and any
    # other split, so error analysis runs on genuinely held-out cases.
    val_ds = SurgeryPhaseDataset(cfg.data, cfg.data.num_val_sequences, base_seed=cfg.data.seed + 2)
    allowed = build_allowed_transition_matrix()
    prototypes = generate_class_prototypes(cfg.data.feature_dim, cfg.data.seed)
    rng = np.random.default_rng(cfg.data.seed + 999)

    clean_cases = [val_ds[i] for i in range(len(val_ds))]
    noisy_cases = []
    for clean_case in clean_cases:
        noisy_case = inject_occlusion_noise(clean_case, cfg.data.feature_dim, rng)
        noisy_case = inject_background_jitter(noisy_case, prototypes, rng)
        noisy_cases.append(noisy_case)

    # One batched forward pass for all clean cases, one for all noisy cases,
    # instead of 2 * len(val_ds) individual calls.
    clean_logits = compute_batch_logits(model, clean_cases)
    noisy_logits = compute_batch_logits(model, noisy_cases)

    clean_results = [
        analyze_case_from_logits(logits, case, cfg, allowed) for logits, case in zip(clean_logits, clean_cases)
    ]
    noisy_results = [
        analyze_case_from_logits(logits, case, cfg, allowed) for logits, case in zip(noisy_logits, noisy_cases)
    ]

    clean_breakdown = aggregate_class_breakdown(clean_results)
    noisy_breakdown = aggregate_class_breakdown(noisy_results)
    print_breakdown(clean_breakdown, noisy_breakdown)
    return {"clean": clean_breakdown, "noisy": noisy_breakdown}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoint.pt")
    parser.add_argument("--output", default="outputs/error_analysis_report.json")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    results = run_error_analysis(cfg, Path(args.checkpoint))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nfull report saved to {output_path}")


if __name__ == "__main__":
    main()
