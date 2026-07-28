"""Thin orchestration: data -> model -> postprocess -> metrics, one
reproducible eval run over the validation split.

Deliberately computes metrics TWICE per case - once on the model's raw
argmax predictions, once on the postprocess.generate_timeline() output -
and reports both. This makes postprocessing's value-add a measured number,
not an assumed one: the report can cite "edit score improved from X to Y"
instead of asserting smoothing helps.

Batched, not per-case, model inference. An earlier version ran one
forward pass per case in a loop; measuring it (not assuming) showed that's
the wrong lever at this model's size - thread-level parallelism across
cases was actually 25-45% SLOWER (thread-pool overhead and reduced
intra-op parallelism dominate for a model this small), while batching
every case into ONE forward pass measured 1.1x-7.9x faster. That's not a
coincidence: it's the same mechanism Sec 4.3.2's Triton dynamic-batching
cost lever describes for production, just exercised locally. Postprocessing
and metrics stay per-case (they're cheap, sequential Python logic - the
model forward pass is what's worth batching).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import (
    NUM_CLASSES,
    PHASE_LABELS,
    ExperimentConfig,
    ModelConfig,
    build_allowed_transition_matrix,
    set_seed,
)
from src.data import SurgeryPhaseDataset, SyntheticCase, collate_cases
from src.metrics import MetricsReport, compute_all_metrics
from src.model import PhaseSegmentationModel
from src.postprocess import frames_to_segments, generate_timeline, segments_to_frames


class CorruptedCheckpointError(RuntimeError):
    """Raised when a checkpoint file exists but can't be loaded as a valid
    model - truncated/corrupted file, wrong format, or an architecture that
    doesn't match its own recorded metadata. Wraps whatever low-level
    exception torch raised (a pickle error, a missing dict key, a
    state_dict shape mismatch) with one consistent, actionable message,
    rather than surfacing a different raw traceback shape for each failure
    mode to whoever is running evaluate.py/error_analysis.py."""


def load_model(cfg: ExperimentConfig, checkpoint_path: Path) -> PhaseSegmentationModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train one first: "
            f"`make train` or `python -m src.train --output {checkpoint_path}`."
        )
    try:
        # weights_only=True (the safe default since PyTorch 2.6): the
        # checkpoint was saved as tensors + plain dict/str/int, no pickled
        # custom classes (see train.train's checkpoint-writing comment), so
        # nothing here needs to unpickle arbitrary code.
        ckpt = torch.load(checkpoint_path, weights_only=True)

        # ckpt["model_config"] takes precedence over cfg.model: the
        # checkpoint is self-describing about the ARCHITECTURE it was
        # actually trained with, so loading it against a caller-supplied
        # cfg.model that has since drifted (e.g. someone edited
        # config/default.yaml after training) fails loudly via a
        # state_dict shape mismatch instead of silently loading garbage.
        model_cfg = ModelConfig(**ckpt["model_config"]) if "model_config" in ckpt else cfg.model
        model = PhaseSegmentationModel(
            model_cfg, ckpt.get("feature_dim", cfg.data.feature_dim), ckpt.get("num_classes", NUM_CLASSES)
        )
        model.load_state_dict(ckpt["model_state"])
    except FileNotFoundError:
        raise
    except Exception as e:
        raise CorruptedCheckpointError(
            f"Could not load checkpoint at {checkpoint_path}: {type(e).__name__}: {e}. "
            f"The file exists but isn't a valid checkpoint for this model - it may be "
            f"truncated, corrupted, or written by an incompatible version. Retrain with "
            f"`make train` to produce a fresh one."
        ) from e
    model.eval()
    return model


def compute_batch_logits(model: PhaseSegmentationModel, cases: list[SyntheticCase], batch_size: int = 32) -> np.ndarray:
    """The batched forward pass: chunks `cases` into `batch_size`-sized
    groups (bounded, so memory use doesn't grow unboundedly with dataset
    size) and runs each chunk through the model in one call. Returns
    [len(cases), seq_len, num_classes]."""
    if not cases:
        return np.empty((0,), dtype=np.float32)
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(cases), batch_size):
            chunk = cases[start : start + batch_size]
            batch = collate_cases(chunk)
            logits = model(batch["features"], batch["camera_mask"])[-1]  # [b, T, C]
            all_logits.append(logits.numpy())
    return np.concatenate(all_logits, axis=0)


def evaluate_case_from_logits(
    logits_np: np.ndarray, case: SyntheticCase, cfg: ExperimentConfig, allowed: np.ndarray
) -> tuple[MetricsReport, MetricsReport]:
    """Same computation as the old evaluate_case, but takes an already-
    computed logits array instead of running the model itself - lets the
    (expensive, batchable) forward pass and the (cheap, inherently
    per-case) postprocessing/metrics stay separate."""
    gt_frames = case.labels.numpy()
    gt_segments = frames_to_segments(gt_frames)

    raw_pred_frames = logits_np.argmax(axis=-1)
    raw_segments = frames_to_segments(raw_pred_frames)
    raw_report = compute_all_metrics(
        raw_pred_frames, gt_frames, raw_segments, gt_segments, cfg.eval, cfg.data.seconds_per_frame
    )

    post_segments = generate_timeline(logits_np, cfg.eval, allowed)
    post_pred_frames = segments_to_frames(post_segments, cfg.data.seq_len)
    post_report = compute_all_metrics(
        post_pred_frames, gt_frames, post_segments, gt_segments, cfg.eval, cfg.data.seconds_per_frame
    )
    return raw_report, post_report


def evaluate_case(model, case: SyntheticCase, cfg: ExperimentConfig, allowed: np.ndarray) -> tuple[MetricsReport, MetricsReport]:
    """Single-case convenience wrapper (used by callers that only have one
    case at hand, e.g. an interactive check) - internally just a batch of
    one. Prefer compute_batch_logits + evaluate_case_from_logits directly
    when processing many cases, which is what run_evaluation does."""
    logits_np = compute_batch_logits(model, [case])[0]
    return evaluate_case_from_logits(logits_np, case, cfg, allowed)


def aggregate(reports: list[MetricsReport]) -> dict:
    """Simple mean-of-cases aggregation for scalar/per-class metrics; the
    cost-weighted event counts are SUMMED (a total cost over the validation
    set is the more natural quantity for that metric than an averaged one).
    Fine at this prototype's dataset scale - a production eval harness would
    pool raw boundary-latency samples across cases before taking percentiles
    rather than averaging per-case medians, noted here rather than silently
    assumed away."""
    n = len(reports)
    f1_keys = reports[0].segmental_f1.keys()
    latencies = [r.boundary_latency_median_frames for r in reports if r.boundary_latency_median_frames is not None]
    return {
        "num_cases": n,
        "frame_acc": float(np.mean([r.frame_acc for r in reports])),
        "edit_score": float(np.mean([r.edit_score for r in reports])),
        "segmental_f1": {k: float(np.mean([r.segmental_f1[k]["f1"] for r in reports])) for k in f1_keys},
        "phase_duration_error_seconds": {
            name: float(np.mean([r.phase_duration_error_seconds[name] for r in reports])) for name in PHASE_LABELS
        },
        "boundary_latency_median_frames_avg": float(np.mean(latencies)) if latencies else None,
        "boundary_latency_missed_total": sum(r.boundary_latency_missed for r in reports),
        "cost_weighted_total": {
            "false_positive_events": sum(r.cost_weighted["false_positive_events"] for r in reports),
            "false_negative_events": sum(r.cost_weighted["false_negative_events"] for r in reports),
            "cost": float(sum(r.cost_weighted["cost"] for r in reports)),
        },
    }


def print_comparison(agg_raw: dict, agg_post: dict) -> None:
    print(f"\n{'metric':<28s} {'raw':>12s} {'postprocessed':>16s}")
    print("-" * 58)
    print(f"{'frame accuracy':<28s} {agg_raw['frame_acc']:>12.3f} {agg_post['frame_acc']:>16.3f}")
    print(f"{'edit score':<28s} {agg_raw['edit_score']:>12.1f} {agg_post['edit_score']:>16.1f}")
    for k in agg_raw["segmental_f1"]:
        print(f"{'segmental ' + k:<28s} {agg_raw['segmental_f1'][k]:>12.3f} {agg_post['segmental_f1'][k]:>16.3f}")
    lat_raw = agg_raw["boundary_latency_median_frames_avg"]
    lat_post = agg_post["boundary_latency_median_frames_avg"]
    print(
        f"{'boundary latency (frames)':<28s} "
        f"{lat_raw if lat_raw is not None else float('nan'):>12.1f} "
        f"{lat_post if lat_post is not None else float('nan'):>16.1f}"
    )
    print(
        f"{'transitions missed (total)':<28s} "
        f"{agg_raw['boundary_latency_missed_total']:>12d} {agg_post['boundary_latency_missed_total']:>16d}"
    )
    print(
        f"{'cost-weighted total':<28s} "
        f"{agg_raw['cost_weighted_total']['cost']:>12.1f} {agg_post['cost_weighted_total']['cost']:>16.1f}"
    )
    print("\nphase-duration error (seconds), postprocessed:")
    for name, val in agg_post["phase_duration_error_seconds"].items():
        print(f"  {name:<18s} {val:>8.1f}s")


def run_evaluation(cfg: ExperimentConfig, checkpoint_path: Path) -> dict:
    set_seed(cfg.data.seed)
    torch.set_num_threads(cfg.train.num_threads)

    model = load_model(cfg, checkpoint_path)
    val_ds = SurgeryPhaseDataset(cfg.data, cfg.data.num_val_sequences, base_seed=cfg.data.seed + 1)
    allowed = build_allowed_transition_matrix()

    cases = [val_ds[i] for i in range(len(val_ds))]
    batch_logits = compute_batch_logits(model, cases)

    raw_reports, post_reports = [], []
    for logits_np, case in zip(batch_logits, cases):
        raw_report, post_report = evaluate_case_from_logits(logits_np, case, cfg, allowed)
        raw_reports.append(raw_report)
        post_reports.append(post_report)

    agg_raw = aggregate(raw_reports)
    agg_post = aggregate(post_reports)
    print_comparison(agg_raw, agg_post)

    for name, report in [("raw", agg_raw), ("postprocessed", agg_post)]:
        for f1_key, f1_val in report["segmental_f1"].items():
            assert 0.0 <= f1_val <= 1.0, f"{name} {f1_key} out of range: {f1_val}"

    return {"raw": agg_raw, "postprocessed": agg_post}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoint.pt")
    parser.add_argument("--output", default="outputs/eval_report.json")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    results = run_evaluation(cfg, Path(args.checkpoint))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nfull report saved to {output_path}")


if __name__ == "__main__":
    main()
