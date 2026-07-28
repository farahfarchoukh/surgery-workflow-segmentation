"""Thin orchestration: data -> model -> postprocess -> metrics, one
reproducible eval run over the validation split.

Deliberately computes metrics TWICE per case - once on the model's raw
argmax predictions, once on the postprocess.generate_timeline() output -
and reports both. This makes postprocessing's value-add a measured number,
not an assumed one: the report can cite "edit score improved from X to Y"
instead of asserting smoothing helps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import NUM_CLASSES, PHASE_LABELS, ExperimentConfig, build_allowed_transition_matrix, set_seed
from src.data import SurgeryPhaseDataset
from src.metrics import MetricsReport, compute_all_metrics
from src.model import PhaseSegmentationModel
from src.postprocess import frames_to_segments, generate_timeline, segments_to_frames


def load_model(cfg: ExperimentConfig, checkpoint_path: Path) -> PhaseSegmentationModel:
    model = PhaseSegmentationModel(cfg.model, cfg.data.feature_dim, NUM_CLASSES)
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def evaluate_case(model, case, cfg: ExperimentConfig, allowed: np.ndarray) -> tuple[MetricsReport, MetricsReport]:
    """Returns (raw_report, postprocessed_report) for one synthetic case."""
    with torch.no_grad():
        logits = model(case.features.unsqueeze(0), case.camera_mask.unsqueeze(0))[-1][0]  # [T, C]
    logits_np = logits.numpy()
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
    print(f"{'transitions missed (total)':<28s} {agg_raw['boundary_latency_missed_total']:>12d} {agg_post['boundary_latency_missed_total']:>16d}")
    print(f"{'cost-weighted total':<28s} {agg_raw['cost_weighted_total']['cost']:>12.1f} {agg_post['cost_weighted_total']['cost']:>16.1f}")
    print("\nphase-duration error (seconds), postprocessed:")
    for name, val in agg_post["phase_duration_error_seconds"].items():
        print(f"  {name:<18s} {val:>8.1f}s")


def run_evaluation(cfg: ExperimentConfig, checkpoint_path: Path) -> dict:
    set_seed(cfg.data.seed)
    torch.set_num_threads(cfg.train.num_threads)

    model = load_model(cfg, checkpoint_path)
    val_ds = SurgeryPhaseDataset(cfg.data, cfg.data.num_val_sequences, base_seed=cfg.data.seed + 1)
    allowed = build_allowed_transition_matrix()

    raw_reports, post_reports = [], []
    for i in range(len(val_ds)):
        raw_report, post_report = evaluate_case(model, val_ds[i], cfg, allowed)
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
