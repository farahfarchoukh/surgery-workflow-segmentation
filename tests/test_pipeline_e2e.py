"""End-to-end smoke test: generate -> train -> evaluate -> error_analysis,
using a deliberately tiny config so the whole thing runs in a couple of
seconds. This is a regression guard on the "visibly demonstrates the two
failure modes" requirement itself, not just "the code runs without
crashing" - see the final assertion.
"""

from pathlib import Path

from src.config import DataConfig, EvalConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.error_analysis import run_error_analysis
from src.evaluate import run_evaluation
from src.train import train


def tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            seed=1,
            feature_dim=16,
            seq_len=80,
            min_cameras=1,
            max_cameras=3,
            num_train_sequences=8,
            num_val_sequences=4,
        ),
        model=ModelConfig(fusion_dim=16, hidden_dim=16, stage1_layers=3, refine_layers=2, num_refine_stages=1),
        train=TrainConfig(epochs=8, batch_size=4, num_threads=2),
        eval=EvalConfig(),
    )


def test_full_pipeline_runs_and_metrics_are_in_valid_ranges(tmp_path):
    cfg = tiny_config()
    checkpoint_path = tmp_path / "checkpoint.pt"

    train(cfg, checkpoint_path)
    assert checkpoint_path.exists()

    eval_results = run_evaluation(cfg, checkpoint_path)
    for split in ("raw", "postprocessed"):
        assert 0.0 <= eval_results[split]["frame_acc"] <= 1.0
        assert 0.0 <= eval_results[split]["edit_score"] <= 100.0
        for f1 in eval_results[split]["segmental_f1"].values():
            assert 0.0 <= f1 <= 1.0

    error_results = run_error_analysis(cfg, checkpoint_path)
    for split in ("clean", "noisy"):
        for class_stats in error_results[split].values():
            if class_stats["raw_frame_acc_mean"] is not None:
                assert 0.0 <= class_stats["raw_frame_acc_mean"] <= 1.0
            assert 0.0 <= class_stats["raw_f1_mean"] <= 1.0
            assert 0.0 <= class_stats["post_f1_mean"] <= 1.0


def test_error_analysis_shows_target_classes_degrade_under_noise(tmp_path):
    """The visible-proof assertion: injected occlusion/jitter noise must
    degrade patient_present/operation's raw frame accuracy at least as much
    as it degrades the average of the other three classes. This is what
    stops the error-analysis script from silently regressing into a no-op
    if corruption parameters or the model architecture change later."""
    cfg = tiny_config()
    checkpoint_path = tmp_path / "checkpoint.pt"
    train(cfg, checkpoint_path)

    results = run_error_analysis(cfg, checkpoint_path)
    clean, noisy = results["clean"], results["noisy"]

    target_classes = {"patient_present", "operation"}
    other_classes = [c for c in clean if c not in target_classes]

    target_delta = sum(
        noisy[c]["raw_frame_acc_mean"] - clean[c]["raw_frame_acc_mean"] for c in target_classes
    ) / len(target_classes)
    other_delta = sum(
        noisy[c]["raw_frame_acc_mean"] - clean[c]["raw_frame_acc_mean"] for c in other_classes
    ) / len(other_classes)

    assert target_delta <= other_delta
