"""Minimal training entrypoint: proves the mock signal in src/data.py is
actually learnable by src/model.py, on CPU, in seconds - no GPU, no real
dataset, no download step. This is the earliest integration checkpoint in
the build (see plan): a shrinking loss here catches data/model shape or
signal-strength bugs before postprocess/metrics/error_analysis add more
code on top.
"""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import NUM_CLASSES, ExperimentConfig, set_seed
from src.data import SurgeryPhaseDataset, collate_cases
from src.logging_config import setup_logging
from src.model import PhaseSegmentationModel, compute_loss, count_params


class TrainingDivergedError(RuntimeError):
    """Raised when a training step produces a non-finite (NaN/Inf) loss.

    Fail LOUDLY and stop immediately rather than silently continuing:
    a diverged run would otherwise still write a checkpoint at the end
    (torch.save doesn't know or care that the weights are garbage), and
    every downstream consumer (evaluate.py, error_analysis.py) would then
    silently produce meaningless metrics from a corrupted model with no
    indication anything went wrong. Common real causes: a learning rate
    too high for a given config change, or a data/config edit that broke
    the signal-to-noise ratio DataConfig's own validation can't catch
    (validation checks values are well-formed, not that they combine into
    a learnable task)."""


def frame_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Quick training-progress signal only - the authoritative, full metric
    stack (frame acc + segmental F1/edit score + product metrics) lives in
    metrics.py and is what evaluate.py/error_analysis.py report."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def save_checkpoint_with_backup(checkpoint: dict, output_path: Path, keep_last: int = 3) -> None:
    """Writes the checkpoint, but first rotates any existing file at
    `output_path` into a timestamped backup instead of silently
    overwriting it - the same rationale as the rotating log handler in
    logging_config.py: bounded history (`keep_last`), never zero history.
    A bad training run overwriting the last good checkpoint with no way
    back is exactly the kind of silent data loss a production training
    pipeline shouldn't allow by default."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backup_dir = output_path.parent / "backups"

    if output_path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        # Microsecond precision, not just seconds: rapid successive saves
        # (e.g. a hyperparameter-search loop calling train() repeatedly)
        # can land within the same second - second-resolution timestamps
        # collided and silently overwrote each other, caught by
        # tests/test_robustness.py::test_checkpoint_backup_rotation_keeps_bounded_history.
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = backup_dir / f"{output_path.stem}_{timestamp}{output_path.suffix}"
        shutil.copy2(output_path, backup_path)

        existing_backups = sorted(backup_dir.glob(f"{output_path.stem}_*{output_path.suffix}"))
        for stale in existing_backups[:-keep_last]:
            stale.unlink()

    torch.save(checkpoint, output_path)


def train(cfg: ExperimentConfig, output_path: Path) -> PhaseSegmentationModel:
    logger = setup_logging("train")
    logger.info("training run starting: output_path=%s", output_path)
    logger.info(
        "config: seed=%d epochs=%d lr=%g batch_size=%d seq_len=%d feature_dim=%d",
        cfg.data.seed, cfg.train.epochs, cfg.train.lr, cfg.train.batch_size, cfg.data.seq_len, cfg.data.feature_dim,
    )

    set_seed(cfg.data.seed)
    torch.set_num_threads(cfg.train.num_threads)

    train_ds = SurgeryPhaseDataset(cfg.data, cfg.data.num_train_sequences, base_seed=cfg.data.seed)
    val_ds = SurgeryPhaseDataset(cfg.data, cfg.data.num_val_sequences, base_seed=cfg.data.seed + 1)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_cases
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, collate_fn=collate_cases)

    model = PhaseSegmentationModel(cfg.model, cfg.data.feature_dim, NUM_CLASSES)
    param_count = count_params(model)
    print(f"model param count: {param_count:,}")
    logger.info("model instantiated: %d parameters", param_count)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    start = time.time()
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            all_logits = model(batch["features"], batch["camera_mask"])
            loss = compute_loss(all_logits, batch["labels"], cfg.train.smoothing_loss_weight)

            if not torch.isfinite(loss):
                logger.error("non-finite loss %s at epoch %d, batch %d - aborting", loss.item(), epoch, batch_idx)
                raise TrainingDivergedError(
                    f"Non-finite loss ({loss.item()}) at epoch {epoch}, batch {batch_idx}. "
                    f"Stopping immediately rather than checkpointing a corrupted model - "
                    f"try a lower TrainConfig.lr, or check whether a recent DataConfig change "
                    f"reduced the class signal-to-noise ratio below what's learnable."
                )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += frame_accuracy(all_logits[-1].detach(), batch["labels"])
            n_batches += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == cfg.train.epochs:
            model.eval()
            val_acc, val_batches = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    all_logits = model(batch["features"], batch["camera_mask"])
                    val_acc += frame_accuracy(all_logits[-1], batch["labels"])
                    val_batches += 1
            msg = (
                f"epoch {epoch:3d}/{cfg.train.epochs}  "
                f"train_loss={epoch_loss / n_batches:.4f}  "
                f"train_frame_acc={epoch_acc / n_batches:.3f}  "
                f"val_frame_acc={val_acc / val_batches:.3f}"
            )
            print(msg)
            logger.info(msg)

    elapsed = time.time() - start
    print(f"training finished in {elapsed:.1f}s on CPU")
    logger.info("training finished in %.1fs", elapsed)

    # Deliberately NOT pickling `cfg` (an ExperimentConfig dataclass) into the
    # checkpoint: torch.load can only skip its weights_only safety check for
    # a small allowlist of built-in types (tensors, dict/list/str/int/float),
    # not arbitrary custom classes. A checkpoint is exactly the kind of file
    # that ends up shared/downloaded, so keeping it loadable in the safe
    # (weights_only=True) mode matters - see evaluate.load_model. Only
    # plain-JSON-safe architecture metadata is stored; hyperparameters live
    # in config/default.yaml (or whichever --config produced this run), which
    # the caller already has.
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": asdict(cfg.model),
        "feature_dim": cfg.data.feature_dim,
        "num_classes": NUM_CLASSES,
    }
    save_checkpoint_with_backup(checkpoint, output_path)
    print(f"checkpoint saved to {output_path}")
    logger.info("checkpoint saved to %s (previous version, if any, backed up)", output_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="outputs/checkpoint.pt")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    train(cfg, Path(args.output))


if __name__ == "__main__":
    main()
