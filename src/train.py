"""Minimal training entrypoint: proves the mock signal in src/data.py is
actually learnable by src/model.py, on CPU, in seconds - no GPU, no real
dataset, no download step. This is the earliest integration checkpoint in
the build (see plan): a shrinking loss here catches data/model shape or
signal-strength bugs before postprocess/metrics/error_analysis add more
code on top.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import NUM_CLASSES, ExperimentConfig, set_seed
from src.data import SurgeryPhaseDataset, collate_cases
from src.model import PhaseSegmentationModel, compute_loss, count_params


def frame_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Quick training-progress signal only - the authoritative, full metric
    stack (frame acc + segmental F1/edit score + product metrics) lives in
    metrics.py and is what evaluate.py/error_analysis.py report."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def train(cfg: ExperimentConfig, output_path: Path) -> PhaseSegmentationModel:
    set_seed(cfg.data.seed)
    torch.set_num_threads(cfg.train.num_threads)

    train_ds = SurgeryPhaseDataset(cfg.data, cfg.data.num_train_sequences, base_seed=cfg.data.seed)
    val_ds = SurgeryPhaseDataset(cfg.data, cfg.data.num_val_sequences, base_seed=cfg.data.seed + 1)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_cases
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, collate_fn=collate_cases)

    model = PhaseSegmentationModel(cfg.model, cfg.data.feature_dim, NUM_CLASSES)
    print(f"model param count: {count_params(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    start = time.time()
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            all_logits = model(batch["features"], batch["camera_mask"])
            loss = compute_loss(all_logits, batch["labels"], cfg.train.smoothing_loss_weight)
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
            print(
                f"epoch {epoch:3d}/{cfg.train.epochs}  "
                f"train_loss={epoch_loss / n_batches:.4f}  "
                f"train_frame_acc={epoch_acc / n_batches:.3f}  "
                f"val_frame_acc={val_acc / val_batches:.3f}"
            )

    elapsed = time.time() - start
    print(f"training finished in {elapsed:.1f}s on CPU")

    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    torch.save(checkpoint, output_path)
    print(f"checkpoint saved to {output_path}")
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
