"""Renders training/evaluation outcomes as static PNGs for the report -
not just printed tables. Reads the real artifacts each pipeline stage
already writes (outputs/training_history.json, outputs/eval_report.json,
outputs/error_analysis_report.json); this script never invents a number,
it only visualizes ones already produced and checked-in as JSON.

Run `make train && make evaluate && make error-analysis` first so the
input files exist. Same palette as scripts/make_report_assets.py - the
first five slots of the dataviz skill's validated categorical palette -
so every graphic in the report reads as one consistent visual system.
One fixed semantic mapping across all three charts: blue = baseline/raw/
train, orange = comparison/postprocessed/val/degraded - never reassigned
per-chart, so color means the same thing everywhere a reader sees it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
DIAGRAMS_DIR = REPO_ROOT / "report" / "diagrams"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e3e2df"

BLUE = "#2a78d6"  # slot 1 - baseline / raw / train
ORANGE = "#eb6834"  # slot 2 - comparison / postprocessed / val / degraded
AQUA = "#1baf7a"  # slot 3 - recovery / target-class highlight
VIOLET = "#4a3aa7"  # slot 7 - secondary highlight

PHASE_LABELS = ["patient_present", "preparation", "operation", "closing", "patient_leave"]
TARGET_CLASSES = {"patient_present", "operation"}


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def build_training_curves() -> None:
    """Real per-epoch history from src/train.py, not just the 5 log lines
    that print to console - the full curve, computed and saved every run."""
    history = json.loads((OUTPUTS_DIR / "training_history.json").read_text())["epochs"]
    epochs = [h["epoch"] for h in history]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURFACE)
    fig.suptitle("Training run (config/default.yaml, seed 42)", color=TEXT_PRIMARY, fontsize=11, y=1.02)

    ax_loss.plot(epochs, [h["train_loss"] for h in history], color=BLUE, linewidth=2, zorder=3)
    ax_loss.set_title("Training loss", color=TEXT_PRIMARY, fontsize=9.5, loc="left")
    ax_loss.set_xlabel("epoch", color=TEXT_SECONDARY, fontsize=8.5)
    _style_axes(ax_loss)

    ax_acc.plot(epochs, [h["train_frame_acc"] for h in history], color=BLUE, linewidth=2, zorder=3, label="train")
    ax_acc.plot(epochs, [h["val_frame_acc"] for h in history], color=ORANGE, linewidth=2, zorder=3, label="val")
    ax_acc.set_title("Frame accuracy", color=TEXT_PRIMARY, fontsize=9.5, loc="left")
    ax_acc.set_xlabel("epoch", color=TEXT_SECONDARY, fontsize=8.5)
    ax_acc.set_ylim(0, 1.02)
    ax_acc.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_SECONDARY, loc="lower right")
    _style_axes(ax_acc)

    fig.tight_layout()
    fig.savefig(DIAGRAMS_DIR / "training_curves.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("saved training_curves.png")


def build_metrics_comparison() -> None:
    """Raw vs postprocessed, from the real eval_report.json - postprocessing's
    value-add as a measured bar-height delta, not an assumed one."""
    report = json.loads((OUTPUTS_DIR / "eval_report.json").read_text())
    raw, post = report["raw"], report["postprocessed"]

    metric_labels = ["frame\naccuracy", "edit score\n(/100)", "F1@10", "F1@25", "F1@50"]

    def _vals(d):
        f1 = d["segmental_f1"]
        return [d["frame_acc"], d["edit_score"] / 100, f1["f1@10"], f1["f1@25"], f1["f1@50"]]

    raw_vals, post_vals = _vals(raw), _vals(post)

    x = range(len(metric_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SURFACE)
    ax.bar([i - width / 2 for i in x], raw_vals, width, color=BLUE, label="raw", zorder=3)
    ax.bar([i + width / 2 for i in x], post_vals, width, color=ORANGE, label="postprocessed", zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_labels, color=TEXT_SECONDARY, fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_title(
        f"Raw vs. postprocessed ({raw['num_cases']} held-out validation cases)", color=TEXT_PRIMARY, fontsize=10.5, loc="left"
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_SECONDARY, loc="lower right")
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(DIAGRAMS_DIR / "metrics_comparison.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("saved metrics_comparison.png")


def build_error_analysis_chart() -> None:
    """The report's Sec 1.5 key finding, visualized: per-class frame-accuracy
    degradation under injected noise (left), and segmental F1@50 raw-vs-post
    under noise per class (right) - showing operation recovers via
    postprocessing but patient_present does not, straight from the real
    error_analysis_report.json."""
    report = json.loads((OUTPUTS_DIR / "error_analysis_report.json").read_text())
    clean, noisy = report["clean"], report["noisy"]

    deltas = [noisy[c]["raw_frame_acc_mean"] - clean[c]["raw_frame_acc_mean"] for c in PHASE_LABELS]
    bar_colors = [AQUA if c in TARGET_CLASSES else VIOLET for c in PHASE_LABELS]

    f1_raw_noisy = [noisy[c]["raw_f1_mean"] for c in PHASE_LABELS]
    f1_post_noisy = [noisy[c]["post_f1_mean"] for c in PHASE_LABELS]

    fig, (ax_delta, ax_f1) = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
    fig.suptitle("Error analysis: injected occlusion / background-jitter noise", color=TEXT_PRIMARY, fontsize=11, y=1.03)

    y_pos = range(len(PHASE_LABELS))
    ax_delta.barh(list(y_pos), deltas, color=bar_colors, zorder=3)
    ax_delta.set_yticks(list(y_pos))
    ax_delta.set_yticklabels(PHASE_LABELS, color=TEXT_SECONDARY, fontsize=8.5)
    ax_delta.invert_yaxis()
    ax_delta.set_xlabel("raw frame-accuracy delta (noisy - clean)", color=TEXT_SECONDARY, fontsize=8.5)
    ax_delta.set_title("Degradation by class (aqua = the two named target classes)", color=TEXT_PRIMARY, fontsize=9.5, loc="left")
    ax_delta.axvline(0, color=TEXT_SECONDARY, linewidth=0.8, zorder=2)
    _style_axes(ax_delta)
    ax_delta.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax_delta.grid(axis="y", visible=False)

    x = range(len(PHASE_LABELS))
    width = 0.35
    ax_f1.bar([i - width / 2 for i in x], f1_raw_noisy, width, color=BLUE, label="raw", zorder=3)
    ax_f1.bar([i + width / 2 for i in x], f1_post_noisy, width, color=ORANGE, label="postprocessed", zorder=3)
    ax_f1.set_xticks(list(x))
    ax_f1.set_xticklabels(PHASE_LABELS, color=TEXT_SECONDARY, fontsize=7.5, rotation=20, ha="right")
    ax_f1.set_ylim(0, 1.12)
    ax_f1.set_title("Segmental F1@50 under noise: recovers or doesn't", color=TEXT_PRIMARY, fontsize=9.5, loc="left")
    ax_f1.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_SECONDARY, loc="upper right")
    _style_axes(ax_f1)

    fig.tight_layout()
    fig.savefig(DIAGRAMS_DIR / "error_analysis.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("saved error_analysis.png")


if __name__ == "__main__":
    missing = [
        name
        for name in ("training_history.json", "eval_report.json", "error_analysis_report.json")
        if not (OUTPUTS_DIR / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {OUTPUTS_DIR} - run `make train && make evaluate && make error-analysis` first."
        )
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    build_training_curves()
    build_metrics_comparison()
    build_error_analysis_chart()
