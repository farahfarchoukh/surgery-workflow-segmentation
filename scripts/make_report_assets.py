"""Renders the AWS architecture diagram (report Sec 4.3.1) as a static PNG.

Neither mermaid-cli (needs node) nor the `diagrams` library (needs the
`dot` graphviz binary) are usable in this dev environment (see README), so
this hand-draws the same structure - swimlanes + boxes + arrows - via
matplotlib, which is pip-installable with no system binary dependency.

Colors are the first five slots of the dataviz skill's validated
categorical palette (references/palette.md) - a fixed, non-cycled hue per
swimlane, each box also direct-labeled so identity never depends on color
alone (relevant since this is a static, non-interactive print asset).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

EDGE_COLOR = "#2a78d6"  # blue - slot 1
STEADY_COLOR = "#1baf7a"  # aqua - slot 3
ONDEMAND_COLOR = "#eb6834"  # orange - slot 2
COMPUTE_COLOR = "#4a3aa7"  # violet - slot 7
STORAGE_COLOR = "#eda100"  # yellow - slot 4

BAND_LABELS = ["EDGE", "INGESTION", "COMPUTE", "STORAGE / OUTPUT"]
BAND_Y = [3, 2, 1, 0]
BAND_HEIGHT = 0.9


def draw_box(ax, xy, width, height, label, color, text_color="white"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=TEXT_PRIMARY,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        color=text_color,
        weight="bold",
        zorder=4,
        wrap=True,
    )
    return (x, y, width, height)


def box_edge_point(box, side):
    x, y, w, h = box
    return {
        "right": (x + w, y + h / 2),
        "left": (x, y + h / 2),
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
    }[side]


def draw_arrow(ax, start, end, label=None, style="-", color=TEXT_SECONDARY, rad=0.0, label_pos=0.5, label_offset=(0, 0.0)):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        linestyle=style,
        color=color,
        zorder=2,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if label:
        # position along the (possibly curved) path, offset further to keep
        # clear of both the arrowhead and any box the curve passes near
        x = start[0] + (end[0] - start[0]) * label_pos + label_offset[0]
        y = start[1] + (end[1] - start[1]) * label_pos + label_offset[1]
        ax.text(
            x, y, label, ha="center", va="center", fontsize=7.2, color=TEXT_SECONDARY,
            style="italic", zorder=5, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.9),
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.0), dpi=200)
    ax.set_xlim(-0.3, 12.5)
    ax.set_ylim(-0.5, 4.3)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Swimlane bands
    for label, y in zip(BAND_LABELS, BAND_Y):
        ax.add_patch(
            plt.Rectangle((-0.3, y - 0.05), 12.8, BAND_HEIGHT, facecolor="#f0f0ee", edgecolor="none", zorder=1)
        )
        ax.text(-0.2, y + BAND_HEIGHT / 2 - 0.05, label, ha="left", va="center", fontsize=9, color=TEXT_PRIMARY, weight="bold", zorder=2)

    # Edge band
    cam = draw_box(ax, (1.6, 3.1), 2.6, 0.75, "1-3 OR Cameras\n(PTP/IEEE-1588 timestamping)", EDGE_COLOR)
    gw = draw_box(ax, (5.0, 3.1), 2.8, 0.75, "AWS IoT Greengrass\n(edge feature extraction)", EDGE_COLOR)
    draw_arrow(ax, box_edge_point(cam, "right"), box_edge_point(gw, "left"))

    # Ingestion band - two separate arrows diverging from the same gateway
    # box, offset left/right so their labels never collide.
    kds = draw_box(ax, (2.5, 2.1), 3.0, 0.75, "Kinesis Data Streams\n(per-camera embeddings)", STEADY_COLOR)
    kvs = draw_box(ax, (7.0, 2.1), 3.3, 0.75, "Kinesis Video Streams\n(WebRTC Ingestion)", ONDEMAND_COLOR)
    draw_arrow(
        ax, box_edge_point(gw, "bottom"), box_edge_point(kds, "top"), rad=0.15,
        label="embeddings\nalways-on", label_pos=0.55, label_offset=(-0.55, 0.0),
    )
    draw_arrow(
        ax, box_edge_point(gw, "bottom"), box_edge_point(kvs, "top"), style="--", rad=-0.15,
        label="raw video\non trigger only", label_pos=0.55, label_offset=(0.6, 0.0),
    )

    # Compute band
    gate = draw_box(ax, (0.6, 1.1), 3.0, 0.75, "Lambda / Fargate\nmotion-gate (always-on)", COMPUTE_COLOR)
    pool = draw_box(ax, (4.6, 1.1), 4.4, 0.75, "SageMaker Real-Time GPU pool\n(Multi-Model Endpoint, Triton batching,\nApplication Auto Scaling)", COMPUTE_COLOR)
    draw_arrow(ax, box_edge_point(kds, "bottom"), box_edge_point(gate, "top"))
    draw_arrow(
        ax, box_edge_point(gate, "right"), box_edge_point(pool, "left"),
        label="active signal", label_pos=0.5, label_offset=(0.0, 0.22),
    )

    # Storage / output band
    out = draw_box(ax, (4.6, 0.1), 2.8, 0.75, "Phase predictions\n+ hard-example embeddings", STORAGE_COLOR, text_color=TEXT_PRIMARY)
    s3 = draw_box(ax, (8.2, 0.1), 3.6, 0.75, "S3 (Parquet) / OpenSearch\nLifecycle + ISM TTL = N days", STORAGE_COLOR, text_color=TEXT_PRIMARY)
    draw_arrow(ax, box_edge_point(pool, "bottom"), box_edge_point(out, "top"))
    draw_arrow(ax, box_edge_point(out, "right"), box_edge_point(s3, "left"))
    # KVS -> S3 would cross straight through the compute band boxes below it;
    # bow it out to the right instead so it visibly skips past that band.
    draw_arrow(
        ax, box_edge_point(kvs, "right"), box_edge_point(s3, "top"), style="--", rad=-0.35,
        label="raw video, short TTL", label_pos=0.5, label_offset=(0.75, 0.0),
    )

    ax.set_title(
        "AWS Architecture: Multi-Camera Ingestion, Live Inference, Retention-Compliant Storage",
        fontsize=12,
        color=TEXT_PRIMARY,
        weight="bold",
        pad=16,
    )

    output_path = Path(__file__).parent.parent / "report" / "diagrams" / "aws_architecture.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    print(f"diagram saved to {output_path}")


if __name__ == "__main__":
    main()
