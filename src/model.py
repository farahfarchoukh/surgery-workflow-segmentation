"""Temporal Classification Layer (Deliverable 1, component 2) + multi-view
fusion for occlusion robustness (report Sec 4.2.1).

Architecture: a causal, dual-dilated, multi-stage TCN - a direct, scaled-down
implementation of the MS-TCN / MS-TCN++ mechanism (Farha & Gall, CVPR 2019;
Li et al., TPAMI 2020), made causal like TeCNO (Czempiel et al., MICCAI 2020)
for online/real-time deployability. This single mechanism is chosen because
it targets BOTH of the assignment's named failure modes at once:

  - Each residual layer runs two dilated conv branches in parallel: one
    branch's dilation GROWS with network depth (long receptive field, up to
    ~255 frames at the deepest layer) for the long, static "patient_present"
    phase; the other branch's dilation SHRINKS with depth, keeping a local,
    fine-grained receptive field available even in deep layers, for the
    rapid, precise boundaries "operation" needs.
  - Multiple refinement stages each take the PREVIOUS stage's softmax output
    as input and re-refine it - this iterative refinement is the mechanism
    MS-TCN uses to reduce over-segmentation/flicker (exactly the
    "patient_present" fragmentation problem), because later stages see a
    already-smoothed signal, not raw noisy per-frame logits.

Alternative considered, not implemented: ASFormer (Yi et al., BMVC 2021) /
LTContext (Bahrami et al., ICCV 2023) windowed-local + sparse-global
attention. Not used here because (a) attention-based action segmentation
models are reported to overfit on small datasets without careful
regularization, (b) a dual-dilated TCN trains in CPU-seconds on this
prototype's synthetic data, and (c) causal masking is native to convolution
padding, whereas causal attention needs an explicit mask - simpler to keep
correct and to defend in an interview.

Multi-view fusion: per-camera features are projected through a SHARED linear
layer (same anatomy, different viewpoint - no reason for camera-specific
weights), then combined via attention-weighted pooling across the (1-3)
available cameras at each timestep (a lightweight analog of Trans-SVNet's
(Gao et al., MICCAI 2021) cross-attention fusion, scoped to the view axis).
Robustness to a camera being fully occluded during "operation" comes from
VIEW-DROPOUT TRAINING: during training, camera(s) are randomly masked out of
the attention pool, forcing the model to never structurally depend on any
single view. At inference the same trained model handles 1, 2, or 3 cameras
with no camera-count-specific branching.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.config import ModelConfig


class CameraFusion(nn.Module):
    """Shared per-camera projection + attention-weighted pooling across the
    camera axis, robust to missing/occluded views via view-dropout training."""

    def __init__(self, feature_dim: int, fusion_dim: int, view_dropout_prob: float):
        super().__init__()
        self.proj = nn.Linear(feature_dim, fusion_dim)
        # A single learned "query" vector scores each camera's projected
        # features at each timestep - the lightweight attention-pooling
        # mechanism (cf. Ilse et al. attention-MIL pooling), not a full
        # multi-head cross-attention, kept simple for CPU speed/explainability.
        self.query = nn.Parameter(torch.randn(fusion_dim) * (fusion_dim**-0.5))
        self.fusion_dim = fusion_dim
        self.view_dropout_prob = view_dropout_prob

    def forward(self, features: torch.Tensor, camera_mask: torch.Tensor) -> torch.Tensor:
        """features: [B, C, T, feature_dim], camera_mask: [B, C] (1=real camera).
        Returns fused: [B, T, fusion_dim]."""
        proj_feats = self.proj(features)  # [B, C, T, fusion_dim]

        effective_mask = camera_mask
        if self.training and self.view_dropout_prob > 0:
            drop = (torch.rand_like(camera_mask) < self.view_dropout_prob).float()
            candidate_mask = camera_mask * (1.0 - drop)
            # Never drop every camera for a case - camera_mask always has
            # >=1 real camera (min_cameras=1), so if dropout would zero all
            # of them, fall back to the undropped mask for that case only.
            all_dropped = candidate_mask.sum(dim=1) == 0
            effective_mask = torch.where(all_dropped.unsqueeze(1).expand_as(camera_mask), camera_mask, candidate_mask)

        scores = torch.einsum("bctf,f->bct", proj_feats, self.query) / (self.fusion_dim**0.5)
        scores = scores.masked_fill(effective_mask.unsqueeze(-1) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=1)  # softmax over the camera axis
        fused = torch.einsum("bct,bctf->btf", attn, proj_feats)
        return fused


class CausalDepthwiseSeparableConv1d(nn.Module):
    """Depthwise (per-channel, cheap) + pointwise (1x1, mixes channels)
    dilated conv, LEFT-padded only so output at time t depends solely on
    inputs at times <= t - this is what makes the whole network causal /
    online-deployable, verified directly by tests/test_model.py's causality
    test rather than merely claimed in the report."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, groups=channels, bias=False
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, T]
        x = F.pad(x, (self.left_pad, 0))
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DualDilatedResidualLayer(nn.Module):
    """One MS-TCN++-style residual layer: two causal dilated branches with
    MIRRORED dilation schedules (one grows with depth, one shrinks),
    concatenated and merged back to hidden_dim, added residually."""

    def __init__(self, hidden_dim: int, kernel_size: int, dilation_a: int, dilation_b: int, dropout: float):
        super().__init__()
        self.branch_a = CausalDepthwiseSeparableConv1d(hidden_dim, kernel_size, dilation_a)
        self.branch_b = CausalDepthwiseSeparableConv1d(hidden_dim, kernel_size, dilation_b)
        self.merge = nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, hidden_dim, T]
        a = F.relu(self.branch_a(x))
        b = F.relu(self.branch_b(x))
        out = self.merge(torch.cat([a, b], dim=1))
        out = self.dropout(out)
        return x + out


def make_dual_dilated_stack(hidden_dim: int, kernel_size: int, num_layers: int, dropout: float) -> nn.ModuleList:
    """Dilation schedule: branch A grows 2^l (long-range context, deepest
    layer sees ~2*(2^(L-1))-ish frames back), branch B mirrors as
    2^(L-1-l) (local precision available at every depth, not just shallow
    layers) - see module docstring for why this dual schedule is the point."""
    layers = []
    for l in range(num_layers):
        dilation_a = 2**l
        dilation_b = 2 ** (num_layers - 1 - l)
        layers.append(DualDilatedResidualLayer(hidden_dim, kernel_size, dilation_a, dilation_b, dropout))
    return nn.ModuleList(layers)


class PhaseSegmentationModel(nn.Module):
    """Full pipeline: CameraFusion -> stage-1 prediction generator -> N
    refinement stages. Returns logits from EVERY stage (used for deep
    supervision during training, see train.py) - inference uses the last
    stage's logits, which have been iteratively refined and are the
    cleanest of the set."""

    def __init__(self, model_cfg: ModelConfig, feature_dim: int, num_classes: int):
        super().__init__()
        self.fusion = CameraFusion(feature_dim, model_cfg.fusion_dim, model_cfg.view_dropout_prob)
        self.input_proj = nn.Conv1d(model_cfg.fusion_dim, model_cfg.hidden_dim, kernel_size=1)
        self.stage1_layers = make_dual_dilated_stack(
            model_cfg.hidden_dim, model_cfg.kernel_size, model_cfg.stage1_layers, model_cfg.dropout
        )
        self.stage1_head = nn.Conv1d(model_cfg.hidden_dim, num_classes, kernel_size=1)

        self.refine_input_projs = nn.ModuleList(
            [nn.Conv1d(num_classes, model_cfg.hidden_dim, kernel_size=1) for _ in range(model_cfg.num_refine_stages)]
        )
        self.refine_layers = nn.ModuleList(
            [
                make_dual_dilated_stack(model_cfg.hidden_dim, model_cfg.kernel_size, model_cfg.refine_layers, model_cfg.dropout)
                for _ in range(model_cfg.num_refine_stages)
            ]
        )
        self.refine_heads = nn.ModuleList(
            [nn.Conv1d(model_cfg.hidden_dim, num_classes, kernel_size=1) for _ in range(model_cfg.num_refine_stages)]
        )

    def forward(self, features: torch.Tensor, camera_mask: torch.Tensor) -> list[torch.Tensor]:
        """features: [B, C, T, feature_dim], camera_mask: [B, C].
        Returns a list of per-stage logits, each [B, T, num_classes]
        (time-major - the shape postprocess.py and metrics.py expect)."""
        fused = self.fusion(features, camera_mask)  # [B, T, fusion_dim]
        x = fused.transpose(1, 2)  # [B, fusion_dim, T] - channels-first for conv1d
        x = self.input_proj(x)
        for layer in self.stage1_layers:
            x = layer(x)
        stage1_logits = self.stage1_head(x)  # [B, num_classes, T]

        all_logits = [stage1_logits]
        prev_logits = stage1_logits
        for input_proj, layers, head in zip(self.refine_input_projs, self.refine_layers, self.refine_heads):
            h = input_proj(F.softmax(prev_logits, dim=1))
            for layer in layers:
                h = layer(h)
            logits = head(h)
            all_logits.append(logits)
            prev_logits = logits

        return [logits.transpose(1, 2) for logits in all_logits]  # -> [B, T, num_classes]


def smoothing_loss(logits: torch.Tensor, tau: float = 4.0) -> torch.Tensor:
    """MS-TCN's truncated-MSE flicker penalty: squared frame-to-frame
    log-probability differences, clamped at tau^2 so a LEGITIMATE sharp
    transition (e.g. the true operation boundary) isn't over-penalized -
    only small, spurious flicker gets pushed down. tau=4.0 matches the
    MS-TCN paper's default."""
    log_probs = F.log_softmax(logits, dim=-1)  # [B, T, C]
    diffs = log_probs[:, 1:, :] - log_probs[:, :-1, :]
    return torch.clamp(diffs**2, max=tau**2).mean()


def compute_loss(all_stage_logits: list[torch.Tensor], labels: torch.Tensor, smoothing_weight: float) -> torch.Tensor:
    """Deep supervision: every stage's cross-entropy + smoothing loss
    contributes equally, matching MS-TCN training practice (each stage
    should independently learn a decent segmentation, not just the last)."""
    total = torch.zeros((), device=labels.device)
    for logits in all_stage_logits:
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        total = total + ce + smoothing_weight * smoothing_loss(logits)
    return total / len(all_stage_logits)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
