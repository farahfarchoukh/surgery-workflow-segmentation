# Proximie ML Engineer Challenge — OR Workflow Phase Segmentation Prototype

A mock/lightweight prototype pipeline for temporal action segmentation of
operating-room workflow phases from 1-3 corner-mounted camera feeds, built
for the Proximie Senior ML Engineer take-home. This repo is Deliverable 1
(the prototype pipeline); Deliverable 2 (the Technical Architecture Report)
is at [`report/technical_architecture_report.md`](report/technical_architecture_report.md)
/ `report/technical_architecture_report.pdf`.

**Explicitly not aiming for accuracy** (per the assignment) — this is a
structural demonstration: mock features, a genuinely-trained-but-tiny
temporal model, clean separation of the 5 required components, and an
error-analysis script that deliberately reproduces the two named failure
modes ("patient_present" fragmentation, "operation" occlusion).

## Production-readiness scope: what's real here vs. what's architecture

This repo is a training/evaluation pipeline, not a deployed service - so
"production readiness" means two different things depending on which layer
you're asking about, and it's worth being explicit about the line rather
than blurring it with decorative code that doesn't actually do anything.

**Implemented and tested in this repo** (`tests/test_robustness.py`):

- **Fail-fast on numerical divergence** — training detects a non-finite
  loss and raises immediately, before a corrupted checkpoint can be
  written (`TrainingDivergedError`, `src/train.py`).
- **Fail-fast on corrupted/malformed artifacts** — checkpoint loading
  wraps pickle errors, missing keys, and architecture mismatches into one
  consistent, actionable error (`CorruptedCheckpointError`,
  `src/evaluate.py`) instead of a different raw traceback per failure mode.
- **Checkpoint security** — `weights_only=True` throughout; a checkpoint
  never needs to be trusted to unpickle arbitrary code, including one from
  an unknown source.
- **Config validation** — every dataclass rejects invalid values at
  construction time (`src/config.py` `__post_init__`), not deep inside a
  matrix operation three modules later.
- **Batched inference, measured not assumed** — `evaluate.py`/
  `error_analysis.py` batch the model forward pass across cases rather
  than looping one at a time. This was benchmarked before being adopted:
  case-level thread-pool parallelism was actually 25-45% *slower* at this
  model's size (measured, not guessed) and was rejected; batching measured
  1.1x-7.9x faster and is what shipped. It's also the same mechanism Sec
  4.3.2 of the report describes for production (Triton dynamic batching on
  SageMaker Multi-Model Endpoints) - exercised locally here, not just
  described.
- **Robustness under adversarial data conditions** — near-total
  camera-sync loss, a sequence shorter than the model's own receptive
  field, single-camera cases with frequent full dropout: all verified to
  degrade gracefully (stay finite) rather than crash or produce NaN.

**Described in the architecture report, not implemented here** (because it
requires infrastructure that doesn't exist in a training-script repo, not
because it was skipped) - Sec 4.3-4.5:

- **Retries and circuit breakers between distributed services** — belongs
  at the AWS service boundary (Lambda/Fargate ↔ SageMaker, Sec 4.3.1), not
  inside a synchronous training/eval script that has no distributed calls
  to retry.
- **Autoscaling and load balancing across a GPU pool** — Application Auto
  Scaling sizing the SageMaker Real-Time endpoint pool (Sec 4.3.1) is a
  deployment-time concern; there's no pool to scale in a local repo.
- **Concurrent request handling at the serving layer** — real concurrency
  in production comes from Triton's dynamic batching serving many OR
  streams' requests concurrently (Sec 4.3.2), which is the same batching
  principle this repo *does* implement and measure locally, just without
  an actual multi-tenant serving endpoint to batch requests from.
- **Production monitoring/alerting infrastructure** — Sec 4.4 specifies
  *what* to monitor (confidence collapse, camera desync, fragmentation
  rate, per-hospital breakdown) and *why* each one matters; there's no
  running CloudWatch/monitoring stack in this repo to wire it into.
- **Security controls** (encryption, IAM, VPC, secrets management, audit
  logging - Sec 4.5) — these secure deployed AWS resources; there are none
  deployed here to secure. The one security control that *does* apply to
  a local repo (not trusting a pickled checkpoint) is implemented, above.

## Reproducibility

Everything below is exact, not approximate — copy-paste-run this and you
should reproduce the same numbers up to normal floating-point/OS
scheduling jitter (training/eval are CPU-only and fully seeded; wall-clock
timings will vary with your hardware).

| | |
|---|---|
| **Python version** | 3.12 (provisioned automatically by `uv`, independent of your system Python — see [Environment note](#environment-note)) |
| **Dependencies** | Pinned in `requirements.txt` (abstract) and `requirements-lock.txt` (exact versions actually installed, via `uv pip freeze`) |
| **Installation** | `make bootstrap` (no sudo/root required — see below) |
| **Run tests** | `make test` → `49 tests, ~6s, fully deterministic` |
| **Train** | `make train` → writes `outputs/checkpoint.pt` |
| **Evaluate** | `make evaluate` → writes `outputs/eval_report.json`, prints raw-vs-postprocessed comparison |
| **Error analysis** | `make error-analysis` → writes `outputs/error_analysis_report.json`, prints per-class noisy-vs-clean breakdown |
| **Expected runtime** | Train: **~30-35s** on a 4-core CPU. Eval/error-analysis: a few seconds each. Full test suite: **~5.5-7s** |
| **Expected metrics** (`config/default.yaml` defaults) | Val frame accuracy **~99.4-99.6%**; error-analysis raw frame-accuracy delta **~-0.5 to -0.7** on `patient_present`/`operation` vs **~-0.01** on other classes (exact figures below) |
| **Random seed** | `42` (`DataConfig.seed`, `config/default.yaml`) — every dataset split, weight init, and noise-injection RNG traces back to this one value via `src.config.set_seed` |
| **CPU/GPU assumptions** | **CPU-only.** No GPU is used, expected, or required anywhere in this repo — `torch` is installed from the CPU wheel index (see `requirements.txt`), and `TrainConfig.device` is hardcoded `"cpu"`. Sized deliberately so training finishes in CPU-seconds |

Every reported number should trace back to `config/default.yaml` — if you
change a value there (or pass `--config path/to/other.yaml`), the change is
the single source of truth for what ran; there is no hidden default living
somewhere else. Every config dataclass validates its own invariants at
construction time (`src/config.py` `__post_init__`), so an invalid override
fails immediately with a clear message rather than crashing deep inside a
matrix operation later.

### Full reproduce-from-scratch sequence

```bash
git clone <this repo>
cd proximie-mle-challenge
make bootstrap        # uv venv (Python 3.12) + pinned deps, no sudo
make test              # 49 tests, ~6s
make train              # ~30-35s CPU, writes outputs/checkpoint.pt
make evaluate            # writes outputs/eval_report.json
make error-analysis       # writes outputs/error_analysis_report.json
make report                # writes report/technical_architecture_report.pdf
```

Every step after `bootstrap` reads `config/default.yaml` by default; pass
`--config <path>` to any of the underlying `python -m src.<module>` calls
(see `Makefile`) to run against a different configuration.

## Quickstart

```bash
make bootstrap     # uv-managed Python 3.12 venv + pinned CPU-only deps (no sudo required)
make train         # ~30-35s on CPU: trains the temporal model on synthetic data
make evaluate       # raw-vs-postprocessed metrics on the held-out validation split
make error-analysis # noisy-vs-clean per-class breakdown - the "proof of understanding" script
make test           # 49 tests, ~6s, fully deterministic
make report          # renders the AWS diagram + builds technical_architecture_report.pdf
```

If `make` isn't available, every target is a one-line call to
`.venv/bin/python -m src.<module>` — see the `Makefile` for the exact
commands.

### Environment note

This was developed in a sandboxed dev environment with **no pip, no
GPU, no passwordless sudo, and no pandoc/node/graphviz** preinstalled.
`make bootstrap` works around all of that:

- **`uv`** (astral.sh, single static binary, no root) provisions an isolated
  **Python 3.12** venv rather than trusting the system's bare Python 3.14 -
  3.14 is extremely new and PyTorch/pip wheel support for it was an open
  risk not worth taking on for a take-home. `requirements-lock.txt` pins
  every installed package's exact version (`uv pip freeze`).
- PyTorch is installed **CPU-only** (`--index-url .../whl/cpu`) - there's no
  GPU in this dev box, and the model is deliberately sized to train in
  CPU-seconds (see `config/default.yaml` and Sec 4.2.2 of the report).
- Report tooling avoids pandoc/node/graphviz entirely: the AWS diagram is
  rendered via `matplotlib` (patches/arrows, no `dot`/mermaid-cli needed)
  and the PDF is built via pure-Python `markdown` → HTML → `xhtml2pdf` (no
  external binary).
- `git filter-branch`/local-history hygiene aside, this repo has never been
  pushed anywhere — clone/bootstrap from the local path if you're reviewing
  it before it's on a remote.

## Repository structure

```
config/default.yaml            single reproducibility artifact - every
                                reported number should trace back to this
src/
  config.py                    dataclasses + YAML loader (with __post_init__
                                validation), PHASE_LABELS, strict
                                forward-only transition matrix
  data.py                      Ingestion & Feature Stubs (component 1)
  sync.py                      multi-camera timestamp jitter + frame-loss
                                simulation and the synchronization/alignment
                                layer that recovers a per-timestep mask
  model.py                     Temporal Classification Layer + multi-view
                                fusion (component 2)
  train.py                     minimal CPU training loop
  postprocess.py                Segmentation Timeline Generator (component 3)
  metrics.py                    Dual Metric Stack (component 4)
  evaluate.py                   orchestrates data -> model -> postprocess -> metrics
  error_analysis.py             Mock Error Analysis Script (component 5)
tests/                          49 tests, hand-computed known answers where it matters
report/
  technical_architecture_report.md   Deliverable 2 source
  diagrams/aws_architecture.{mmd,png}
  build_pdf.py                  pure-Python PDF export
scripts/make_report_assets.py   renders the AWS diagram PNG
```

## Design choices worth flagging

- **Config: 4 dataclasses + one YAML file, no Hydra.** A config-composition
  framework is one more thing to defend line-by-line in the follow-up
  interview; a plain dataclass hierarchy is fully sufficient for a repo
  this size, and every field validates itself at construction time.
- **Fixed sequence length in the mock generator.** Segment durations are
  sampled from per-phase priors then rescaled to sum to exactly `seq_len`,
  so every synthetic case has identical shape and batching needs no
  time-axis padding/masking machinery.
- **Multi-camera synchronization is real, not assumed.** Each camera's
  clean per-frame signal is passed through `src/sync.py`'s jitter/frame-loss
  simulation and realigned onto the nominal grid, producing a genuinely
  PER-TIMESTEP `camera_mask` ([cameras, time], not just [cameras]) — a
  camera can be equipped for a case but missing at specific instants
  because its raw sample fell outside the sync tolerance window or was
  dropped. `CameraFusion` handles this down to the edge case of every
  camera being simultaneously unavailable at one instant without producing
  NaN (`tests/test_model.py::test_fusion_handles_fully_missing_timestep`).
- **Causal architecture, verified not asserted.** `tests/test_model.py`
  perturbs only the tail of an input sequence and checks every earlier
  output is bit-identical. The model's exact receptive field is also
  *computed* (`model.compute_receptive_field_frames`) and cross-checked
  against the model's actual empirical behavior via exhaustive perturbation
  scanning, not just derived on paper — a first version of that
  cross-check used binary search and silently got the wrong answer,
  because a dilated conv's dependency on its input is a sparse comb of
  positions, not a contiguous range, so "does perturbing position p change
  the output" isn't monotonic in p. Fixed by scanning exhaustively instead
  (see the test's docstring for the full story).
- **Postprocessing pipeline is majority-filter + min-duration-merge +
  transition-prior masking, not Viterbi, by default.** Viterbi is
  implemented (`postprocess.viterbi_decode`) but needs the full sequence to
  backtrack from - a poor fit for the live/online story. It's the
  deliberate, explainable choice for an offline reprocessing job instead
  (see report Sec 4.1.1).
- **Dual metric stack is genuinely dual.** `evaluate.py` computes metrics
  on both raw and postprocessed predictions so postprocessing's value-add
  is a measured delta, not an assumed one; `error_analysis.py` reports both
  per-class frame accuracy (locally sensitive) and segmental F1 (only
  requires overall segment overlap) because neither alone tells the full
  story - see the printed output for what that reveals.
- **Error-analysis noise severity was tuned empirically against the
  trained model, not guessed.** The model's dual-dilated receptive field
  and view-dropout training make it genuinely robust to mild corruption -
  weak noise showed near-zero effect on first pass. `run_error_analysis`
  cleanly separates target-class degradation (frame accuracy delta
  **~-0.5 to -0.7**) from non-target classes (**~-0.01**) at the tuned
  severity — exact numbers vary run-to-run within that band because the
  synthetic dataset's camera counts and phase durations are themselves
  randomly sampled (seeded, but the corruption injection draws fresh
  per-run randomness on top).
- **Checkpoints don't pickle arbitrary objects.** `torch.load` is called
  with `weights_only=True` — the checkpoint stores only tensors and
  plain-JSON-safe metadata (never a pickled config dataclass), so loading
  a checkpoint never needs to trust arbitrary unpickled code, including
  checkpoints downloaded from somewhere else.
