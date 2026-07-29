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
modes ("patient_present" fragmentation, "operation" occlusion). See
[`MODEL_CARD.md`](MODEL_CARD.md) for intended use, training data
provenance, measured performance, and known limitations.

## Production-readiness scope: what's real here vs. what's architecture

This repo is a training/evaluation pipeline, not a deployed service - so
"production readiness" means two different things depending on which layer
you're asking about, and it's worth being explicit about the line rather
than blurring it with decorative code that doesn't actually do anything.

**Implemented and tested in this repo** (`tests/test_robustness.py`,
`tests/test_serve.py`):

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
- **Checkpoint backup rotation** — the previous checkpoint is backed up
  with a timestamped filename (microsecond precision - a first version
  collided on rapid successive saves within the same second, caught by its
  own test) before being overwritten, capped at the last 3 versions
  (`train.save_checkpoint_with_backup`). No silent, unrecoverable overwrite
  of the last good model.
- **Structured logging** — every entrypoint logs to both console and a
  rotating file (`outputs/logs/<name>.log`, `src/logging_config.py`),
  separate from the human-facing tabular CLI output (the operational/audit
  trail vs. the command's actual result - see the module docstring for why
  those are kept distinct rather than conflated).
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
- **A real local serving API** (`src/serve.py`, FastAPI) — mirrors the
  SageMaker Real-Time endpoint pattern from Sec 4.3.1: a `/predict`
  endpoint loading the trained checkpoint, request-size limiting, request
  logging, a global exception handler that logs full detail server-side
  while returning a safe message to the client, and graceful degradation
  if the checkpoint is missing/corrupted (`/health` reports why instead of
  the process crash-looping). See [Local API demo](#local-api-demo) below.
- **A real `/metrics` endpoint** (`src/serve.py`) — Prometheus-format
  scrape endpoint: request counts by endpoint/status, inference latency,
  and the ground-truth-free model-quality signals Sec 4.4 specifically
  names (per-frame softmax-margin confidence, per-class segment
  fragmentation, per-camera availability). No Prometheus server is
  deployed to scrape it here (see the "not implemented" list below), but
  the metrics themselves are real, wired into the actual inference path,
  and tested (`tests/test_serve.py`).
- **Containerization** (`Dockerfile`) — non-root user, `HEALTHCHECK`
  against `/health`, CPU-only base image. Build verified end-to-end both
  locally and on the GitHub Actions runner - see [Docker](#docker) below.
- **CI** (`.github/workflows/ci.yml`) — lint, a pre-commit hook check,
  the full test suite with coverage, a pipeline smoke test, a dependency
  vulnerability scan (`pip-audit`) against the exact locked dependency set
  the Docker image ships, a separate Docker build+smoke-test job, and a
  report-build job. All jobs verified green on a real GitHub Actions
  runner - see [CI/CD](#cicd) below.
- **Dependency vulnerability scanning** (`pip-audit`) — run locally and as
  its own CI job against `requirements-lock.txt`. The first real run found
  4 CVEs (`idna`, `pillow`, `requests`, `urllib3` - old versions pulled in
  transitively by the PDF-rendering stack); they were fixed with explicit
  version floors in `requirements.txt` (see
  [Design choices worth flagging](#design-choices-worth-flagging)), not
  just logged and left.
- **Pre-commit hooks** (`.pre-commit-config.yaml`) — the same `ruff check`
  CI enforces, installable locally (`.venv/bin/pre-commit install`) and
  verified in CI itself (`pre-commit run --all-files`), so "the hook
  config exists" and "the hook actually runs" are both demonstrated, not
  just the former asserted.

**Described in the architecture report, not implemented here** (because it
requires infrastructure that doesn't exist in a training-script repo, not
because it was skipped) - Sec 4.3-4.5:

- **Retries and circuit breakers between distributed services** — belongs
  at the AWS service boundary (Lambda/Fargate ↔ SageMaker, Sec 4.3.1), not
  inside a synchronous training/eval script that has no distributed calls
  to retry.
- **Autoscaling and load balancing across a GPU pool** — Application Auto
  Scaling sizing the SageMaker Real-Time endpoint pool (Sec 4.3.1) is a
  deployment-time concern; there's no pool to scale behind one local
  serving process.
- **Multi-tenant concurrent request batching at scale** — Triton's dynamic
  batching serving many OR streams' *concurrent HTTP requests* (Sec 4.3.2)
  is the same batching principle this repo implements and measures for
  *offline* eval (`evaluate.compute_batch_logits`), just not wired into
  `serve.py`'s request path - the local demo serves one request at a time,
  which is the right scope for a demo but not what a real multi-tenant
  endpoint would do.
- **A running monitoring/alerting stack** — Sec 4.4 specifies *what* to
  monitor (confidence collapse, camera desync, fragmentation rate,
  per-hospital breakdown) and *why*. `GET /metrics` (above) is a real step
  toward that - the signals themselves are computed and exposed - but
  there's no actual Prometheus server, CloudWatch integration, or
  alerting/paging infrastructure deployed here to scrape and act on it.
- **Deployed security controls** (encryption, IAM, VPC, secrets
  management, audit logging - Sec 4.5) — these secure deployed AWS
  resources; there are none deployed here to secure. The security controls
  that *do* apply to a local repo/container (not trusting a pickled
  checkpoint, running as non-root) are implemented, above.

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
| **Run tests** | `make test` → `66 tests, ~2 min with coverage instrumentation, fully deterministic` |
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
cd surgery-workflow-segmentation
make bootstrap        # uv venv (Python 3.12) + exactly the pinned lock, no sudo
make test              # 66 tests
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
make bootstrap      # uv-managed Python 3.12 venv + exactly the pinned lock (no sudo required)
make bootstrap-dev  # + pip-audit/pytest-cov/pre-commit (dev/CI tools, kept out of the Docker image)
make train          # ~30-35s on CPU: trains the temporal model on synthetic data
make evaluate       # raw-vs-postprocessed metrics on the held-out validation split
make error-analysis # noisy-vs-clean per-class breakdown - the "proof of understanding" script
make test           # 66 tests, fully deterministic
make coverage       # tests + line/branch coverage report (~90% - see CI's coverage-report artifact)
make audit          # pip-audit against the exact locked runtime dependency set
make hooks-install  # installs the pre-commit git hook (same ruff check CI runs)
make report         # renders the AWS diagram + builds technical_architecture_report.pdf
make serve          # starts the local API demo on http://127.0.0.1:8000
make docker-build   # builds the Docker image (needs a running Docker daemon)
make docker-run     # runs it, publishing port 8000
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

## Local API demo

`src/serve.py` is a real FastAPI service loading the trained checkpoint,
mirroring the SageMaker Real-Time endpoint pattern from report Sec 4.3.1 -
runnable and curl-testable locally. It is **not** a deployed cloud
endpoint (see the [production-readiness scope](#production-readiness-scope-whats-real-here-vs-whats-architecture)
section above for why a real AWS deployment is out of scope here).

```bash
make serve   # or: .venv/bin/python -m uvicorn src.serve:app --host 127.0.0.1 --port 8000
```

Then, from another terminal:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","model_loaded":true,"load_error":null,"checkpoint_path":"outputs/checkpoint.pt"}

curl -X POST "http://127.0.0.1:8000/predict/synthetic?seed=42"
# generates a synthetic case server-side and returns its predicted timeline -
# the easiest way to exercise the API without hand-building a feature payload

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[[0.1, ...]]], "camera_mask": [[1.0, ...]]}'
# the real production contract: raw feature/camera_mask arrays, matching
# what the live-inference data path (Sec 4.3.1) would actually hand it

curl http://127.0.0.1:8000/metrics  # Prometheus-format scrape endpoint - report Sec 4.4

curl http://127.0.0.1:8000/docs   # interactive OpenAPI/Swagger UI
```

If no checkpoint exists yet (`outputs/checkpoint.pt`), the server still
starts - `/health` reports `"status": "degraded"` and `/predict` returns
`503` with a clear message, instead of crashing. Run `make train` first,
or restart the server after training.

## Docker

```bash
make docker-build
make docker-run   # publishes port 8000; visit http://127.0.0.1:8000/health
```

The image has no checkpoint baked in (`outputs/` is gitignored/dockerignored
by design - see [Docker containerization](#implemented-and-tested-in-this-repo)
above), so a freshly-built container starts in the same graceful
`"degraded"` state described above. To serve real predictions, mount a
directory containing a trained checkpoint:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/outputs:/app/outputs" surgery-workflow-segmentation
```

**Verified, not just written.** `docker build` was run end-to-end (both
locally via the Windows-side Docker Desktop binary and again on the GitHub
Actions runner in CI - ~5-17 min depending on cache/network, image
`surgery-workflow-segmentation:latest`), and a running container's
`/health` was confirmed to report `"degraded"` with no checkpoint mounted,
exactly as designed. One real bug was caught along the way:
`requirements-lock.txt` (`uv pip freeze` output) drops the
`--extra-index-url` pragma needed to resolve `torch==...+cpu`, fixed by
passing it explicitly on the Dockerfile's `pip install` line - a bug that
manual review alone would not have caught, only an actual build attempt did.

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`, four jobs:

- **Lint & Test** — `ruff check`, `pre-commit run --all-files` (proves the
  hooks in `.pre-commit-config.yaml` actually execute, not just parse),
  the full test suite with coverage (`pytest-cov`, report uploaded as an
  artifact), and a pipeline smoke test (train → evaluate →
  error-analysis, with logs uploaded as an artifact).
- **Dependency vulnerability scan (pip-audit)** — audits the exact locked
  dependency set the Docker image ships (`requirements-lock.txt`), not
  just the abstract floors in `requirements.txt`.
- **Docker build + smoke test** — builds the image, starts it with no
  checkpoint mounted, and asserts `/health` correctly reports `"degraded"`.
- **Build technical architecture report** — renders the diagram and PDF,
  uploaded as an artifact.

**Verified, not just written.** All four jobs are green on a real GitHub
Actions runner (`github.com/farahfarchoukh/surgery-workflow-segmentation`,
Actions tab), including a failure found and fixed on that runner but not
locally: two tests asserted bit-exact equality (`np.array_equal`) between
batched and single-case inference outputs, which passed on this dev box's
CPU but failed on GitHub's different runner CPU due to ordinary
floating-point non-associativity across different convolution kernel
selections - a real lesson that CPU bit-exactness across batch sizes isn't
a guarantee PyTorch actually makes. Fixed by asserting numerical closeness
(`np.allclose`, `rtol=1e-4`) instead of bit-exactness, which is the
guarantee that actually holds.

## Repository structure

```
config/default.yaml             single reproducibility artifact - every
                                 reported number should trace back to this
src/
  config.py                     dataclasses + YAML loader (with __post_init__
                                 validation), PHASE_LABELS, strict
                                 forward-only transition matrix
  data.py                       Ingestion & Feature Stubs (component 1)
  sync.py                       multi-camera timestamp jitter + frame-loss
                                 simulation and the synchronization/alignment
                                 layer that recovers a per-timestep mask
  model.py                      Temporal Classification Layer + multi-view
                                 fusion (component 2)
  train.py                      training loop + checkpoint backup rotation
  postprocess.py                 Segmentation Timeline Generator (component 3)
  metrics.py                     Dual Metric Stack (component 4)
  evaluate.py                    orchestrates data -> model -> postprocess -> metrics
  error_analysis.py              Mock Error Analysis Script (component 5)
  logging_config.py              structured logging setup, shared by every entrypoint
  serve.py                       local FastAPI serving demo (SageMaker-endpoint
                                  pattern) + /metrics Prometheus endpoint
tests/                           66 tests, hand-computed known answers where it matters
report/
  technical_architecture_report.md   Deliverable 2 source
  diagrams/aws_architecture.{mmd,png}
  build_pdf.py                   pure-Python PDF export
scripts/make_report_assets.py    renders the AWS diagram PNG
MODEL_CARD.md                    intended use, limitations, training data, performance
Dockerfile, .dockerignore        containerization (see Docker section above)
.github/workflows/ci.yml         CI: lint, pre-commit check, test+coverage, pip-audit,
                                  pipeline smoke test, Docker build, report build
.pre-commit-config.yaml          local git hooks (ruff check) - CI runs the same ones
requirements.txt                 runtime deps (abstract floors, incl. CVE-fix pins)
requirements-lock.txt            exact pinned versions - what the Docker image installs
requirements-dev.txt             + pip-audit/pytest-cov/pre-commit, kept out of the image
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
- **`pip-audit` found real CVEs, fixed with explicit version floors, not
  ignored.** The first run flagged `idna==3.4`, `pillow==12.2.0`,
  `requests==2.28.1`, `urllib3==1.26.13` - none direct dependencies, all
  pulled in transitively by the PDF-rendering stack (`xhtml2pdf`/`svglib`/
  `pyhanko`), which is exactly why they'd be easy to miss without actually
  running a scanner. Fixed by adding explicit lower-bound floors for these
  four in `requirements.txt` (with a comment explaining they're transitive,
  not direct) and regenerating `requirements-lock.txt` from a clean
  throwaway venv (`make relock`) so the resolved versions are pinned, not
  just floor-constrained. `pip-audit` now reports zero known
  vulnerabilities against the exact locked set; re-checked in CI on every
  push so a future transitive-dependency CVE doesn't go unnoticed.
- **Coverage is reported honestly, not gated at a number that gets gamed.**
  `make coverage` → ~90% line coverage. The gaps are specific and named,
  not hidden: a handful of config-validation branches in `src/config.py`
  that are individually exhaustive but not all independently exercised,
  and `__main__` CLI entry blocks that run in the pipeline smoke test
  (`make train`/`evaluate`/`error-analysis`) but not under `pytest` itself.
  No `--cov-fail-under` threshold is set - the number is a signal to read,
  not a gate to satisfy by writing low-value tests.
- **Ruff line-length is 130, not the 88/100 default.** This codebase
  deliberately writes longer explanatory comments/docstrings (the WHY
  behind non-obvious decisions) rather than fragmenting them - see
  `pyproject.toml`. The whole codebase passes `ruff check .` cleanly (CI
  gate); the config was adjusted to fit this repo's actual documentation
  style rather than mechanically wrapping ~130 lines to fit a default that
  doesn't match how this project chooses to write comments.
- **What's genuinely verified vs. architecture-only, stated explicitly.**
  Every claim in this README distinguishes what was actually run and
  observed (tests, benchmarks, curl calls against a live local server, a
  real `docker build`, a real green run on GitHub Actions) from what's
  deliberately out of scope for a training-script repo and is instead
  covered as architecture in the report (a deployed monitoring/alerting
  stack, autoscaling, IAM/VPC controls) - see the
  [production-readiness scope](#production-readiness-scope-whats-real-here-vs-whats-architecture)
  section at the top for the full breakdown. The goal is that nothing here
  is asserted with more confidence than the evidence actually supports.
