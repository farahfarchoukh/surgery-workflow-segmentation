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

## Quickstart

```bash
make bootstrap     # uv-managed Python 3.12 venv + pinned CPU-only deps (no sudo required)
make train         # ~25s on CPU: trains the temporal model on synthetic data
make evaluate       # raw-vs-postprocessed metrics on the held-out validation split
make error-analysis # noisy-vs-clean per-class breakdown - the "proof of understanding" script
make test           # 26 tests, ~5.5s, fully deterministic
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

## Repository structure

```
config/default.yaml            single reproducibility artifact - every
                                reported number should trace back to this
src/
  config.py                    dataclasses + YAML loader, PHASE_LABELS,
                                strict forward-only transition matrix
  data.py                      Ingestion & Feature Stubs (component 1)
  model.py                     Temporal Classification Layer + multi-view
                                fusion (component 2)
  train.py                     minimal CPU training loop
  postprocess.py                Segmentation Timeline Generator (component 3)
  metrics.py                    Dual Metric Stack (component 4)
  evaluate.py                   orchestrates data -> model -> postprocess -> metrics
  error_analysis.py             Mock Error Analysis Script (component 5)
tests/                          26 tests, hand-computed known answers where it matters
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
  this size.
- **Fixed sequence length in the mock generator.** Segment durations are
  sampled from per-phase priors then rescaled to sum to exactly `seq_len`,
  so every synthetic case has identical shape and batching needs no
  time-axis padding/masking machinery. Camera-axis padding (1-3 real
  cameras vs. `max_cameras`) is kept via `camera_mask`, because that IS the
  multi-view occlusion problem the model has to handle.
- **Causal architecture, verified not asserted.** `tests/test_model.py`
  perturbs only the tail of an input sequence and checks every earlier
  output is bit-identical - a direct proof of the "online-deployable" claim
  the report makes, not just a comment saying so.
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
  cleanly separates target-class degradation (~-0.67 frame accuracy) from
  non-target classes (~-0.002) at the tuned severity.
