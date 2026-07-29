# Single-stage build: the pinned CPU-only requirements.txt is small enough
# (no CUDA toolkit, no build-from-source deps) that a multi-stage build's
# extra complexity isn't worth it here - see README's production-readiness
# scope section for the reasoning behind other scope calls like this one.
FROM python:3.12-slim

# Match the CPU-only PyTorch story from requirements.txt/README: this image
# is never expected to see a GPU, so a slim base (no CUDA) is the correct
# choice, not a compromise.
WORKDIR /app

# Install dependencies before copying source: with the Docker layer cache,
# `docker build` after a source-only change skips reinstalling ~600MB of
# torch/matplotlib/etc. and only re-copies src/ - this ordering is the
# single highest-leverage thing for iteration speed on this image.
COPY requirements.txt requirements-lock.txt ./
# --extra-index-url is passed explicitly on the CLI, not relied on being
# embedded in requirements-lock.txt: `uv pip freeze` (which generated that
# file) only emits `package==version` lines, dropping the index-url pragma
# that requirements.txt has. Without this flag, a plain `pip install -r
# requirements-lock.txt` would look for `torch==2.13.0+cpu` on the default
# PyPI index, where that exact local-version build doesn't exist, and fail.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt

COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml ./

# Run as a non-root user - a real, minimal security improvement (report
# Sec 4.5's least-privilege principle, applied at the container level: no
# reason the process serving requests needs root inside its own container).
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/outputs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Matches serve.py's own liveness contract: /health responds even when the
# model failed to load (status "degraded" instead of "ok"), so this
# healthcheck reflects "is the process alive and answering," not "is a
# model loaded" - that distinction is exactly what report Sec 4.4 argues
# monitoring needs to make (infra health vs. model/data quality are
# different questions).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Default: start the serving API. Override for other pipeline stages, e.g.:
#   docker run --rm surgery-workflow-segmentation python -m src.train
#   docker run --rm surgery-workflow-segmentation python -m src.evaluate
#   docker run --rm surgery-workflow-segmentation pytest -q
# A trained checkpoint must exist at outputs/checkpoint.pt for the serving
# command to have a model to load - mount it in (see README) or run
# `python -m src.train` first inside the same container/volume.
CMD ["python", "-m", "uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
