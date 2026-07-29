"""Local model-serving demo: a real HTTP API in front of the trained
checkpoint, mirroring the SageMaker Real-Time endpoint pattern described in
the report (Sec 4.3.1) - runnable and testable via curl, not a live cloud
deployment (see README's Production-readiness scope section for why a real
AWS deployment is out of scope here).

Two prediction routes:
  - POST /predict/synthetic - generates a synthetic case server-side (via
    the same SyntheticSurgeryGenerator training/eval use) and returns its
    prediction. Easiest way to demo the API without hand-building a
    128x240x3-dim feature payload.
  - POST /predict - accepts real feature/camera_mask arrays directly, the
    actual production contract: a real deployment receives pre-extracted
    embeddings (Sec 1.1/4.3.1), not raw video, so this endpoint's input
    shape is exactly what the live-inference data path would hand it.

Graceful startup: if the checkpoint is missing/corrupted, the server still
starts (so orchestration health checks can see WHY it's not ready) rather
than crashing the whole process - /health reports model_loaded=false and
/predict returns 503, instead of the process refusing to boot at all.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import PHASE_LABELS, ExperimentConfig, build_allowed_transition_matrix
from src.data import SyntheticSurgeryGenerator
from src.evaluate import CorruptedCheckpointError, load_model
from src.logging_config import setup_logging
from src.model import PhaseSegmentationModel
from src.postprocess import generate_timeline

logger = setup_logging("serve")

CONFIG_PATH = Path("config/default.yaml")
CHECKPOINT_PATH = Path("outputs/checkpoint.pt")

state: dict = {"model": None, "cfg": None, "generator": None, "allowed": None, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("server starting up: config=%s checkpoint=%s", CONFIG_PATH, CHECKPOINT_PATH)
    try:
        cfg = ExperimentConfig.from_yaml(CONFIG_PATH)
        model = load_model(cfg, CHECKPOINT_PATH)
        state["model"] = model
        state["cfg"] = cfg
        state["generator"] = SyntheticSurgeryGenerator(cfg.data)
        state["allowed"] = build_allowed_transition_matrix()
        logger.info("model loaded successfully")
    except (FileNotFoundError, CorruptedCheckpointError) as e:
        # Deliberately don't re-raise: a missing/corrupt checkpoint shouldn't
        # take the whole process down. /health reports the problem, /predict
        # returns 503 - an orchestrator (k8s readiness probe, ALB health
        # check) can see exactly why this instance isn't serving yet instead
        # of just seeing a crash-looping container with no diagnostic.
        logger.error("model failed to load at startup: %s", e)
        state["load_error"] = str(e)
    yield
    logger.info("server shutting down")


app = FastAPI(title="Proximie OR Phase Segmentation - Local Serving Demo", lifespan=lifespan)

MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10MB - generous for any realistic feature payload at this model's scale


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    # Rejected on the Content-Length header alone, before the body is read
    # or JSON-parsed: checking array shape AFTER building a numpy array
    # from the payload (as the route handlers used to) means an oversized
    # payload has already cost the memory/CPU to parse and convert by the
    # time it's rejected. A missing/absent Content-Length (e.g. chunked
    # transfer encoding) is let through to the route handler's own
    # validation rather than guessed at here.
    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > MAX_REQUEST_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": f"request body {content_length} bytes exceeds the {MAX_REQUEST_BODY_BYTES}-byte limit"},
        )
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    logger.info("%s %s -> %d (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Fail loudly internally (full exception logged server-side), fail
    # safely externally (no internal detail leaked to the client) - the
    # same "clear diagnostic, safe surface" split as the CorruptedCheckpointError/
    # TrainingDivergedError pattern in train.py/evaluate.py, applied at the API boundary.
    logger.exception("unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error - see server logs"})


class Segment(BaseModel):
    label: str
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float


class PredictionResponse(BaseModel):
    num_frames: int
    raw_labels: list[str]
    segments: list[Segment]
    model_param_count: int


class RawPredictRequest(BaseModel):
    features: list[list[list[float]]] = Field(..., description="[cameras, frames, feature_dim]")
    camera_mask: list[list[float]] = Field(..., description="[cameras, frames], 1.0 = camera available at that frame")


def _require_model() -> PhaseSegmentationModel:
    if state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {state['load_error'] or 'unknown startup error'}. "
            f"Train one first (`make train`) and restart the server.",
        )
    return state["model"]


def _run_inference(features: torch.Tensor, camera_mask: torch.Tensor) -> PredictionResponse:
    model = _require_model()
    cfg: ExperimentConfig = state["cfg"]
    with torch.no_grad():
        logits = model(features.unsqueeze(0), camera_mask.unsqueeze(0))[-1][0]  # [T, C]
    logits_np = logits.numpy()

    raw_labels_idx = logits_np.argmax(axis=-1)
    segments = generate_timeline(logits_np, cfg.eval, state["allowed"])

    return PredictionResponse(
        num_frames=logits_np.shape[0],
        raw_labels=[PHASE_LABELS[i] for i in raw_labels_idx],
        segments=[
            Segment(
                label=PHASE_LABELS[s.label],
                start_frame=s.start,
                end_frame=s.end,
                start_seconds=s.start * cfg.data.seconds_per_frame,
                end_seconds=s.end * cfg.data.seconds_per_frame,
            )
            for s in segments
        ],
        model_param_count=sum(p.numel() for p in model.parameters()),
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if state["model"] is not None else "degraded",
        "model_loaded": state["model"] is not None,
        "load_error": state["load_error"],
        "checkpoint_path": str(CHECKPOINT_PATH),
    }


@app.get("/")
def root() -> dict:
    return {
        "service": "Proximie OR Phase Segmentation - local serving demo",
        "note": "Local demonstration of the SageMaker Real-Time endpoint pattern (report Sec 4.3.1), "
        "not a deployed cloud service.",
        "endpoints": ["/health", "/predict/synthetic (POST)", "/predict (POST)", "/docs"],
    }


@app.post("/predict/synthetic", response_model=PredictionResponse)
def predict_synthetic(seed: int = 0) -> PredictionResponse:
    """Generates a synthetic case server-side (deterministic given `seed`)
    and returns its prediction - the easiest way to exercise this API
    without constructing a full feature payload by hand."""
    _require_model()
    generator: SyntheticSurgeryGenerator = state["generator"]
    case = generator.generate(case_seed=seed)
    return _run_inference(case.features, case.camera_mask)


@app.post("/predict", response_model=PredictionResponse)
def predict(req: RawPredictRequest) -> PredictionResponse:
    """Accepts real feature/camera_mask arrays directly - the actual
    production contract (Sec 1.1/4.3.1): a live deployment receives
    pre-extracted embeddings, not raw video."""
    _require_model()
    cfg: ExperimentConfig = state["cfg"]

    try:
        # A ragged nested list (inconsistent inner-list lengths - e.g. one
        # camera with a different frame count than another) raises a numpy
        # ValueError here. Left uncaught, that fell through to the global
        # exception handler as a generic 500 - a client validation mistake
        # deserves a clean 422 with a real explanation, not "internal error,
        # see server logs" for something the server did nothing wrong to cause.
        features_np = np.array(req.features, dtype=np.float32)
        mask_np = np.array(req.camera_mask, dtype=np.float32)
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"features/camera_mask must be regular (non-ragged) nested arrays: {e}",
        ) from e

    if features_np.ndim != 3 or features_np.shape[-1] != cfg.data.feature_dim:
        raise HTTPException(
            status_code=422,
            detail=f"features must have shape [cameras, frames, {cfg.data.feature_dim}], got {list(features_np.shape)}",
        )
    if mask_np.shape != features_np.shape[:2]:
        raise HTTPException(
            status_code=422,
            detail=f"camera_mask shape {list(mask_np.shape)} must match "
            f"features' [cameras, frames] = {list(features_np.shape[:2])}",
        )

    return _run_inference(torch.from_numpy(features_np), torch.from_numpy(mask_np))
