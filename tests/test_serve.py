"""Tests for src/serve.py using FastAPI's in-process TestClient - no live
uvicorn process needed, but exercises the real ASGI app: routing,
pydantic validation, the lifespan startup hook, and the global exception
handler, exactly as a live server would."""


import pytest
from fastapi.testclient import TestClient

from src.config import DataConfig, EvalConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.train import train


@pytest.fixture()
def trained_checkpoint(tmp_path, monkeypatch):
    """Trains a tiny model and points serve.py's module-level paths at it,
    so the app's lifespan hook loads a real (if tiny) model on startup -
    the same code path production traffic would exercise, just fast."""
    cfg = ExperimentConfig(
        data=DataConfig(seed=9, feature_dim=8, seq_len=20, num_train_sequences=4, num_val_sequences=2),
        model=ModelConfig(fusion_dim=8, hidden_dim=8, stage1_layers=2, refine_layers=1, num_refine_stages=1),
        train=TrainConfig(epochs=1, batch_size=2, num_threads=1),
        eval=EvalConfig(),
    )
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(
        yaml.dump(
            {
                "data": {"seed": 9, "feature_dim": 8, "seq_len": 20, "num_train_sequences": 4, "num_val_sequences": 2},
                "model": {"fusion_dim": 8, "hidden_dim": 8, "stage1_layers": 2, "refine_layers": 1, "num_refine_stages": 1},
                "train": {"epochs": 1, "batch_size": 2, "num_threads": 1},
            }
        )
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    train(cfg, checkpoint_path)

    import src.serve as serve_module

    monkeypatch.setattr(serve_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(serve_module, "CHECKPOINT_PATH", checkpoint_path)
    return cfg


@pytest.fixture()
def missing_checkpoint(tmp_path, monkeypatch):
    """Points serve.py's module-level paths at a config/checkpoint pair
    that genuinely doesn't exist - the repo's own outputs/checkpoint.pt
    (real, from earlier training runs) would otherwise mask this scenario
    if the module-level defaults were left untouched."""
    import src.serve as serve_module

    monkeypatch.setattr(serve_module, "CONFIG_PATH", tmp_path / "no_such_config.yaml")
    monkeypatch.setattr(serve_module, "CHECKPOINT_PATH", tmp_path / "no_such_checkpoint.pt")


def test_health_and_root_work_without_a_model(missing_checkpoint):
    """/health and / must respond even if the model failed to load - an
    orchestrator's liveness probe shouldn't itself depend on the model."""
    import src.serve as serve_module

    with TestClient(serve_module.app) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] is False
        assert health["status"] == "degraded"

        assert client.get("/").status_code == 200


def test_predict_returns_503_when_model_not_loaded(missing_checkpoint):
    import src.serve as serve_module

    with TestClient(serve_module.app) as client:
        resp = client.post("/predict/synthetic", params={"seed": 1})
        assert resp.status_code == 503
        assert "not loaded" in resp.json()["detail"].lower()


def test_health_reports_loaded_after_successful_startup(trained_checkpoint):
    import src.serve as serve_module

    with TestClient(serve_module.app) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] is True
        assert health["status"] == "ok"


def test_predict_synthetic_returns_valid_segmentation(trained_checkpoint):
    import src.serve as serve_module

    with TestClient(serve_module.app) as client:
        resp = client.post("/predict/synthetic", params={"seed": 7})
        assert resp.status_code == 200
        body = resp.json()
        assert body["num_frames"] == 20
        assert len(body["raw_labels"]) == 20
        assert body["model_param_count"] > 0
        assert len(body["segments"]) >= 1
        # segments must tile the sequence with no gaps
        assert body["segments"][0]["start_frame"] == 0
        assert body["segments"][-1]["end_frame"] == 20


def test_predict_synthetic_is_deterministic_given_same_seed(trained_checkpoint):
    import src.serve as serve_module

    with TestClient(serve_module.app) as client:
        r1 = client.post("/predict/synthetic", params={"seed": 3}).json()
        r2 = client.post("/predict/synthetic", params={"seed": 3}).json()
        assert r1["raw_labels"] == r2["raw_labels"]


def test_predict_rejects_wrong_feature_dim(trained_checkpoint):
    import src.serve as serve_module

    with TestClient(serve_module.app) as client:
        resp = client.post(
            "/predict",
            json={"features": [[[1.0, 2.0]]], "camera_mask": [[1.0]]},  # feature_dim=2, expected 8
        )
        assert resp.status_code == 422
        assert "feature_dim" in resp.json()["detail"] or "shape" in resp.json()["detail"]


def test_predict_rejects_mismatched_mask_shape(trained_checkpoint):
    import src.serve as serve_module

    with TestClient(serve_module.app) as client:
        features = [[[0.0] * 8 for _ in range(20)]]  # [1 camera, 20 frames, 8 dims]
        resp = client.post("/predict", json={"features": features, "camera_mask": [[1.0, 1.0]]})  # wrong frame count
        assert resp.status_code == 422


def test_predict_accepts_well_formed_raw_payload(trained_checkpoint):
    import src.serve as serve_module

    with TestClient(serve_module.app) as client:
        features = [[[0.1] * 8 for _ in range(20)]]
        mask = [[1.0] * 20]
        resp = client.post("/predict", json={"features": features, "camera_mask": mask})
        assert resp.status_code == 200
        assert resp.json()["num_frames"] == 20
