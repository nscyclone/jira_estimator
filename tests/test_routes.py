import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from unittest.mock import MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

import state
from config import CONFIG
from feedback import init_db, insert_model_run
from routes import router


@pytest.fixture
def client(tmp_path):
    original_db = CONFIG["feedback_db_path"]
    CONFIG["feedback_db_path"] = str(tmp_path / "test.db")
    init_db(CONFIG["feedback_db_path"])
    insert_model_run(
        CONFIG["feedback_db_path"],
        run_id="testrun0001ff",
        trigger="manual",
        model_type="estimate",
        r2=0.184,
        mae=1.19,
        rmse=1.65,
        overrun_recall=None,
        fold_scores=[0.18] * 5,
    )

    mock_vec = MagicMock()
    mock_vec.transform.return_value = csr_matrix(np.zeros((1, 100)))
    mock_svd = MagicMock()
    mock_svd.transform.return_value = np.zeros((1, 32))
    mock_est = MagicMock()
    mock_est.predict.return_value = np.array([0.5])
    mock_risk = MagicMock()
    mock_risk.predict_proba.return_value = np.array([[0.6, 0.3, 0.1]])

    state.ml_models["vectorizer"] = mock_vec
    state.ml_models["svd_transformer"] = mock_svd
    state.ml_models["est_folds"] = [mock_est] * 5
    state.ml_models["risk_folds"] = [mock_risk] * 5

    test_app = FastAPI()
    test_app.include_router(router)
    with TestClient(test_app) as c:
        yield c

    state.ml_models.clear()
    CONFIG["feedback_db_path"] = original_db


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_expected_keys(client):
    resp = client.post("/predict", json={
        "summary": "Реализовать выгрузку СЭМД протокола осмотра",
        "region": "БАЗОВЫЙ",
        "subsystem": "СЭМД/Выгрузка",
        "commitments": "ТЗР",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_time_hours" in data
    assert "adjusted_time_with_buffer_hours" in data
    assert "prediction_std_days" in data
    assert "risk_profile" in data
    assert data["predicted_time_hours"] >= 0.0


def test_predict_empty_summary_rejected(client):
    resp = client.post("/predict", json={"summary": "   "})
    assert resp.status_code == 400


def test_feedback_accepted(client):
    resp = client.post("/feedback", json={
        "summary": "Добавить фильтрацию по МКБ-10",
        "predicted_days": 2.0,
        "actual_days": 3.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["feedback_count"] == 1
    assert "retrain_ready" in data
    assert "delta_pct" in data


def test_metrics_returns_model_info(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_r2"] is not None
    assert "feedbacks_collected" in data


def test_retrain_skips_when_below_threshold(client):
    resp = client.post("/retrain")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "skipped"
    assert "threshold" in data
