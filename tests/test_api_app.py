from fastapi.testclient import TestClient

from src.api.app import app
from src.api.inference import clear_model_cache
from src.config import SERVING_MODELS

client = TestClient(app)

SAMPLE_TRIP = {
    "vendor_id": 1,
    "passenger_count": 1,
    "pickup_datetime": "2016-03-14T09:30:00",
    "pickup_longitude": -73.9855,
    "pickup_latitude": 40.7580,
    "dropoff_longitude": -73.9654,
    "dropoff_latitude": 40.7829,
    "store_and_fwd_flag": "N",
}


def setup_function():
    clear_model_cache()


def test_ui_page_is_served_at_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<form" in response.text
    assert "/predict" in response.text


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models():
    response = client.get("/models")
    body = response.json()
    assert "gradient_boosting" in body["available_models"]
    assert "linear_regression" in body["available_models"]
    assert body["default_model"] == "gradient_boosting"


def test_predict_with_default_model_uses_gradient_boosting():
    response = client.post("/predict", json=SAMPLE_TRIP)
    assert response.status_code == 200
    body = response.json()
    assert body["model_used"] == "gradient_boosting"
    assert body["predicted_trip_duration_seconds"] > 0


def test_predict_with_explicit_linear_regression():
    response = client.post("/predict", json={**SAMPLE_TRIP, "algorithm": "linear_regression"})
    assert response.status_code == 200
    assert response.json()["model_used"] == "linear_regression"


def test_predict_rejects_unknown_algorithm():
    response = client.post("/predict", json={**SAMPLE_TRIP, "algorithm": "not-a-model"})
    assert response.status_code == 422


def test_predict_rejects_invalid_store_and_fwd_flag():
    response = client.post("/predict", json={**SAMPLE_TRIP, "store_and_fwd_flag": "X"})
    assert response.status_code == 422


def test_predict_returns_503_when_model_file_missing(tmp_path):
    missing_path = tmp_path / "missing.joblib"
    original_path = SERVING_MODELS["gradient_boosting"]
    SERVING_MODELS["gradient_boosting"] = missing_path
    try:
        response = client.post("/predict", json=SAMPLE_TRIP)
        assert response.status_code == 503
    finally:
        SERVING_MODELS["gradient_boosting"] = original_path
