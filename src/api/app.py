"""FastAPI service that serves the pre-trained trip-duration models,
plus a small static UI for trying it out.

Run locally with:
    uvicorn src.api.app:app --reload
Then open http://127.0.0.1:8000/ for the UI, or /docs for the API.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src.api.inference import load_model, predict_trip_duration
from src.api.schemas import PredictionResponse, TripRequest
from src.config import DEFAULT_SERVING_MODEL, SERVING_MODELS
from src.monitoring.drift import run_drift_demo
from src.monitoring.logging_store import clear_all, get_batch_summary, get_recent_predictions, log_prediction

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Trip Duration Prediction API",
    description="Predicts NYC taxi trip duration using pre-trained models.",
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui() -> str:
    return (STATIC_DIR / "index.html").read_text()


@app.get("/monitoring", response_class=HTMLResponse, include_in_schema=False)
def monitoring_ui() -> str:
    return (STATIC_DIR / "monitoring.html").read_text()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict:
    return {
        "available_models": list(SERVING_MODELS),
        "default_model": DEFAULT_SERVING_MODEL,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TripRequest) -> PredictionResponse:
    model_name = request.algorithm or DEFAULT_SERVING_MODEL

    try:
        model = load_model(model_name)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    predicted_seconds = predict_trip_duration(model, request)

    log_prediction(
        batch_label="live",
        model_used=model_name,
        predicted_seconds=predicted_seconds,
        vendor_id=request.vendor_id,
        passenger_count=request.passenger_count,
        pickup_datetime=request.pickup_datetime,
        pickup_latitude=request.pickup_latitude,
        pickup_longitude=request.pickup_longitude,
        dropoff_latitude=request.dropoff_latitude,
        dropoff_longitude=request.dropoff_longitude,
        store_and_fwd_flag=request.store_and_fwd_flag,
    )

    return PredictionResponse(
        predicted_trip_duration_seconds=predicted_seconds,
        model_used=model_name,
    )


@app.get("/monitoring/summary")
def monitoring_summary() -> dict:
    return get_batch_summary()


@app.get("/monitoring/recent")
def monitoring_recent(limit: int = 20) -> list[dict]:
    return get_recent_predictions(limit=limit)


@app.post("/monitoring/simulate-drift")
def monitoring_simulate_drift(model: str = DEFAULT_SERVING_MODEL) -> dict:
    """Replay a normal baseline batch and a simulated festival/rush-hour
    surge batch through `model`, logging predicted vs. actual for each,
    and report whether the surge batch's error indicates drift."""
    try:
        return run_drift_demo(model_name=model)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/monitoring/reset")
def monitoring_reset() -> dict:
    clear_all()
    return {"status": "cleared"}
