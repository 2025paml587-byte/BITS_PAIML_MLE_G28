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

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Trip Duration Prediction API",
    description="Predicts NYC taxi trip duration using pre-trained models.",
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui() -> str:
    return (STATIC_DIR / "index.html").read_text()


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
    return PredictionResponse(
        predicted_trip_duration_seconds=predicted_seconds,
        model_used=model_name,
    )
