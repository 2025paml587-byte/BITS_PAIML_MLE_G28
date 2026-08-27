"""FastAPI service that serves the pre-trained trip-duration models.

Run locally with:
    uvicorn src.api.app:app --reload
"""

from fastapi import FastAPI, HTTPException

from src.api.inference import load_model, predict_trip_duration
from src.api.schemas import PredictionResponse, TripRequest
from src.config import DEFAULT_SERVING_MODEL, SERVING_MODELS

app = FastAPI(
    title="Trip Duration Prediction API",
    description="Predicts NYC taxi trip duration using pre-trained models.",
)


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
