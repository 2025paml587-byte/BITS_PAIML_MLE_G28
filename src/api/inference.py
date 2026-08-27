"""Load the pre-trained joblib models and run predictions against them.

Models are loaded lazily and cached in-process by name, so the (large)
joblib files are only read from disk once.
"""

import joblib
import pandas as pd

from src.api.schemas import TripRequest
from src.config import SERVING_MODELS
from src.features.build_features import engineer_features

_model_cache: dict[str, object] = {}


def load_model(name: str):
    if name not in SERVING_MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {list(SERVING_MODELS)}")
    if name not in _model_cache:
        path = SERVING_MODELS[name]
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found for '{name}': {path}. "
                "Train it first, or check configs/config.yaml."
            )
        _model_cache[name] = joblib.load(path)
    return _model_cache[name]


def clear_model_cache() -> None:
    _model_cache.clear()


def build_feature_row(request: TripRequest) -> pd.DataFrame:
    """Turn a single trip request into the one-row feature DataFrame the
    trained pipelines expect."""
    raw = pd.DataFrame(
        [
            {
                "vendor_id": request.vendor_id,
                "passenger_count": request.passenger_count,
                "pickup_datetime": request.pickup_datetime,
                "pickup_longitude": request.pickup_longitude,
                "pickup_latitude": request.pickup_latitude,
                "dropoff_longitude": request.dropoff_longitude,
                "dropoff_latitude": request.dropoff_latitude,
                "store_and_fwd_flag": request.store_and_fwd_flag,
            }
        ]
    )
    return engineer_features(raw)


def predict_trip_duration(model, request: TripRequest) -> float:
    features = build_feature_row(request)
    ordered = features[list(model.feature_names_in_)]
    prediction = model.predict(ordered)[0]
    return float(prediction)
