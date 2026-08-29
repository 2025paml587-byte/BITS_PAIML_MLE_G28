"""Load the pre-trained joblib models and run predictions against them.

Models are loaded lazily and cached in-process by name, so the (large)
joblib files are only read from disk once. Feature building matches
the cleaned/feature-engineered schema (src.features.cleaned_features)
that the currently-served models were trained on - not the older,
simpler schema in src.features.build_features.
"""

import joblib
import pandas as pd

from src.api.schemas import TripRequest
from src.config import SERVING_MODELS
from src.data.zone_reference import load_high_traffic_zones
from src.features.build_features import engineer_features
from src.features.cleaned_features import process_chunk

_model_cache: dict[str, object] = {}
_high_traffic_zones_cache: set | None = None
_warned_missing_zones = False


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


def get_high_traffic_zones() -> set:
    """Load the persisted high-traffic-zones set once per process.

    Falls back to an empty set (nothing flagged as high-traffic) if it
    hasn't been generated yet by src.features.cleaning_pipeline -
    predictions still work, just without that signal being accurate to
    what the model was actually trained on.
    """
    global _high_traffic_zones_cache, _warned_missing_zones
    if _high_traffic_zones_cache is None:
        _high_traffic_zones_cache = load_high_traffic_zones()
        if not _high_traffic_zones_cache and not _warned_missing_zones:
            print(
                "Warning: no high_traffic_zones.json found - serving without "
                "accurate high-traffic-route features. Rerun "
                "src.features.cleaning_pipeline.run_feature_cleaning() to "
                "generate it."
            )
            _warned_missing_zones = True
    return _high_traffic_zones_cache


def clear_high_traffic_zones_cache() -> None:
    global _high_traffic_zones_cache, _warned_missing_zones
    _high_traffic_zones_cache = None
    _warned_missing_zones = False


def build_feature_row(request: TripRequest) -> pd.DataFrame:
    """Turn a single trip request into the one-row feature DataFrame the
    trained pipelines expect.

    train_cleaned.zip (what the served models were trained on) is built
    by running the cleaning stage (src.features.cleaned_features) on top
    of the EDA stage's output (haversine_distance + basic pickup-time
    parts), not on raw request fields directly - so both stages have to
    run here, in the same order, for a live request.
    """
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
    eda_features = engineer_features(raw)
    return process_chunk(eda_features, get_high_traffic_zones(), [])


def predict_trip_duration(model, request: TripRequest) -> float:
    features = build_feature_row(request)
    ordered = features[list(model.feature_names_in_)]
    prediction = model.predict(ordered)[0]
    return float(prediction)
