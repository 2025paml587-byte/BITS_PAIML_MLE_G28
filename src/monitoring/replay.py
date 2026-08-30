"""Replay trips with a known actual duration through the deployed
model, logging predicted vs. actual.

This is the credible way to get real actual-vs-predicted pairs for
monitoring in a live demo: a prediction made right now has no future
ground truth to compare against yet, but a historical (or simulated)
trip whose real duration is already known does.
"""

from pathlib import Path

import pandas as pd

from src.api.inference import load_model, predict_trip_duration
from src.api.schemas import TripRequest
from src.monitoring.logging_store import log_prediction

REQUIRED_COLUMNS = [
    "vendor_id",
    "passenger_count",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag",
    "trip_duration",
]


def replay_batch(
    df: pd.DataFrame,
    batch_label: str,
    model_name: str = "gradient_boosting",
    db_path: Path | None = None,
) -> list[dict]:
    """Replay every row of df (must have REQUIRED_COLUMNS) through
    model_name, logging predicted vs. actual duration for each, and
    return the logged rows."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for replay: {missing}")

    model = load_model(model_name)
    logged = []
    for _, row in df.iterrows():
        request = TripRequest(
            vendor_id=int(row["vendor_id"]),
            passenger_count=int(row["passenger_count"]),
            pickup_datetime=row["pickup_datetime"],
            pickup_longitude=float(row["pickup_longitude"]),
            pickup_latitude=float(row["pickup_latitude"]),
            dropoff_longitude=float(row["dropoff_longitude"]),
            dropoff_latitude=float(row["dropoff_latitude"]),
            store_and_fwd_flag=row["store_and_fwd_flag"],
        )
        predicted_seconds = predict_trip_duration(model, request)
        logged.append(
            log_prediction(
                batch_label=batch_label,
                model_used=model_name,
                predicted_seconds=predicted_seconds,
                actual_seconds=float(row["trip_duration"]),
                vendor_id=request.vendor_id,
                passenger_count=request.passenger_count,
                pickup_datetime=request.pickup_datetime,
                pickup_latitude=request.pickup_latitude,
                pickup_longitude=request.pickup_longitude,
                dropoff_latitude=request.dropoff_latitude,
                dropoff_longitude=request.dropoff_longitude,
                store_and_fwd_flag=request.store_and_fwd_flag,
                db_path=db_path,
            )
        )
    return logged
