"""Simulate a festival/rush-hour surge to demonstrate drift detection.

A surge batch represents real-world congestion the model never saw at
training time. To make that comparison meaningful without real
historical data, "actual" duration for the baseline batch is derived
from the model's own prediction plus small realistic noise (so a
well-behaved model naturally shows low error there), while the surge
batch's actual duration is that same prediction inflated by a large
multiplier (representing congestion the model has no way to know
about) - so its error spikes. This is standard synthetic-drift-
injection, not circular: replay_batch() below computes its own fresh
prediction for each row and compares it against the actual we set
here, so the error genuinely reflects predicted-vs-actual on rows the
model was never told were perturbed.

Uses synthetic (but realistically-shaped) trip data rather than the
real historical dataset, so the demo works even without DVC-pulled
data - swap generate_demo_inputs() for a real held-out sample when
actual historical data is available.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.api.inference import load_model, predict_trip_duration
from src.api.schemas import TripRequest
from src.config import DRIFT_MAE_RATIO_THRESHOLD
from src.monitoring.logging_store import get_batch_summary
from src.monitoring.replay import replay_batch

NYC_LAT_RANGE = (40.65, 40.85)
NYC_LON_RANGE = (-74.02, -73.90)
RUSH_HOURS = [7, 8, 9, 17, 18, 19]


def generate_demo_inputs(
    n_rows: int = 30, seed: int = 0, bias_to_rush_hour: bool = False
) -> pd.DataFrame:
    """Synthetic NYC-taxi-shaped trip request fields (no trip_duration -
    that gets filled in separately once we know what the model predicts)."""
    rng = np.random.default_rng(seed)
    pickup_hour = rng.choice(RUSH_HOURS, n_rows) if bias_to_rush_hour else rng.integers(0, 24, n_rows)
    pickup_datetime = pd.Timestamp("2016-06-04") + pd.to_timedelta(pickup_hour, unit="h")

    return pd.DataFrame(
        {
            "vendor_id": rng.choice([1, 2], n_rows),
            "passenger_count": rng.integers(1, 5, n_rows),
            "pickup_datetime": pickup_datetime,
            "pickup_latitude": rng.uniform(*NYC_LAT_RANGE, n_rows),
            "pickup_longitude": rng.uniform(*NYC_LON_RANGE, n_rows),
            "dropoff_latitude": rng.uniform(*NYC_LAT_RANGE, n_rows),
            "dropoff_longitude": rng.uniform(*NYC_LON_RANGE, n_rows),
            "store_and_fwd_flag": rng.choice(["Y", "N"], n_rows),
        }
    )


def _predict_each_row(model, inputs: pd.DataFrame) -> np.ndarray:
    predictions = []
    for _, row in inputs.iterrows():
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
        predictions.append(predict_trip_duration(model, request))
    return np.array(predictions)


def _simulate_actual_durations(
    predicted_seconds: np.ndarray, surge_multiplier: float, noise_ratio: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_ratio * predicted_seconds)
    return np.clip(predicted_seconds * surge_multiplier + noise, 60, None)


def run_drift_demo(model_name: str = "gradient_boosting", db_path: Path | None = None) -> dict:
    """Replay a normal baseline batch and a simulated festival/rush-hour
    surge batch, then compare their MAE to flag drift."""
    model = load_model(model_name)

    baseline_inputs = generate_demo_inputs(n_rows=30, seed=1)
    baseline_predictions = _predict_each_row(model, baseline_inputs)
    baseline_df = baseline_inputs.copy()
    baseline_df["trip_duration"] = _simulate_actual_durations(
        baseline_predictions, surge_multiplier=1.0, noise_ratio=0.1, seed=101
    )

    surge_inputs = generate_demo_inputs(n_rows=30, seed=2, bias_to_rush_hour=True)
    surge_predictions = _predict_each_row(model, surge_inputs)
    surge_df = surge_inputs.copy()
    surge_df["trip_duration"] = _simulate_actual_durations(
        surge_predictions, surge_multiplier=1.8, noise_ratio=0.1, seed=102
    )

    replay_batch(baseline_df, batch_label="baseline", model_name=model_name, db_path=db_path)
    replay_batch(surge_df, batch_label="festival_surge", model_name=model_name, db_path=db_path)

    summary = get_batch_summary(db_path=db_path)
    baseline = summary.get("baseline")
    surge = summary.get("festival_surge")

    drift_detected = bool(
        baseline
        and surge
        and baseline["mae"]
        and surge["mae"]
        and surge["mae"] > baseline["mae"] * DRIFT_MAE_RATIO_THRESHOLD
    )

    return {
        "baseline": baseline,
        "festival_surge": surge,
        "drift_detected": drift_detected,
        "threshold_ratio": DRIFT_MAE_RATIO_THRESHOLD,
    }
