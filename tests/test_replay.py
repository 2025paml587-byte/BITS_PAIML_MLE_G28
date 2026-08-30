import pandas as pd
import pytest

from src.monitoring.logging_store import get_recent_predictions
from src.monitoring.replay import replay_batch


def make_historical_df(n_rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vendor_id": [1, 2, 1][:n_rows],
            "passenger_count": [1, 2, 1][:n_rows],
            "pickup_datetime": pd.date_range("2016-03-14 09:00", periods=n_rows, freq="h"),
            "pickup_longitude": [-73.9855, -73.98, -73.97][:n_rows],
            "pickup_latitude": [40.7580, 40.75, 40.76][:n_rows],
            "dropoff_longitude": [-73.9654, -73.96, -73.95][:n_rows],
            "dropoff_latitude": [40.7829, 40.78, 40.79][:n_rows],
            "store_and_fwd_flag": ["N", "Y", "N"][:n_rows],
            "trip_duration": [700.0, 900.0, 500.0][:n_rows],
        }
    )


def test_replay_batch_raises_on_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        replay_batch(pd.DataFrame({"vendor_id": [1]}), batch_label="baseline")


def test_replay_batch_logs_one_row_per_input_row(tmp_path):
    db_path = tmp_path / "predictions.db"
    df = make_historical_df(3)

    logged = replay_batch(df, batch_label="baseline", db_path=db_path)

    assert len(logged) == 3
    recent = get_recent_predictions(limit=10, db_path=db_path)
    assert len(recent) == 3
    assert all(row["batch_label"] == "baseline" for row in recent)


def test_replay_batch_records_actual_and_predicted(tmp_path):
    db_path = tmp_path / "predictions.db"
    df = make_historical_df(1)

    logged = replay_batch(df, batch_label="baseline", db_path=db_path)

    assert logged[0]["actual_seconds"] == 700.0
    assert logged[0]["predicted_seconds"] > 0
    assert logged[0]["absolute_error"] == abs(logged[0]["predicted_seconds"] - 700.0)
