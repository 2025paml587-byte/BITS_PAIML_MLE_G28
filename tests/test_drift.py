from src.monitoring.drift import generate_demo_inputs, run_drift_demo
from src.monitoring.logging_store import get_batch_summary


def test_generate_demo_inputs_has_required_columns():
    df = generate_demo_inputs(n_rows=10, seed=0)
    expected = {
        "vendor_id", "passenger_count", "pickup_datetime",
        "pickup_latitude", "pickup_longitude",
        "dropoff_latitude", "dropoff_longitude",
        "store_and_fwd_flag",
    }
    assert expected.issubset(df.columns)
    assert "trip_duration" not in df.columns
    assert len(df) == 10


def test_bias_to_rush_hour_only_uses_rush_hour_pickups():
    df = generate_demo_inputs(n_rows=20, seed=0, bias_to_rush_hour=True)
    hours = set(df["pickup_datetime"].dt.hour)
    assert hours.issubset({7, 8, 9, 17, 18, 19})


def test_run_drift_demo_flags_drift_for_a_large_surge(tmp_path):
    db_path = tmp_path / "predictions.db"

    result = run_drift_demo(db_path=db_path)

    assert result["baseline"]["count"] == 30
    assert result["festival_surge"]["count"] == 30
    assert result["drift_detected"] is True
    assert result["festival_surge"]["mae"] > result["baseline"]["mae"]

    summary = get_batch_summary(db_path=db_path)
    assert set(summary) == {"baseline", "festival_surge"}
