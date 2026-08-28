import pandas as pd

from src.eda.quality_checks import (
    count_duplicate_rows,
    count_invalid_coordinates,
    count_invalid_time_sequence,
    count_negative_durations,
    missing_value_report,
    run_quality_checks,
    unexpected_store_and_fwd_flag_values,
)


def make_train_df(**overrides) -> pd.DataFrame:
    base = {
        "trip_duration": [300, 600],
        "pickup_datetime": pd.to_datetime(["2016-01-01 08:00:00", "2016-01-01 09:00:00"]),
        "dropoff_datetime": pd.to_datetime(["2016-01-01 08:05:00", "2016-01-01 09:10:00"]),
        "pickup_latitude": [40.75, 40.76],
        "pickup_longitude": [-73.98, -73.97],
        "dropoff_latitude": [40.75, 40.80],
        "dropoff_longitude": [-73.98, -73.90],
        "store_and_fwd_flag": ["Y", "N"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_missing_value_report_excludes_complete_columns():
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    report = missing_value_report(df)
    assert list(report.index) == ["a"]
    assert report.loc["a", "missing_count"] == 1


def test_count_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2]})
    assert count_duplicate_rows(df) == 1


def test_count_negative_durations():
    df = make_train_df(trip_duration=[300, -5])
    assert count_negative_durations(df) == 1


def test_count_invalid_time_sequence_detects_dropoff_before_pickup():
    df = make_train_df(
        pickup_datetime=pd.to_datetime(["2016-01-01 08:00:00", "2016-01-01 09:00:00"]),
        dropoff_datetime=pd.to_datetime(["2016-01-01 07:00:00", "2016-01-01 09:10:00"]),
    )
    assert count_invalid_time_sequence(df) == 1


def test_count_invalid_time_sequence_without_dropoff_column_returns_zero():
    df = pd.DataFrame({"pickup_datetime": pd.to_datetime(["2016-01-01"])})
    assert count_invalid_time_sequence(df) == 0


def test_count_invalid_coordinates_flags_out_of_bounds_points():
    df = make_train_df(pickup_latitude=[40.75, 10.0])
    result = count_invalid_coordinates(df)
    assert result["invalid_pickup_lat"] == 1
    assert result["invalid_pickup_lon"] == 0


def test_unexpected_store_and_fwd_flag_values():
    df = make_train_df(store_and_fwd_flag=["Y", "X"])
    assert unexpected_store_and_fwd_flag_values(df) == ["X"]


def test_run_quality_checks_returns_all_keys_on_clean_data():
    df = make_train_df()
    report = run_quality_checks(df)
    assert report["duplicate_rows"] == 0
    assert report["negative_duration_trips"] == 0
    assert report["invalid_time_sequence"] == 0
    assert report["unexpected_store_and_fwd_flag_values"] == []
    assert report["invalid_pickup_lat"] == 0
