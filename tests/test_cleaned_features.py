import numpy as np
import pandas as pd

from src.features.cleaned_features import (
    extract_interaction_features,
    extract_location_features,
    extract_time_features,
    extract_trip_features,
    find_high_traffic_zones,
    integrate_external_features,
    prepare_model_features,
    process_chunk,
)


def make_eda_processed_df(n_rows: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pickup = pd.date_range("2016-01-01", periods=n_rows, freq="h")
    dropoff = pickup + pd.to_timedelta(rng.integers(5, 60, n_rows), unit="m")
    return pd.DataFrame(
        {
            "id": [f"id{i}" for i in range(n_rows)],
            "vendor_id": rng.choice([1, 2], n_rows),
            "pickup_datetime": pickup,
            "dropoff_datetime": dropoff,
            "passenger_count": rng.integers(0, 7, n_rows),
            "pickup_longitude": rng.uniform(-74.0, -73.8, n_rows),
            "pickup_latitude": rng.uniform(40.6, 40.9, n_rows),
            "dropoff_longitude": rng.uniform(-74.0, -73.8, n_rows),
            "dropoff_latitude": rng.uniform(40.6, 40.9, n_rows),
            "store_and_fwd_flag": rng.choice(["Y", "N"], n_rows),
            "trip_duration": rng.uniform(120, 3600, n_rows),
            "haversine_distance": rng.uniform(0.1, 15, n_rows),
            "pickup_hour": pickup.hour,
            "pickup_day_of_week": pickup.dayofweek,
            "pickup_month": pickup.month,
            "pickup_day": pickup.day,
            "pickup_quarter": pickup.quarter,
        }
    )


def test_extract_time_features_adds_pickup_and_dropoff_calendar_parts():
    df = make_eda_processed_df()
    result = extract_time_features(df.copy())

    for prefix in ("pickup", "dropoff"):
        for suffix in ("season", "day_of_year", "week_of_year", "is_holiday", "part_of_day"):
            assert f"{prefix}_{suffix}" in result.columns


def test_extract_time_features_flags_new_years_day_as_holiday():
    df = pd.DataFrame({"pickup_datetime": ["2016-01-01 00:00:00", "2016-01-02 00:00:00"]})
    result = extract_time_features(df)
    assert result.loc[0, "pickup_is_holiday"] == 1
    assert result.loc[1, "pickup_is_holiday"] == 0


def test_extract_location_features_computes_distance_and_zones():
    df = make_eda_processed_df()
    result = extract_location_features(df.copy())

    assert (result["manhattan_distance"] >= 0).all()
    assert result["bearing"].between(0, 360).all()
    assert "pickup_zone" in result.columns
    assert "route_zone" in result.columns


def test_extract_trip_features_normalizes_flag_casing():
    df = pd.DataFrame({"store_and_fwd_flag": [" y ", "n"]})
    result = extract_trip_features(df)
    assert list(result["store_and_fwd_flag"]) == ["Y", "N"]


def test_extract_interaction_features_computes_products():
    df = pd.DataFrame(
        {
            "haversine_distance": [2.0],
            "pickup_hour": [5],
            "pickup_day_of_week": [3],
            "pickup_is_holiday": [1],
        }
    )
    result = extract_interaction_features(df)
    assert result.loc[0, "distance_hour_interaction"] == 10.0
    assert result.loc[0, "day_of_week_holiday_interaction"] == 3


def test_find_high_traffic_zones_returns_a_set():
    df = make_eda_processed_df()
    zones = find_high_traffic_zones(df)
    assert isinstance(zones, set)


def test_process_chunk_drops_all_dropoff_columns_and_has_no_nulls():
    df = make_eda_processed_df()
    zones = find_high_traffic_zones(df)
    result = process_chunk(df.copy(), zones, [])

    assert [c for c in result.columns if c.startswith("dropoff_")] == []
    assert result.isnull().sum().sum() == 0
    assert len(result) == len(df)


def test_process_chunk_encodes_store_and_fwd_flag_as_int():
    df = make_eda_processed_df()
    zones = find_high_traffic_zones(df)
    result = process_chunk(df.copy(), zones, [])
    assert result["store_and_fwd_flag"].dtype == "int8"
    assert set(result["store_and_fwd_flag"].unique()).issubset({0, 1})


def test_integrate_external_features_merges_on_pickup_datetime():
    df = process_chunk(make_eda_processed_df(), set(), [])
    external = pd.DataFrame(
        {
            "pickup_datetime": df["pickup_datetime"],
            "temperature": np.linspace(0, 30, len(df)),
        }
    )
    merged = integrate_external_features(df.copy(), external)
    assert "temperature" in merged.columns


def test_integrate_external_features_returns_input_unchanged_when_no_external_data():
    df = pd.DataFrame({"a": [1, 2]})
    result = integrate_external_features(df, None)
    assert result.equals(df)


def test_prepare_model_features_clips_out_of_bounds_coordinates():
    df = pd.DataFrame({"pickup_latitude": [10.0, 40.7], "pickup_longitude": [-73.9, -73.9]})
    result = prepare_model_features(df)
    assert result.loc[0, "pickup_latitude"] == 40.5
    assert result.loc[1, "pickup_latitude"] == 40.7
