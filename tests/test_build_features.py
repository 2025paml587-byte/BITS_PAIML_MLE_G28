import pandas as pd

from src.features.build_features import (
    add_distance_feature,
    add_temporal_features,
    engineer_features,
    haversine_distance_km,
)


def test_haversine_distance_zero_for_identical_points():
    assert haversine_distance_km(40.75, -73.98, 40.75, -73.98) == 0.0


def test_haversine_distance_known_value():
    # Times Square to Central Park South, roughly 3.2 km apart.
    distance = haversine_distance_km(40.7580, -73.9855, 40.7829, -73.9654)
    assert 3.0 < distance < 3.5


def test_add_distance_feature_adds_column_without_mutating_input():
    df = pd.DataFrame(
        {
            "pickup_latitude": [40.75, 40.76],
            "pickup_longitude": [-73.98, -73.97],
            "dropoff_latitude": [40.75, 40.80],
            "dropoff_longitude": [-73.98, -73.90],
        }
    )
    result = add_distance_feature(df)

    assert "haversine_distance" in result.columns
    assert result.loc[0, "haversine_distance"] == 0.0
    assert result.loc[1, "haversine_distance"] > 0
    assert "haversine_distance" not in df.columns


def test_add_temporal_features_extracts_calendar_parts():
    df = pd.DataFrame({"pickup_datetime": ["2016-03-14 09:30:00"]})
    result = add_temporal_features(df)

    assert result.loc[0, "pickup_hour"] == 9
    assert result.loc[0, "pickup_month"] == 3
    assert result.loc[0, "pickup_day"] == 14
    assert result.loc[0, "pickup_quarter"] == 1
    assert result.loc[0, "pickup_day_of_week"] == 0  # Monday


def test_engineer_features_adds_all_expected_columns():
    df = pd.DataFrame(
        {
            "pickup_datetime": ["2016-01-01 00:00:00"],
            "pickup_latitude": [40.75],
            "pickup_longitude": [-73.98],
            "dropoff_latitude": [40.76],
            "dropoff_longitude": [-73.99],
        }
    )
    result = engineer_features(df)

    expected_columns = {
        "haversine_distance",
        "pickup_hour",
        "pickup_day_of_week",
        "pickup_month",
        "pickup_day",
        "pickup_quarter",
    }
    assert expected_columns.issubset(result.columns)
