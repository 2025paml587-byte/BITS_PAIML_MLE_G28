from src.api.inference import build_feature_row
from src.api.schemas import TripRequest

SAMPLE_TRIP = {
    "vendor_id": 1,
    "passenger_count": 1,
    "pickup_datetime": "2016-03-14T09:30:00",
    "pickup_longitude": -73.9855,
    "pickup_latitude": 40.7580,
    "dropoff_longitude": -73.9654,
    "dropoff_latitude": 40.7829,
    "store_and_fwd_flag": "N",
}


def test_build_feature_row_produces_expected_columns():
    request = TripRequest(**SAMPLE_TRIP)
    row = build_feature_row(request)

    expected_columns = {
        "vendor_id",
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "store_and_fwd_flag",
        "haversine_distance",
        "pickup_hour",
        "pickup_day_of_week",
        "pickup_month",
        "pickup_day",
        "pickup_quarter",
    }
    assert expected_columns.issubset(set(row.columns))
    assert len(row) == 1


def test_build_feature_row_computes_positive_distance():
    request = TripRequest(**SAMPLE_TRIP)
    row = build_feature_row(request)
    assert row.loc[0, "haversine_distance"] > 0


def test_build_feature_row_extracts_temporal_parts():
    request = TripRequest(**SAMPLE_TRIP)
    row = build_feature_row(request)
    assert row.loc[0, "pickup_hour"] == 9
    assert row.loc[0, "pickup_month"] == 3
