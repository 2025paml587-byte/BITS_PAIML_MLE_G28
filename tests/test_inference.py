from src.api.inference import build_feature_row, clear_high_traffic_zones_cache
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


def setup_function():
    clear_high_traffic_zones_cache()


def test_build_feature_row_produces_expected_columns():
    request = TripRequest(**SAMPLE_TRIP)
    row = build_feature_row(request)

    expected_columns = {
        "vendor_id",
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "store_and_fwd_flag",
        "haversine_distance",
        "pickup_hour",
        "pickup_day_of_week",
        "pickup_month",
        "pickup_day",
        "pickup_quarter",
        "pickup_season",
        "pickup_day_of_year",
        "pickup_week_of_year",
        "pickup_is_holiday",
        "pickup_part_of_day",
        "manhattan_distance",
        "bearing",
        "pickup_zone",
        "route_zone",
        "same_zone",
        "pickup_high_traffic",
        "high_traffic_route",
        "distance_hour_interaction",
        "day_of_week_holiday_interaction",
    }
    assert expected_columns.issubset(set(row.columns))
    assert len(row) == 1


def test_build_feature_row_drops_dropoff_coordinates():
    # The cleaned schema drops raw dropoff_* columns in favor of the
    # derived route/zone features - the served models don't expect them.
    request = TripRequest(**SAMPLE_TRIP)
    row = build_feature_row(request)
    assert [c for c in row.columns if c.startswith("dropoff_")] == []


def test_build_feature_row_computes_positive_distance():
    request = TripRequest(**SAMPLE_TRIP)
    row = build_feature_row(request)
    assert row.loc[0, "haversine_distance"] > 0
    assert row.loc[0, "manhattan_distance"] > 0


def test_build_feature_row_extracts_temporal_parts():
    request = TripRequest(**SAMPLE_TRIP)
    row = build_feature_row(request)
    assert row.loc[0, "pickup_hour"] == 9
    assert row.loc[0, "pickup_month"] == 3
    assert row.loc[0, "pickup_season"] == "spring"
