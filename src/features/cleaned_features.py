"""Feature engineering/cleaning transforms for the "cleaned" dataset
stage, ported from notebooks/featureengineering.py.

Pure DataFrame-transform functions (no file or DVC I/O), so they're
unit testable in isolation. src/features/cleaning_pipeline.py wires
them into the full chunked read/process/write pipeline.

Unlike src/features/build_features.py (which produces the schema the
currently-served models expect, including raw dropoff coordinates),
this module drops all dropoff_* columns from the final feature set in
favor of derived route/zone features - a different, richer feature
set intended for a future retraining, not compatible with the models
currently served by src/api.
"""

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def extract_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features for each datetime column available in a frame."""
    holiday_calendar = USFederalHolidayCalendar()

    for datetime_column in ("pickup_datetime", "dropoff_datetime"):
        if datetime_column not in dataframe.columns:
            continue

        datetime_values = pd.to_datetime(dataframe[datetime_column], errors="coerce")
        feature_prefix = datetime_column.removesuffix("_datetime")
        hour_values = datetime_values.dt.hour

        dataframe[datetime_column] = datetime_values
        dataframe[f"{feature_prefix}_hour"] = hour_values
        dataframe[f"{feature_prefix}_day_of_week"] = datetime_values.dt.dayofweek
        dataframe[f"{feature_prefix}_month"] = datetime_values.dt.month
        dataframe[f"{feature_prefix}_season"] = datetime_values.dt.month.map(SEASON_BY_MONTH)
        dataframe[f"{feature_prefix}_day_of_year"] = datetime_values.dt.dayofyear
        dataframe[f"{feature_prefix}_week_of_year"] = (
            datetime_values.dt.isocalendar().week.astype("Int64")
        )

        valid_datetime_values = datetime_values.dropna()
        if valid_datetime_values.empty:
            holiday_flags = pd.Series(pd.NA, index=dataframe.index, dtype="Int64")
        else:
            holiday_dates = holiday_calendar.holidays(
                start=valid_datetime_values.min().normalize(),
                end=valid_datetime_values.max().normalize(),
            ).date
            holiday_flags = datetime_values.dt.date.isin(holiday_dates).astype("Int64")
        dataframe[f"{feature_prefix}_is_holiday"] = holiday_flags

        dataframe[f"{feature_prefix}_part_of_day"] = pd.cut(
            hour_values,
            bins=[-1, 5, 11, 17, 23],
            labels=["night", "morning", "afternoon", "evening"],
        )

    return dataframe


def extract_location_features(
    dataframe: pd.DataFrame, high_traffic_zones: set | None = None
) -> pd.DataFrame:
    """Add distance, direction, grid-zone, and location interaction features."""
    required_columns = [
        "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
    ]
    if not all(column in dataframe.columns for column in required_columns):
        return dataframe

    pickup_latitude = pd.to_numeric(dataframe["pickup_latitude"], errors="coerce")
    pickup_longitude = pd.to_numeric(dataframe["pickup_longitude"], errors="coerce")
    dropoff_latitude = pd.to_numeric(dataframe["dropoff_latitude"], errors="coerce")
    dropoff_longitude = pd.to_numeric(dataframe["dropoff_longitude"], errors="coerce")

    dataframe["manhattan_distance"] = (
        (dropoff_latitude - pickup_latitude).abs()
        + (dropoff_longitude - pickup_longitude).abs()
    )

    latitude_difference = np.radians(dropoff_latitude - pickup_latitude)
    longitude_difference = np.radians(dropoff_longitude - pickup_longitude)
    pickup_latitude_radians = np.radians(pickup_latitude)
    dropoff_latitude_radians = np.radians(dropoff_latitude)
    bearing = np.degrees(
        np.arctan2(
            np.sin(longitude_difference) * np.cos(dropoff_latitude_radians),
            np.cos(pickup_latitude_radians) * np.sin(dropoff_latitude_radians)
            - np.sin(pickup_latitude_radians)
            * np.cos(dropoff_latitude_radians)
            * np.cos(longitude_difference),
        )
    )
    dataframe["bearing"] = (bearing + 360) % 360

    # A fixed 0.01-degree grid keeps zone IDs consistent between train and test.
    pickup_zone = (
        pickup_latitude.round(2).astype("string") + "_" + pickup_longitude.round(2).astype("string")
    )
    dropoff_zone = (
        dropoff_latitude.round(2).astype("string") + "_" + dropoff_longitude.round(2).astype("string")
    )
    dataframe["pickup_zone"] = pickup_zone
    dataframe["dropoff_zone"] = dropoff_zone
    dataframe["route_zone"] = pickup_zone + "_to_" + dropoff_zone
    dataframe["same_zone"] = (pickup_zone == dropoff_zone).astype("Int64")

    if high_traffic_zones is None:
        high_traffic_zones = set()
    dataframe["pickup_high_traffic"] = pickup_zone.isin(high_traffic_zones).astype("Int64")
    dataframe["dropoff_high_traffic"] = dropoff_zone.isin(high_traffic_zones).astype("Int64")
    dataframe["high_traffic_route"] = (
        (dataframe["pickup_high_traffic"] == 1) | (dataframe["dropoff_high_traffic"] == 1)
    ).astype("Int64")

    return dataframe


def extract_trip_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Prepare trip attributes without deriving features from the target."""
    if "passenger_count" in dataframe.columns:
        dataframe["passenger_count"] = pd.to_numeric(dataframe["passenger_count"], errors="coerce")
    if "store_and_fwd_flag" in dataframe.columns:
        dataframe["store_and_fwd_flag"] = (
            dataframe["store_and_fwd_flag"].astype("string").str.strip().str.upper()
        )
    return dataframe


def extract_interaction_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Combine non-target distance and pickup-time features."""
    if "haversine_distance" in dataframe.columns and "pickup_hour" in dataframe.columns:
        dataframe["distance_hour_interaction"] = (
            pd.to_numeric(dataframe["haversine_distance"], errors="coerce")
            * pd.to_numeric(dataframe["pickup_hour"], errors="coerce")
        )
    if "pickup_day_of_week" in dataframe.columns and "pickup_is_holiday" in dataframe.columns:
        dataframe["day_of_week_holiday_interaction"] = (
            pd.to_numeric(dataframe["pickup_day_of_week"], errors="coerce")
            * pd.to_numeric(dataframe["pickup_is_holiday"], errors="coerce")
        )
    return dataframe


def prepare_model_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean predictors while preserving trip_duration only as the train target."""
    dropoff_columns = [
        column for column in dataframe.columns
        if column.startswith("dropoff_") or column == "dropoff_datetime"
    ]
    dataframe = dataframe.drop(columns=dropoff_columns, errors="ignore")

    # These bounds remove impossible predictor values without modifying the target.
    if "passenger_count" in dataframe.columns:
        dataframe["passenger_count"] = dataframe["passenger_count"].clip(lower=1, upper=6)
    for column in ("pickup_latitude", "dropoff_latitude"):
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].clip(lower=40.5, upper=41.0)
    for column in ("pickup_longitude", "dropoff_longitude"):
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].clip(lower=-74.25, upper=-73.7)
    for column in ("haversine_distance", "manhattan_distance", "bearing"):
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    if "haversine_distance" in dataframe.columns:
        dataframe["haversine_distance"] = dataframe["haversine_distance"].clip(lower=0, upper=100)
    if "manhattan_distance" in dataframe.columns:
        dataframe["manhattan_distance"] = dataframe["manhattan_distance"].clip(lower=0)
    if "bearing" in dataframe.columns:
        dataframe["bearing"] = dataframe["bearing"].clip(lower=0, upper=360)
    if "distance_hour_interaction" in dataframe.columns:
        dataframe["distance_hour_interaction"] = dataframe["distance_hour_interaction"].clip(
            lower=0, upper=2400
        )

    if "store_and_fwd_flag" in dataframe.columns:
        dataframe["store_and_fwd_flag"] = (
            dataframe["store_and_fwd_flag"].astype("string").str.strip().str.upper()
            .map({"N": 0, "Y": 1})
            .fillna(0)
            .astype("int8")
        )
    for column in ("pickup_season", "pickup_part_of_day"):
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype("string").fillna("unknown")

    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if column != "trip_duration":
            dataframe[column] = dataframe[column].fillna(dataframe[column].median())
    for column in dataframe.select_dtypes(include=["object", "string"]).columns:
        if column != "trip_duration":
            dataframe[column] = dataframe[column].fillna("unknown")
    return dataframe


def integrate_external_features(
    dataframe: pd.DataFrame, external_data: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Merge optional weather or traffic data without importing the target."""
    if external_data is None or external_data.empty:
        return dataframe

    external_data = external_data.copy()
    if "trip_duration" in external_data.columns:
        external_data = external_data.drop(columns=["trip_duration"])
    merge_keys = [
        column for column in ("pickup_datetime", "pickup_zone", "dropoff_zone", "route_zone")
        if column in dataframe.columns and column in external_data.columns
    ]
    if not merge_keys:
        return dataframe

    if "pickup_datetime" in merge_keys:
        dataframe = dataframe.copy()
        external_data["pickup_datetime"] = pd.to_datetime(
            external_data["pickup_datetime"], errors="coerce"
        )
        dataframe["pickup_datetime"] = pd.to_datetime(dataframe["pickup_datetime"], errors="coerce")

    allowed_columns = {
        "temperature", "precipitation", "wind_speed", "weather_type",
        "average_speed", "traffic_congestion",
    }
    columns_to_merge = merge_keys + [
        column for column in external_data.columns
        if column in allowed_columns and column not in merge_keys
    ]
    if len(columns_to_merge) == len(merge_keys):
        return dataframe

    external_data = external_data[columns_to_merge].drop_duplicates(merge_keys)
    return dataframe.merge(external_data, on=merge_keys, how="left", sort=False)


def find_high_traffic_zones(dataframe: pd.DataFrame, quantile: float = 0.9) -> set:
    """Find frequently used pickup/dropoff grid zones from training data."""
    location_features = extract_location_features(dataframe.copy())
    zone_counts = pd.concat(
        [location_features["pickup_zone"], location_features["dropoff_zone"]]
    ).value_counts()
    if zone_counts.empty:
        return set()
    return set(zone_counts[zone_counts >= zone_counts.quantile(quantile)].index)


def process_chunk(dataframe: pd.DataFrame, high_traffic_zones: set, external_data: list) -> pd.DataFrame:
    """Apply all feature transformations to one bounded data chunk."""
    dataframe = extract_time_features(dataframe)
    dataframe = extract_trip_features(dataframe)
    dataframe = extract_location_features(dataframe, high_traffic_zones)
    dataframe = extract_interaction_features(dataframe)
    for external_frame in external_data:
        dataframe = integrate_external_features(dataframe, external_frame)
    return prepare_model_features(dataframe)
