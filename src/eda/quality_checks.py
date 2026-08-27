"""Data-quality checks for the raw trip-duration data.

Pure functions of a DataFrame (no plotting, no file I/O), so they're
unit testable in isolation. src/eda/report.py wires them into the
full EDA report.
"""

import pandas as pd

NYC_LAT_MIN, NYC_LAT_MAX = 40.5, 41.0
NYC_LON_MIN, NYC_LON_MAX = -74.25, -73.7
VALID_STORE_AND_FWD_FLAGS = {"Y", "N"}


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column, restricted
    to columns that actually have missing values."""
    counts = df.isnull().sum()
    percentages = 100 * counts / len(df)
    report = pd.DataFrame({"missing_count": counts, "missing_percentage": percentages})
    return report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)


def count_duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def count_negative_durations(df: pd.DataFrame) -> int:
    return int((df["trip_duration"] < 0).sum())


def count_invalid_time_sequence(df: pd.DataFrame) -> int:
    """Trips where dropoff_datetime is before pickup_datetime."""
    if "dropoff_datetime" not in df.columns:
        return 0
    return int((df["dropoff_datetime"] < df["pickup_datetime"]).sum())


def count_invalid_coordinates(df: pd.DataFrame) -> dict:
    """Count pickup/dropoff lat/lon values outside the NYC bounding box."""
    return {
        "invalid_pickup_lat": int(
            ((df["pickup_latitude"] < NYC_LAT_MIN) | (df["pickup_latitude"] > NYC_LAT_MAX)).sum()
        ),
        "invalid_pickup_lon": int(
            ((df["pickup_longitude"] < NYC_LON_MIN) | (df["pickup_longitude"] > NYC_LON_MAX)).sum()
        ),
        "invalid_dropoff_lat": int(
            ((df["dropoff_latitude"] < NYC_LAT_MIN) | (df["dropoff_latitude"] > NYC_LAT_MAX)).sum()
        ),
        "invalid_dropoff_lon": int(
            ((df["dropoff_longitude"] < NYC_LON_MIN) | (df["dropoff_longitude"] > NYC_LON_MAX)).sum()
        ),
    }


def unexpected_store_and_fwd_flag_values(df: pd.DataFrame) -> list:
    return [v for v in df["store_and_fwd_flag"].unique() if v not in VALID_STORE_AND_FWD_FLAGS]


def run_quality_checks(train_df: pd.DataFrame) -> dict:
    """Run every check against the training data and return one summary dict."""
    report = {
        "duplicate_rows": count_duplicate_rows(train_df),
        "negative_duration_trips": count_negative_durations(train_df),
        "invalid_time_sequence": count_invalid_time_sequence(train_df),
        "unexpected_store_and_fwd_flag_values": unexpected_store_and_fwd_flag_values(train_df),
    }
    report.update(count_invalid_coordinates(train_df))
    return report
