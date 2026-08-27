"""Feature engineering for the trip-duration model.

Turns raw pickup/dropoff coordinates and the pickup timestamp into
the numeric features the models train on: haversine distance and
calendar features.
"""

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometers between two points.

    Accepts scalars or array-likes (e.g. pandas Series); fully
    vectorized via numpy, unlike a row-wise DataFrame.apply.
    """
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a `haversine_distance` (km) column."""
    df = df.copy()
    df["haversine_distance"] = haversine_distance_km(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    return df


def add_temporal_features(
    df: pd.DataFrame,
    datetime_col: str = "pickup_datetime",
    prefix: str = "pickup",
) -> pd.DataFrame:
    """Return a copy of df with hour/day-of-week/month/day/quarter columns
    derived from `datetime_col`."""
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])
    df[datetime_col] = dt
    df[f"{prefix}_hour"] = dt.dt.hour
    df[f"{prefix}_day_of_week"] = dt.dt.dayofweek  # Monday=0, Sunday=6
    df[f"{prefix}_month"] = dt.dt.month
    df[f"{prefix}_day"] = dt.dt.day
    df[f"{prefix}_quarter"] = dt.dt.quarter
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature set used for EDA and model training."""
    df = add_distance_feature(df)
    df = add_temporal_features(df)
    return df
