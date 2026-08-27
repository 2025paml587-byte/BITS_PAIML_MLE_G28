"""Chart generation for the EDA report.

Uses the non-interactive Agg backend so this runs headless (CI,
servers, containers) - every chart is saved to disk, never shown
interactively. Each function returns the path it wrote, and expects
train_df to already have `haversine_distance` / `pickup_hour` /
`pickup_day_of_week` / `pickup_month` (see src.features.build_features).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_passenger_count(train_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.countplot(data=train_df, x="passenger_count", ax=axes[0])
    axes[0].set_title("Passenger Count Distribution (Train)")
    axes[0].set_xlabel("Passenger Count")
    axes[0].set_ylabel("Number of Trips")

    sns.boxplot(data=train_df, x="passenger_count", ax=axes[1])
    axes[1].set_title("Passenger Count Box Plot (Train)")
    axes[1].set_xlabel("Passenger Count")

    plt.tight_layout()
    path = output_dir / "passenger_count_distribution.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_categorical_distributions(train_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=train_df, x="vendor_id", ax=axes[0])
    axes[0].set_title("Distribution of Vendor ID (Train)")
    axes[0].set_xlabel("Vendor ID")
    axes[0].set_ylabel("Number of Trips")

    sns.countplot(data=train_df, x="store_and_fwd_flag", ax=axes[1])
    axes[1].set_title("Distribution of Store and Fwd Flag (Train)")
    axes[1].set_xlabel("Store and Fwd Flag")
    axes[1].set_ylabel("Number of Trips")

    plt.tight_layout()
    path = output_dir / "categorical_feature_distributions.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_pickup_dropoff_map(
    train_df: pd.DataFrame,
    output_dir: Path,
    sample_size: int = 1000,
    random_state: int = 42,
) -> Path:
    nyc_center = [40.75, -73.98]
    m = folium.Map(location=nyc_center, zoom_start=12)

    sample_df = train_df.sample(n=min(len(train_df), sample_size), random_state=random_state)
    for _, row in sample_df.iterrows():
        folium.CircleMarker(
            location=[row["pickup_latitude"], row["pickup_longitude"]],
            radius=3,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
            tooltip=f"Pickup: {row['pickup_datetime']}",
        ).add_to(m)
        if "dropoff_datetime" in row.index:
            folium.CircleMarker(
                location=[row["dropoff_latitude"], row["dropoff_longitude"]],
                radius=3,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.6,
                tooltip=f"Dropoff: {row['dropoff_datetime']}",
            ).add_to(m)

    path = output_dir / "pickup_dropoff_locations.html"
    m.save(str(path))
    return path


def plot_haversine_distance_distribution(train_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(train_df["haversine_distance"], bins=50, kde=True, ax=axes[0])
    axes[0].set_title("Haversine Distance Distribution (Train)")
    axes[0].set_xlabel("Haversine Distance (km)")
    axes[0].set_ylabel("Frequency")

    sns.boxplot(x=train_df["haversine_distance"], ax=axes[1])
    axes[1].set_title("Haversine Distance Box Plot (Train)")
    axes[1].set_xlabel("Haversine Distance (km)")

    plt.tight_layout()
    path = output_dir / "haversine_distance_distributions.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_distance_vs_duration(
    train_df: pd.DataFrame,
    output_dir: Path,
    sample_size: int = 100_000,
    random_state: int = 42,
) -> Path:
    sample_df = train_df.sample(n=min(len(train_df), sample_size), random_state=random_state)

    fig = plt.figure(figsize=(12, 8))
    sns.scatterplot(x="haversine_distance", y="trip_duration", data=sample_df, alpha=0.5)
    plt.title("Haversine Distance vs. Trip Duration (Sampled)")
    plt.xlabel("Haversine Distance (km)")
    plt.ylabel("Trip Duration (seconds)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", c="0.7")

    path = output_dir / "haversine_distance_v_trip_duration.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_pickup_distribution_by_time(train_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    sns.countplot(data=train_df, x="pickup_hour", ax=axes[0])
    axes[0].set_title("Distribution of Pickups by Hour of Day (Train)")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Number of Trips")

    sns.countplot(data=train_df, x="pickup_day_of_week", ax=axes[1])
    axes[1].set_title("Distribution of Pickups by Day of Week (Train)")
    axes[1].set_xlabel("Day of Week (0=Monday, 6=Sunday)")
    axes[1].set_ylabel("Number of Trips")

    sns.countplot(data=train_df, x="pickup_month", ax=axes[2])
    axes[2].set_title("Distribution of Pickups by Month (Train)")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Number of Trips")

    plt.tight_layout()
    path = output_dir / "pickup_distribution_by_hourofday_dayofweek_month.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_duration_by_time(train_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    sns.barplot(data=train_df, x="pickup_hour", y="trip_duration", ax=axes[0], estimator=np.median)
    axes[0].set_title("Median Trip Duration by Hour of Day (Train)")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Median Trip Duration (seconds)")

    sns.barplot(data=train_df, x="pickup_day_of_week", y="trip_duration", ax=axes[1], estimator=np.median)
    axes[1].set_title("Median Trip Duration by Day of Week (Train)")
    axes[1].set_xlabel("Day of Week (0=Monday, 6=Sunday)")
    axes[1].set_ylabel("Median Trip Duration (seconds)")

    sns.barplot(data=train_df, x="pickup_month", y="trip_duration", ax=axes[2], estimator=np.median)
    axes[2].set_title("Median Trip Duration by Month (Train)")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Median Trip Duration (seconds)")

    plt.tight_layout()
    path = output_dir / "median_trip_duration_distribution_by_hourofday_dayofweek_month.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    return path
