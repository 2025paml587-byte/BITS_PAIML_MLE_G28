"""Diagnostic charts for the cleaned/feature-engineered dataset,
ported from notebooks/featureengineering.py.

Uses the non-interactive Agg backend so this runs headless, matching
src/eda/plots.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_feature_engineering_charts(dataframe: pd.DataFrame, output_dir: Path) -> None:
    """Save EDA-style charts for the engineered dataset, plus compatibility plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_df = dataframe.sample(min(len(dataframe), 100_000), random_state=42)

    if "passenger_count" in sample_df.columns:
        figure, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.countplot(data=sample_df, x="passenger_count", ax=axes[0])
        axes[0].set_title("Passenger Count Distribution (Feature Engineered)")
        axes[0].set_xlabel("Passenger Count")
        axes[0].set_ylabel("Number of Trips")

        sns.boxplot(data=sample_df, x="passenger_count", ax=axes[1])
        axes[1].set_title("Passenger Count Box Plot (Feature Engineered)")
        axes[1].set_xlabel("Passenger Count")
        plt.tight_layout()
        figure.savefig(output_dir / "passenger_count_distribution.png", dpi=300)
        plt.close(figure)

    categorical_columns = [
        column for column in ("vendor_id", "store_and_fwd_flag", "pickup_season", "pickup_part_of_day")
        if column in sample_df.columns
    ]
    if categorical_columns:
        figure, axes = plt.subplots(1, min(len(categorical_columns), 2), figsize=(14, 5))
        for axis, column in zip(axes.flat, categorical_columns[: min(len(categorical_columns), 2)]):
            sample_df[column].value_counts(dropna=False).plot.bar(ax=axis)
            axis.set_title(f"Distribution of {column}")
            axis.set_xlabel(column)
            axis.set_ylabel("Number of Trips")
            axis.tick_params(axis="x", rotation=30)
        for axis in axes.flat[len(categorical_columns[: min(len(categorical_columns), 2)]):]:
            axis.remove()
        figure.tight_layout()
        figure.savefig(output_dir / "categorical_feature_distributions.png", dpi=300)
        plt.close(figure)

    if "haversine_distance" in sample_df.columns:
        figure, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.histplot(sample_df["haversine_distance"], bins=50, kde=True, ax=axes[0])
        axes[0].set_title("Haversine Distance Distribution (Feature Engineered)")
        axes[0].set_xlabel("Haversine Distance (km)")
        axes[0].set_ylabel("Frequency")

        sns.boxplot(x=sample_df["haversine_distance"], ax=axes[1])
        axes[1].set_title("Haversine Distance Box Plot (Feature Engineered)")
        axes[1].set_xlabel("Haversine Distance (km)")
        plt.tight_layout()
        figure.savefig(output_dir / "haversine_distance_distributions.png", dpi=300)
        plt.close(figure)

    if {"haversine_distance", "trip_duration"}.issubset(sample_df.columns):
        plotting_df = sample_df.sample(min(len(sample_df), 100_000), random_state=42)
        figure = plt.figure(figsize=(12, 8))
        sns.scatterplot(x="haversine_distance", y="trip_duration", data=plotting_df, alpha=0.5)
        plt.title("Haversine Distance vs. Trip Duration (Feature Engineered)")
        plt.xlabel("Haversine Distance (km)")
        plt.ylabel("Trip Duration (seconds)")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, which="both", ls="--", c="0.7")
        figure.tight_layout()
        figure.savefig(output_dir / "haversine_distance_v_trip_duration.png", dpi=300)
        plt.close(figure)

    time_columns = ["pickup_hour", "pickup_day_of_week", "pickup_month"]
    if all(column in sample_df.columns for column in time_columns):
        figure, axes = plt.subplots(3, 1, figsize=(16, 18))
        sns.countplot(data=sample_df, x="pickup_hour", ax=axes[0])
        axes[0].set_title("Distribution of Pickups by Hour of Day (Feature Engineered)")
        axes[0].set_xlabel("Hour of Day")
        axes[0].set_ylabel("Number of Trips")

        sns.countplot(data=sample_df, x="pickup_day_of_week", ax=axes[1])
        axes[1].set_title("Distribution of Pickups by Day of Week (Feature Engineered)")
        axes[1].set_xlabel("Day of Week (0=Monday, 6=Sunday)")
        axes[1].set_ylabel("Number of Trips")

        sns.countplot(data=sample_df, x="pickup_month", ax=axes[2])
        axes[2].set_title("Distribution of Pickups by Month (Feature Engineered)")
        axes[2].set_xlabel("Month")
        axes[2].set_ylabel("Number of Trips")
        plt.tight_layout()
        figure.savefig(output_dir / "pickup_distribution_by_hourofday_dayofweek_month.png", dpi=300)
        plt.close(figure)

    if all(column in sample_df.columns for column in ["pickup_hour", "pickup_day_of_week", "pickup_month", "trip_duration"]):
        figure, axes = plt.subplots(3, 1, figsize=(16, 18))
        sns.barplot(data=sample_df, x="pickup_hour", y="trip_duration", ax=axes[0], estimator=np.median)
        axes[0].set_title("Median Trip Duration by Hour of Day (Feature Engineered)")
        axes[0].set_xlabel("Hour of Day")
        axes[0].set_ylabel("Median Trip Duration (seconds)")

        sns.barplot(data=sample_df, x="pickup_day_of_week", y="trip_duration", ax=axes[1], estimator=np.median)
        axes[1].set_title("Median Trip Duration by Day of Week (Feature Engineered)")
        axes[1].set_xlabel("Day of Week (0=Monday, 6=Sunday)")
        axes[1].set_ylabel("Median Trip Duration (seconds)")

        sns.barplot(data=sample_df, x="pickup_month", y="trip_duration", ax=axes[2], estimator=np.median)
        axes[2].set_title("Median Trip Duration by Month (Feature Engineered)")
        axes[2].set_xlabel("Month")
        axes[2].set_ylabel("Median Trip Duration (seconds)")
        plt.tight_layout()
        figure.savefig(output_dir / "median_trip_duration_distribution_by_hourofday_dayofweek_month.png", dpi=300)
        plt.close(figure)

    if {"pickup_latitude", "pickup_longitude"}.issubset(sample_df.columns):
        nyc_center = [40.75, -73.98]
        map_object = folium.Map(location=nyc_center, zoom_start=12)
        sampling = sample_df.sample(min(len(sample_df), 1000), random_state=42)
        for _, row in sampling.iterrows():
            folium.CircleMarker(
                location=[row["pickup_latitude"], row["pickup_longitude"]],
                radius=3,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.6,
                tooltip="Pickup",
            ).add_to(map_object)
            if {"dropoff_latitude", "dropoff_longitude"}.issubset(sample_df.columns):
                folium.CircleMarker(
                    location=[row["dropoff_latitude"], row["dropoff_longitude"]],
                    radius=3,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.6,
                    tooltip="Dropoff",
                ).add_to(map_object)
        map_object.save(str(output_dir / "pickup_dropoff_locations.html"))

    distribution_columns = [
        column for column in (
            "haversine_distance", "manhattan_distance", "bearing",
            "distance_hour_interaction", "passenger_count", "trip_duration",
        )
        if column in sample_df.columns
    ]
    if distribution_columns:
        figure, axes = plt.subplots(2, 3, figsize=(18, 10))
        for axis, column in zip(axes.flat, distribution_columns):
            sns.histplot(data=sample_df, x=column, bins=40, ax=axis)
            axis.set_title(f"Distribution of {column}")
        for axis in axes.flat[len(distribution_columns):]:
            axis.remove()
        figure.tight_layout()
        figure.savefig(output_dir / "engineered_feature_distributions.png", dpi=200)
        plt.close(figure)

    numeric_data = sample_df.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        figure, axis = plt.subplots(figsize=(12, 9))
        sns.heatmap(numeric_data.corr(), cmap="coolwarm", center=0, ax=axis)
        axis.set_title("Numeric Feature Correlation Matrix")
        figure.tight_layout()
        figure.savefig(output_dir / "engineered_feature_correlation.png", dpi=200)
        plt.close(figure)

    categorical_columns = [
        column for column in ("vendor_id", "store_and_fwd_flag", "pickup_season", "pickup_part_of_day")
        if column in sample_df.columns
    ]
    if categorical_columns:
        figure, axes = plt.subplots(2, 2, figsize=(14, 10))
        for axis, column in zip(axes.flat, categorical_columns):
            sample_df[column].value_counts(dropna=False).plot.bar(ax=axis)
            axis.set_title(f"Counts of {column}")
            axis.tick_params(axis="x", rotation=30)
        for axis in axes.flat[len(categorical_columns):]:
            axis.remove()
        figure.tight_layout()
        figure.savefig(output_dir / "engineered_categorical_distributions.png", dpi=200)
        plt.close(figure)

    boxplot_columns = [
        column for column in ("haversine_distance", "manhattan_distance", "bearing", "trip_duration")
        if column in sample_df.columns
    ]
    if boxplot_columns:
        figure, axis = plt.subplots(figsize=(14, 7))
        sample_df[boxplot_columns].plot.box(ax=axis)
        axis.set_title("Engineered Feature Outlier Review")
        axis.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        figure.savefig(output_dir / "engineered_feature_boxplots.png", dpi=200)
        plt.close(figure)
