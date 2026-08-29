"""Diagnostic charts for the cleaned/feature-engineered dataset,
ported from notebooks/featureengineering.py.

Uses the non-interactive Agg backend so this runs headless, matching
src/eda/plots.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_feature_engineering_charts(dataframe: pd.DataFrame, output_dir: Path) -> None:
    """Save bounded-sample diagnostics for the engineered training features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = dataframe.sample(min(len(dataframe), 100_000), random_state=42)

    distribution_columns = [
        column for column in (
            "haversine_distance", "manhattan_distance", "bearing",
            "distance_hour_interaction", "passenger_count", "trip_duration",
        )
        if column in dataframe.columns
    ]
    if distribution_columns:
        figure, axes = plt.subplots(2, 3, figsize=(18, 10))
        for axis, column in zip(axes.flat, distribution_columns):
            sns.histplot(data=dataframe, x=column, bins=40, ax=axis)
            axis.set_title(f"Distribution of {column}")
        for axis in axes.flat[len(distribution_columns):]:
            axis.remove()
        figure.tight_layout()
        figure.savefig(output_dir / "engineered_feature_distributions.png", dpi=200)
        plt.close(figure)

    numeric_data = dataframe.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        figure, axis = plt.subplots(figsize=(12, 9))
        sns.heatmap(numeric_data.corr(), cmap="coolwarm", center=0, ax=axis)
        axis.set_title("Numeric Feature Correlation Matrix")
        figure.tight_layout()
        figure.savefig(output_dir / "engineered_feature_correlation.png", dpi=200)
        plt.close(figure)

    categorical_columns = [
        column for column in ("vendor_id", "store_and_fwd_flag", "pickup_season", "pickup_part_of_day")
        if column in dataframe.columns
    ]
    if categorical_columns:
        figure, axes = plt.subplots(2, 2, figsize=(14, 10))
        for axis, column in zip(axes.flat, categorical_columns):
            dataframe[column].value_counts(dropna=False).plot.bar(ax=axis)
            axis.set_title(f"Counts of {column}")
            axis.tick_params(axis="x", rotation=30)
        for axis in axes.flat[len(categorical_columns):]:
            axis.remove()
        figure.tight_layout()
        figure.savefig(output_dir / "engineered_categorical_distributions.png", dpi=200)
        plt.close(figure)

    boxplot_columns = [
        column for column in ("haversine_distance", "manhattan_distance", "bearing", "trip_duration")
        if column in dataframe.columns
    ]
    if boxplot_columns:
        figure, axis = plt.subplots(figsize=(14, 7))
        dataframe[boxplot_columns].plot.box(ax=axis)
        axis.set_title("Engineered Feature Outlier Review")
        axis.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        figure.savefig(output_dir / "engineered_feature_boxplots.png", dpi=200)
        plt.close(figure)
