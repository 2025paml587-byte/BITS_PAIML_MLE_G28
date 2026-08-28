"""Generate the full EDA report: inspection printout, data-quality
checks, charts, and the feature-engineered processed datasets.

Replaces notebooks/EDA.py, built on top of src.data and src.features.
"""

import shutil
import subprocess

import pandas as pd

from src.config import EDA_OUTPUT_DIR, PROJECT_ROOT, TEST_PROCESSED_PATH, TRAIN_PROCESSED_PATH
from src.data.load import load_raw_train_test
from src.eda import plots
from src.eda.quality_checks import missing_value_report, run_quality_checks
from src.features.build_features import engineer_features


def inspect(df: pd.DataFrame, name: str) -> None:
    if df.empty:
        return
    print(f"\n--- {name} Head ---")
    print(df.head())
    print(f"\n--- {name} Info ---")
    df.info()
    print(f"\n--- {name} Description ---")
    print(df.describe())
    missing = missing_value_report(df)
    print(f"\n--- {name} Missing Values ---")
    print(missing if not missing.empty else "No missing values found.")


def generate_plots(train_df: pd.DataFrame) -> None:
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plots.plot_passenger_count(train_df, EDA_OUTPUT_DIR)
    plots.plot_categorical_distributions(train_df, EDA_OUTPUT_DIR)
    plots.plot_pickup_dropoff_map(train_df, EDA_OUTPUT_DIR)
    plots.plot_haversine_distance_distribution(train_df, EDA_OUTPUT_DIR)
    plots.plot_distance_vs_duration(train_df, EDA_OUTPUT_DIR)
    plots.plot_pickup_distribution_by_time(train_df, EDA_OUTPUT_DIR)
    plots.plot_duration_by_time(train_df, EDA_OUTPUT_DIR)
    print(f"\nAll EDA charts saved to: {EDA_OUTPUT_DIR}")


def save_and_track_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    TRAIN_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Train is zipped (it's large); test is left as a plain CSV, matching
    # what src.data.load already knows how to read.
    train_csv_path = TRAIN_PROCESSED_PATH.with_suffix(".csv")
    train_df.to_csv(train_csv_path, index=False)
    shutil.make_archive(
        str(TRAIN_PROCESSED_PATH.with_suffix("")),
        "zip",
        root_dir=train_csv_path.parent,
        base_dir=train_csv_path.name,
    )
    train_csv_path.unlink()
    print(f"Processed train_df zipped to: {TRAIN_PROCESSED_PATH}")

    test_df.to_csv(TEST_PROCESSED_PATH, index=False)
    print(f"Processed test_df saved to: {TEST_PROCESSED_PATH}")

    for path in (TRAIN_PROCESSED_PATH, TEST_PROCESSED_PATH):
        subprocess.run(
            ["dvc", "add", str(path.relative_to(PROJECT_ROOT))],
            cwd=PROJECT_ROOT,
            check=True,
        )

    print(
        "\nProcessed data staged and added to DVC. Next steps:\n"
        "  git add data/data_folder/train/processed/train_eda_processed.zip.dvc "
        "data/data_folder/test/processed/test_eda_processed.csv.dvc\n"
        "  git commit -m 'Track EDA-processed data with DVC'\n"
        "  dvc push"
    )


def run_eda_report() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading raw train/test data...")
    train_df, test_df = load_raw_train_test()

    train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
    train_df["dropoff_datetime"] = pd.to_datetime(train_df["dropoff_datetime"])
    test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])

    inspect(train_df, "Train DataFrame")
    inspect(test_df, "Test DataFrame")

    print("\n--- Data Quality Report ---")
    for key, value in run_quality_checks(train_df).items():
        print(f"{key}: {value}")

    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    generate_plots(train_df)
    save_and_track_processed_data(train_df, test_df)

    print("\nEDA and data preparation completed successfully.")
    return train_df, test_df


if __name__ == "__main__":
    run_eda_report()
