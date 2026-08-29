"""Chunked read/process/write pipeline for the cleaned/feature-engineered
dataset stage, ported from notebooks/featureengineering.py.

Replaces that script's main(): reads the EDA-processed data, applies
src.features.cleaned_features chunk by chunk (so it scales to the full
~1.4M-row dataset), writes train_cleaned.zip/test_cleaned.csv, charts
the result, and DVC-tracks the outputs.
"""

import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from src.config import (
    EXTERNAL_DATA_DIR,
    FEATURE_ENGINEERING_OUTPUT_DIR,
    HIGH_TRAFFIC_ZONES_PATH,
    PROJECT_ROOT,
    TEST_CLEANED_PATH,
    TEST_PROCESSED_PATH,
    TRAIN_CLEANED_PATH,
    TRAIN_PROCESSED_PATH,
)
from src.features.cleaned_features import extract_location_features, process_chunk
from src.features.cleaning_plots import generate_feature_engineering_charts


def find_high_traffic_zones_from_csv(path: Path, chunksize: int = 100_000, quantile: float = 0.9) -> set:
    """Find traffic zones without loading the complete training file."""
    zone_counts = pd.Series(dtype="int64")
    for dataframe in pd.read_csv(path, chunksize=chunksize):
        location_features = extract_location_features(dataframe)
        zone_counts = zone_counts.add(
            pd.concat([location_features["pickup_zone"], location_features["dropoff_zone"]]).value_counts(),
            fill_value=0,
        )
    if zone_counts.empty:
        return set()
    return set(zone_counts[zone_counts >= zone_counts.quantile(quantile)].index)


def write_processed_csv(
    input_path: Path,
    output_path: Path,
    high_traffic_zones: set,
    external_data: list,
    compression: str | None = None,
    chunksize: int = 100_000,
) -> None:
    """Transform and stream a processed CSV, optionally inside a ZIP archive."""
    reader = pd.read_csv(input_path, chunksize=chunksize)
    if compression == "zip":
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            inner_name = os.path.basename(output_path).replace(".zip", ".csv")
            with archive.open(inner_name, "w") as binary_file:
                with io.TextIOWrapper(binary_file, encoding="utf-8", newline="") as text_file:
                    for chunk_number, dataframe in enumerate(reader):
                        process_chunk(dataframe, high_traffic_zones, external_data).to_csv(
                            text_file, index=False, header=chunk_number == 0
                        )
    else:
        with open(output_path, "w", encoding="utf-8", newline="") as text_file:
            for chunk_number, dataframe in enumerate(reader):
                process_chunk(dataframe, high_traffic_zones, external_data).to_csv(
                    text_file, index=False, header=chunk_number == 0
                )


def save_high_traffic_zones(zones: set, path: Path = HIGH_TRAFFIC_ZONES_PATH) -> None:
    """Persist the zones set so serving can reuse the exact same
    definition of "high traffic" the model was trained on, instead of
    guessing at inference time (a single request has no dataset to
    compute a frequency quantile from)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sorted(zones), f)


def load_high_traffic_zones(path: Path = HIGH_TRAFFIC_ZONES_PATH) -> set:
    """Load a previously persisted zones set, or an empty set if none
    has been saved yet."""
    if not path.exists():
        return set()
    with open(path, "r") as f:
        return set(json.load(f))


def load_external_data(path: Path) -> pd.DataFrame | None:
    """Load an optional external CSV, returning None when it is unavailable."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def track_cleaned_files_with_dvc(paths: list[Path]) -> None:
    subprocess.run(
        ["dvc", "add", *(str(p.relative_to(PROJECT_ROOT)) for p in paths)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print("Cleaned datasets tracked with DVC.")


def run_feature_cleaning(chunksize: int = 100_000) -> None:
    print("Finding high-traffic zones from EDA processed train data...")
    high_traffic_zones = find_high_traffic_zones_from_csv(TRAIN_PROCESSED_PATH, chunksize=chunksize)
    print(f"High-traffic zones identified: {len(high_traffic_zones)}")
    save_high_traffic_zones(high_traffic_zones)
    print(f"High-traffic zones saved to: {HIGH_TRAFFIC_ZONES_PATH}")

    external_data = []
    for filename in ("weather.csv", "traffic.csv"):
        frame = load_external_data(EXTERNAL_DATA_DIR / filename)
        if frame is not None:
            external_data.append(frame)

    print("Processing and saving train data...")
    write_processed_csv(
        TRAIN_PROCESSED_PATH, TRAIN_CLEANED_PATH, high_traffic_zones, external_data,
        compression="zip", chunksize=chunksize,
    )
    print("Processing and saving test data...")
    write_processed_csv(
        TEST_PROCESSED_PATH, TEST_CLEANED_PATH, high_traffic_zones, external_data, chunksize=chunksize
    )

    print("Generating feature engineering charts...")
    chart_sample = next(pd.read_csv(TRAIN_CLEANED_PATH, chunksize=chunksize))
    generate_feature_engineering_charts(chart_sample, FEATURE_ENGINEERING_OUTPUT_DIR)

    track_cleaned_files_with_dvc([TRAIN_CLEANED_PATH, TEST_CLEANED_PATH])
    print("Feature engineering completed and processed datasets saved successfully.")


if __name__ == "__main__":
    run_feature_cleaning()
