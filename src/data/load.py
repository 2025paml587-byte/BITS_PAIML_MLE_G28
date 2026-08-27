"""Load train/test data, pulling from the DVC remote if not present
locally.

Replaces notebooks/ExtractData_Using_DVC.py. Works for both the raw
files (train.zip/test.csv) and the EDA-processed files
(train_eda_processed.zip/test_eda_processed.csv), since both are
DVC-tracked CSV-or-single-CSV-ZIP files.
"""

import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from src.config import (
    PROJECT_ROOT,
    TEST_PROCESSED_PATH,
    TEST_RAW_PATH,
    TRAIN_PROCESSED_PATH,
    TRAIN_RAW_PATH,
)


def dvc_pull(data_path: Path) -> None:
    """Pull data_path from the DVC remote if it isn't present locally."""
    if data_path.exists():
        return
    dvc_file = Path(f"{data_path}.dvc")
    if not dvc_file.exists():
        raise FileNotFoundError(f"DVC metadata file not found: {dvc_file}")
    subprocess.run(
        ["dvc", "pull", str(dvc_file.relative_to(PROJECT_ROOT))],
        cwd=PROJECT_ROOT,
        check=True,
    )


def load_dvc_tracked_csv(data_path: Path) -> pd.DataFrame:
    """Load a DVC-tracked CSV, or the single CSV inside a DVC-tracked ZIP,
    into a DataFrame."""
    dvc_pull(data_path)

    if data_path.suffix != ".zip":
        return pd.read_csv(data_path)

    with zipfile.ZipFile(data_path) as archive:
        csv_files = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_files) != 1:
            raise ValueError(
                f"Expected exactly one CSV in {data_path.name}, found {csv_files}"
            )
        with archive.open(csv_files[0]) as csv_file:
            return pd.read_csv(csv_file)


def load_raw_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw train and test sets."""
    return load_dvc_tracked_csv(TRAIN_RAW_PATH), load_dvc_tracked_csv(TEST_RAW_PATH)


def load_processed_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the EDA-processed train and test sets."""
    return (
        load_dvc_tracked_csv(TRAIN_PROCESSED_PATH),
        load_dvc_tracked_csv(TEST_PROCESSED_PATH),
    )


if __name__ == "__main__":
    train_df, test_df = load_processed_train_test()
    print(f"Loaded train_df: {len(train_df):,} rows and {len(train_df.columns):,} columns.")
    print(f"Loaded test_df: {len(test_df):,} rows and {len(test_df.columns):,} columns.")
