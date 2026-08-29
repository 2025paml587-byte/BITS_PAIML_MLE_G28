import zipfile

import numpy as np
import pandas as pd

from src.features.cleaning_pipeline import (
    find_high_traffic_zones_from_csv,
    load_external_data,
    write_processed_csv,
)


def make_eda_processed_df(n_rows: int = 250, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pickup = pd.date_range("2016-01-01", periods=n_rows, freq="h")
    dropoff = pickup + pd.to_timedelta(rng.integers(5, 60, n_rows), unit="m")
    return pd.DataFrame(
        {
            "id": [f"id{i}" for i in range(n_rows)],
            "vendor_id": rng.choice([1, 2], n_rows),
            "pickup_datetime": pickup,
            "dropoff_datetime": dropoff,
            "passenger_count": rng.integers(0, 7, n_rows),
            "pickup_longitude": rng.uniform(-74.0, -73.8, n_rows),
            "pickup_latitude": rng.uniform(40.6, 40.9, n_rows),
            "dropoff_longitude": rng.uniform(-74.0, -73.8, n_rows),
            "dropoff_latitude": rng.uniform(40.6, 40.9, n_rows),
            "store_and_fwd_flag": rng.choice(["Y", "N"], n_rows),
            "trip_duration": rng.uniform(120, 3600, n_rows),
            "haversine_distance": rng.uniform(0.1, 15, n_rows),
            "pickup_hour": pickup.hour,
            "pickup_day_of_week": pickup.dayofweek,
            "pickup_month": pickup.month,
            "pickup_day": pickup.day,
            "pickup_quarter": pickup.quarter,
        }
    )


def test_find_high_traffic_zones_from_csv_reads_in_chunks(tmp_path):
    csv_path = tmp_path / "train_eda_processed.csv"
    make_eda_processed_df(250).to_csv(csv_path, index=False)

    zones = find_high_traffic_zones_from_csv(csv_path, chunksize=100)
    assert isinstance(zones, set)
    assert len(zones) > 0


def test_write_processed_csv_zip_roundtrip_preserves_row_count(tmp_path):
    df = make_eda_processed_df(250)
    input_zip = tmp_path / "train_eda_processed.zip"
    with zipfile.ZipFile(input_zip, "w") as archive:
        archive.writestr("train_eda_processed.csv", df.to_csv(index=False))

    output_zip = tmp_path / "train_cleaned.zip"
    zones = find_high_traffic_zones_from_csv(input_zip, chunksize=100)
    write_processed_csv(input_zip, output_zip, zones, [], compression="zip", chunksize=100)

    with zipfile.ZipFile(output_zip) as archive:
        names = archive.namelist()
        assert names == ["train_cleaned.csv"]
        with archive.open(names[0]) as f:
            result = pd.read_csv(f)

    assert len(result) == len(df)
    assert [c for c in result.columns if c.startswith("dropoff_")] == []


def test_write_processed_csv_plain_csv_output(tmp_path):
    df = make_eda_processed_df(80, seed=1).drop(columns=["trip_duration"])
    input_csv = tmp_path / "test_eda_processed.csv"
    df.to_csv(input_csv, index=False)

    output_csv = tmp_path / "test_cleaned.csv"
    write_processed_csv(input_csv, output_csv, set(), [], chunksize=100)

    result = pd.read_csv(output_csv)
    assert len(result) == len(df)
    assert result.isnull().sum().sum() == 0


def test_load_external_data_returns_none_when_missing(tmp_path):
    assert load_external_data(tmp_path / "does_not_exist.csv") is None


def test_load_external_data_loads_existing_csv(tmp_path):
    path = tmp_path / "weather.csv"
    pd.DataFrame({"temperature": [1, 2]}).to_csv(path, index=False)
    result = load_external_data(path)
    assert list(result["temperature"]) == [1, 2]
