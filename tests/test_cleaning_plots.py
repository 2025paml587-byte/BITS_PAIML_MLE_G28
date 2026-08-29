import numpy as np
import pandas as pd

from src.features.cleaning_plots import generate_feature_engineering_charts


def make_cleaned_df(n_rows: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "vendor_id": rng.choice([1, 2], n_rows),
            "store_and_fwd_flag": rng.choice([0, 1], n_rows),
            "pickup_season": rng.choice(["winter", "summer"], n_rows),
            "pickup_part_of_day": rng.choice(["night", "morning"], n_rows),
            "haversine_distance": rng.uniform(0.1, 15, n_rows),
            "manhattan_distance": rng.uniform(0.1, 15, n_rows),
            "bearing": rng.uniform(0, 360, n_rows),
            "distance_hour_interaction": rng.uniform(0, 100, n_rows),
            "passenger_count": rng.integers(1, 6, n_rows),
            "trip_duration": rng.uniform(120, 3600, n_rows),
        }
    )


def test_generate_feature_engineering_charts_writes_all_expected_files(tmp_path):
    generate_feature_engineering_charts(make_cleaned_df(), tmp_path)

    expected = {
        "engineered_feature_distributions.png",
        "engineered_feature_correlation.png",
        "engineered_categorical_distributions.png",
        "engineered_feature_boxplots.png",
    }
    written = {p.name for p in tmp_path.iterdir()}
    assert expected.issubset(written)
    for name in expected:
        assert (tmp_path / name).stat().st_size > 0
