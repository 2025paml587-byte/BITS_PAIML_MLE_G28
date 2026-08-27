import numpy as np
import pandas as pd

from src.models.common import (
    build_preprocessor,
    evaluate_regression,
    format_metrics,
    split_features_target,
)


def make_processed_df(n_rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"id{i}" for i in range(n_rows)],
            "pickup_datetime": pd.date_range("2016-01-01", periods=n_rows, freq="h"),
            "dropoff_datetime": pd.date_range("2016-01-01 00:10", periods=n_rows, freq="h"),
            "vendor_id": [1, 2] * (n_rows // 2),
            "haversine_distance": np.linspace(0.5, 5.0, n_rows),
            "pickup_hour": list(range(n_rows)),
            "trip_duration": np.linspace(300, 1800, n_rows),
        }
    )


def test_split_features_target_drops_non_feature_columns():
    df = make_processed_df()
    X, y = split_features_target(df)

    assert "id" not in X.columns
    assert "pickup_datetime" not in X.columns
    assert "dropoff_datetime" not in X.columns
    assert "trip_duration" not in X.columns
    assert len(X) == len(y) == len(df)


def test_split_features_target_drops_rows_with_invalid_target():
    df = make_processed_df()
    df["trip_duration"] = df["trip_duration"].astype(object)
    df.loc[0, "trip_duration"] = "not-a-number"
    X, y = split_features_target(df)

    assert len(X) == len(df) - 1
    assert len(y) == len(df) - 1


def test_build_preprocessor_fits_and_transforms_mixed_types():
    X = pd.DataFrame(
        {
            "haversine_distance": [1.0, 2.0, None, 4.0],
            "vendor_id": ["a", "b", "a", None],
        }
    )
    preprocessor = build_preprocessor(X, scale_numeric=True)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 4
    # 1 numeric column + one-hot of 2 categories ("a", "b") = 3 columns.
    assert transformed.shape[1] == 3


def test_build_preprocessor_without_scaling_has_no_scaler_step():
    X = pd.DataFrame({"haversine_distance": [1.0, 2.0]})
    preprocessor = build_preprocessor(X, scale_numeric=False)
    numeric_pipeline = next(
        transformer for name, transformer, _ in preprocessor.transformers if name == "numeric"
    )

    assert "scaler" not in dict(numeric_pipeline.steps)


def test_evaluate_regression_returns_expected_metrics_for_perfect_predictions():
    y_true = [100, 200, 300]
    metrics = evaluate_regression(y_true, y_true)

    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["r2"] == 1


def test_format_metrics_produces_readable_string():
    text = format_metrics({"mae": 1.2345, "rmse": 2.3456, "r2": 0.8765})
    assert text == "MAE=1.234 | RMSE=2.346 | R2=0.876"
