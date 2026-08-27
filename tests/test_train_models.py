import numpy as np
import pandas as pd

from src.models.train_gradient_boosting import train_model as train_gradient_boosting
from src.models.train_linear_regression import train_model as train_linear_regression


def make_training_df(n_rows: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    distance = rng.uniform(0.5, 10.0, n_rows)
    duration = 60 + distance * 120 + rng.normal(0, 20, n_rows)
    return pd.DataFrame(
        {
            "id": [f"id{i}" for i in range(n_rows)],
            "pickup_datetime": pd.date_range("2016-01-01", periods=n_rows, freq="h"),
            "dropoff_datetime": pd.date_range("2016-01-01 00:10", periods=n_rows, freq="h"),
            "vendor_id": rng.choice([1, 2], n_rows),
            "store_and_fwd_flag": rng.choice(["Y", "N"], n_rows),
            "haversine_distance": distance,
            "pickup_hour": rng.integers(0, 24, n_rows),
            "pickup_day_of_week": rng.integers(0, 7, n_rows),
            "trip_duration": duration,
        }
    )


def test_train_linear_regression_fits_and_saves_model(tmp_path):
    model_path = tmp_path / "linear_regression.joblib"

    model = train_linear_regression(
        train_df=make_training_df(), model_path=model_path, log_to_mlflow=False
    )

    assert model_path.exists()
    prediction = model.predict(make_training_df(n_rows=3, seed=1).drop(columns=["trip_duration"]))
    assert len(prediction) == 3


def test_train_gradient_boosting_fits_and_saves_model(tmp_path):
    model_path = tmp_path / "gradient_boosting.joblib"

    model = train_gradient_boosting(
        train_df=make_training_df(), model_path=model_path, log_to_mlflow=False
    )

    assert model_path.exists()
    prediction = model.predict(make_training_df(n_rows=3, seed=1).drop(columns=["trip_duration"]))
    assert len(prediction) == 3
