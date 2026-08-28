import mlflow
import numpy as np
import pandas as pd

from src.config import MLFLOW_EXPERIMENT_NAME
from src.models.tracking import configure_mlflow, start_run_if_enabled
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


def test_start_run_if_enabled_false_yields_none_and_starts_no_run():
    with start_run_if_enabled("unused", enabled=False) as run:
        assert run is None
    assert mlflow.active_run() is None


def test_start_run_if_enabled_true_creates_a_run(tmp_path):
    tracking_uri = f"file:{tmp_path}"
    with start_run_if_enabled("test-run", enabled=True, tracking_uri=tracking_uri) as run:
        assert run is not None
        assert mlflow.active_run() is not None
    assert mlflow.active_run() is None


def test_train_linear_regression_logs_params_and_metrics_to_mlflow(tmp_path):
    tracking_uri = f"file:{tmp_path}"
    model_path = tmp_path / "model.joblib"

    train_linear_regression(
        train_df=make_training_df(),
        model_path=model_path,
        log_to_mlflow=True,
        tracking_uri=tracking_uri,
    )

    configure_mlflow(tracking_uri)
    runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT_NAME])

    assert len(runs) == 1
    assert runs.iloc[0]["tags.mlflow.runName"] == "linear_regression"
    assert "metrics.mae" in runs.columns
    assert runs.iloc[0]["metrics.mae"] >= 0
    assert runs.iloc[0]["params.test_size"] == "0.2"
