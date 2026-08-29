"""Train a gradient boosting model for trip duration prediction.

Replaces notebooks/gradientBoosting.py. Trains incrementally via
warm_start so progress (MAE/RMSE/R2) is logged every `progress_every`
trees, exactly like the original script. Optionally logs params, the
per-stage progress curve, and the final model to MLflow.

Trains on the cleaned/feature-engineered dataset (src.features.cleaned_features
via src.features.cleaning_pipeline), not the older EDA-processed one -
see src.data.load.load_processed_train_test for that schema instead.
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import GRADIENT_BOOSTING, MODELS_DIR, RANDOM_STATE, TEST_SIZE, TRAIN_CLEANED_PATH
from src.data.load import load_cleaned_train_test
from src.models.common import (
    build_preprocessor,
    evaluate_regression,
    format_metrics,
    save_model,
    split_features_target,
)
from src.models.tracking import log_dataset_tags, start_run_if_enabled


def train_model(
    train_df: pd.DataFrame | None = None,
    model_path: Path | None = None,
    log_to_mlflow: bool = True,
    tracking_uri: str | None = None,
) -> Pipeline:
    if train_df is None:
        print("Loading cleaned training data...")
        train_df, _ = load_cleaned_train_test()
    train_df = train_df.replace([np.inf, -np.inf], np.nan)

    X, y = split_features_target(train_df)
    if X.empty:
        raise ValueError("The dataset contains no valid rows for training.")

    preprocessor = build_preprocessor(X, scale_numeric=True)
    regressor = GradientBoostingRegressor(
        n_estimators=GRADIENT_BOOSTING["n_estimators"],
        learning_rate=GRADIENT_BOOSTING["learning_rate"],
        max_depth=GRADIENT_BOOSTING["max_depth"],
        loss=GRADIENT_BOOSTING["loss"],
        random_state=RANDOM_STATE,
        warm_start=True,
    )
    model = Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    final_stage = GRADIENT_BOOSTING["n_estimators"]
    progress_every = GRADIENT_BOOSTING["progress_every"]

    with start_run_if_enabled("gradient_boosting", log_to_mlflow, tracking_uri) as run:
        if run is not None:
            mlflow.log_params(
                {
                    "n_estimators": GRADIENT_BOOSTING["n_estimators"],
                    "learning_rate": GRADIENT_BOOSTING["learning_rate"],
                    "max_depth": GRADIENT_BOOSTING["max_depth"],
                    "loss": GRADIENT_BOOSTING["loss"],
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                }
            )
            log_dataset_tags("cleaned", TRAIN_CLEANED_PATH)

        print("Starting model training...")
        for stage in range(progress_every, final_stage + 1, progress_every):
            model.named_steps["regressor"].set_params(n_estimators=stage)
            model.fit(X_train, y_train)
            metrics = evaluate_regression(y_test, model.predict(X_test))
            print(f"Training progress: {stage}/{final_stage} trees | {format_metrics(metrics)}")
            if run is not None:
                mlflow.log_metrics(metrics, step=stage)

        if final_stage % progress_every != 0:
            model.named_steps["regressor"].set_params(n_estimators=final_stage)
            model.fit(X_train, y_train)
            metrics = evaluate_regression(y_test, model.predict(X_test))
            print(f"Training progress: {final_stage}/{final_stage} trees | {format_metrics(metrics)}")
            if run is not None:
                mlflow.log_metrics(metrics, step=final_stage)

        print("Training complete. Final evaluation:")
        metrics = evaluate_regression(y_test, model.predict(X_test))
        print(format_metrics(metrics))
        if run is not None:
            mlflow.log_metrics({f"final_{key}": value for key, value in metrics.items()})
            mlflow.sklearn.log_model(
                model, name="model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
            )

    model_path = model_path or MODELS_DIR / GRADIENT_BOOSTING["model_filename"]
    save_model(model, model_path)
    return model


if __name__ == "__main__":
    train_model()
