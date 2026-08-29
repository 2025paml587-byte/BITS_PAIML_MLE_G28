"""Train a linear regression baseline for trip duration prediction.

Replaces notebooks/linearRegressionmodel.py. Optionally logs the run
(params, metrics, model) to MLflow.

Trains on the cleaned/feature-engineered dataset (src.features.cleaned_features
via src.features.cleaning_pipeline), not the older EDA-processed one -
see src.data.load.load_processed_train_test for that schema instead.
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import LINEAR_REGRESSION, MODELS_DIR, RANDOM_STATE, TEST_SIZE, TRAIN_CLEANED_PATH
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

    X, y = split_features_target(train_df)
    if X.empty:
        raise ValueError("The dataset contains no valid rows for training.")

    preprocessor = build_preprocessor(X, scale_numeric=False)
    model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("Training linear regression model...")
    with start_run_if_enabled("linear_regression", log_to_mlflow, tracking_uri) as run:
        if run is not None:
            mlflow.log_params({"test_size": TEST_SIZE, "random_state": RANDOM_STATE})
            log_dataset_tags("cleaned", TRAIN_CLEANED_PATH)

        model.fit(X_train, y_train)
        metrics = evaluate_regression(y_test, model.predict(X_test))
        print(f"Training metrics: {format_metrics(metrics)}")

        if run is not None:
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model, name="model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
            )

    model_path = model_path or MODELS_DIR / LINEAR_REGRESSION["model_filename"]
    save_model(model, model_path)
    return model


if __name__ == "__main__":
    train_model()
