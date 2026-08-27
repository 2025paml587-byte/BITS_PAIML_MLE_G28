"""Train a linear regression baseline for trip duration prediction.

Replaces notebooks/linearRegressionmodel.py.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import LINEAR_REGRESSION, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.data.load import load_processed_train_test
from src.models.common import (
    build_preprocessor,
    evaluate_regression,
    format_metrics,
    save_model,
    split_features_target,
)


def train_model(
    train_df: pd.DataFrame | None = None,
    model_path: Path | None = None,
) -> Pipeline:
    if train_df is None:
        print("Loading processed training data...")
        train_df, _ = load_processed_train_test()

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
    model.fit(X_train, y_train)
    metrics = evaluate_regression(y_test, model.predict(X_test))
    print(f"Training metrics: {format_metrics(metrics)}")

    model_path = model_path or MODELS_DIR / LINEAR_REGRESSION["model_filename"]
    save_model(model, model_path)
    return model


if __name__ == "__main__":
    train_model()
