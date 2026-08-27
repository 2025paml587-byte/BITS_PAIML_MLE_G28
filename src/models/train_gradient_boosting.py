"""Train a gradient boosting model for trip duration prediction.

Replaces notebooks/gradientBoosting.py. Trains incrementally via
warm_start so progress (MAE/RMSE/R2) is logged every `progress_every`
trees, exactly like the original script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import GRADIENT_BOOSTING, MODELS_DIR, RANDOM_STATE, TEST_SIZE
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

    print("Starting model training...")
    for stage in range(progress_every, final_stage + 1, progress_every):
        model.named_steps["regressor"].set_params(n_estimators=stage)
        model.fit(X_train, y_train)
        metrics = evaluate_regression(y_test, model.predict(X_test))
        print(f"Training progress: {stage}/{final_stage} trees | {format_metrics(metrics)}")

    if final_stage % progress_every != 0:
        model.named_steps["regressor"].set_params(n_estimators=final_stage)
        model.fit(X_train, y_train)
        metrics = evaluate_regression(y_test, model.predict(X_test))
        print(f"Training progress: {final_stage}/{final_stage} trees | {format_metrics(metrics)}")

    print("Training complete. Final evaluation:")
    metrics = evaluate_regression(y_test, model.predict(X_test))
    print(format_metrics(metrics))

    model_path = model_path or MODELS_DIR / GRADIENT_BOOSTING["model_filename"]
    save_model(model, model_path)
    return model


if __name__ == "__main__":
    train_model()
