"""Shared preprocessing, evaluation, and persistence helpers for the
trip-duration regression models.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DROP_COLUMNS_FOR_TRAINING, TARGET_COLUMN


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a processed DataFrame into a feature matrix and target
    series, dropping non-feature columns and rows with an invalid
    target."""
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    X = df.drop(columns=DROP_COLUMNS_FOR_TRAINING + [TARGET_COLUMN], errors="ignore")
    valid = y.notna()
    return X.loc[valid], y.loc[valid]


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes numeric/categorical
    columns (and optionally scales numeric ones) and one-hot encodes
    categoricals."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers)


def evaluate_regression(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def format_metrics(metrics: dict) -> str:
    return f"MAE={metrics['mae']:.3f} | RMSE={metrics['rmse']:.3f} | R2={metrics['r2']:.3f}"


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")
