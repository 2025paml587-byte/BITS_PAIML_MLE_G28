"""One-time backfill: register the already-trained joblib models in
MLflow without retraining them.

These models were trained by the original notebook scripts before
MLflow tracking existed, so there's no historical params/metrics to
log for them - just the model artifact itself, clearly tagged as a
backfilled entry so it's not confused with a real training run.
"""

import joblib
import mlflow
import mlflow.sklearn

from src.config import SERVING_MODELS
from src.models.tracking import configure_mlflow


def backfill_model(name: str, path=None) -> None:
    path = path or SERVING_MODELS[name]
    if not path.exists():
        print(f"Skipping '{name}': file not found at {path}")
        return

    print(f"Loading '{name}' from {path}...")
    model = joblib.load(path)

    with mlflow.start_run(run_name=f"{name}-backfill"):
        mlflow.set_tag("backfilled", "true")
        mlflow.set_tag("source_path", str(path))
        mlflow.set_tag(
            "note",
            "Registered from a pre-existing joblib file trained before "
            "MLflow tracking was added; original training metrics are "
            "unavailable.",
        )
        mlflow.sklearn.log_model(
            model, name="model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
        )
    print(f"Backfilled '{name}' into MLflow.")


def backfill_all() -> None:
    configure_mlflow()
    for name in SERVING_MODELS:
        backfill_model(name)


if __name__ == "__main__":
    backfill_all()
