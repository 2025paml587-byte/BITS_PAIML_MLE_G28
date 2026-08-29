"""MLflow experiment-tracking helpers shared by the training scripts.

Tracking is purely additive: every training function still produces
the same joblib file when `log_to_mlflow=False`, so nothing about the
existing pipeline behavior depends on MLflow being configured.
"""

import os
from contextlib import contextmanager
from pathlib import Path

import mlflow

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from src.data.load import read_dvc_md5

# MLflow 3.x puts the plain local filesystem store ("./mlruns") into
# maintenance mode by default and requires opting back in. A local
# SQL backend isn't worth the extra dependency (sqlalchemy/alembic)
# for a project this size, so we opt in explicitly.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def configure_mlflow(tracking_uri: str | None = None) -> None:
    mlflow.set_tracking_uri(tracking_uri or MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


@contextmanager
def start_run_if_enabled(run_name: str, enabled: bool, tracking_uri: str | None = None):
    """Yield an active MLflow run, or None if `enabled` is False.

    Lets callers write `if run is not None: mlflow.log_...(...)` around
    each logging call instead of branching the whole training function.
    """
    if not enabled:
        yield None
        return
    configure_mlflow(tracking_uri)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_dataset_tags(feature_set: str, dataset_path: Path) -> None:
    """Tag the active run with which feature-set and DVC-tracked data
    version fed it, so a run, its data, and DVC stay traceable to each
    other. No-op if there's no active run or no .dvc pointer yet."""
    if mlflow.active_run() is None:
        return
    mlflow.set_tag("feature_set", feature_set)
    mlflow.set_tag("dataset_path", str(dataset_path))
    md5 = read_dvc_md5(dataset_path)
    if md5 is not None:
        mlflow.set_tag("dataset_md5", md5)
