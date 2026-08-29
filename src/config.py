"""Central configuration for the pipeline.

Loads configs/config.yaml and exposes every path as an absolute Path
resolved against the project root, so scripts behave the same
regardless of the machine or working directory they're run from.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open(CONFIG_PATH, "r") as _f:
    _config = yaml.safe_load(_f)

_paths = _config["paths"]
_data = _config["data"]
_training = _config["training"]
_serving = _config["serving"]
_mlflow = _config["mlflow"]

# --- Paths -------------------------------------------------------------
TRAIN_RAW_PATH = PROJECT_ROOT / _paths["train_raw"]
TEST_RAW_PATH = PROJECT_ROOT / _paths["test_raw"]
TRAIN_PROCESSED_PATH = PROJECT_ROOT / _paths["train_processed"]
TEST_PROCESSED_PATH = PROJECT_ROOT / _paths["test_processed"]
TRAIN_CLEANED_PATH = PROJECT_ROOT / _paths["train_cleaned"]
TEST_CLEANED_PATH = PROJECT_ROOT / _paths["test_cleaned"]
EDA_OUTPUT_DIR = PROJECT_ROOT / _paths["eda_output_dir"]
FEATURE_ENGINEERING_OUTPUT_DIR = PROJECT_ROOT / _paths["feature_engineering_output_dir"]
EXTERNAL_DATA_DIR = PROJECT_ROOT / _paths["external_data_dir"]
MODELS_DIR = PROJECT_ROOT / _paths["models_dir"]
HIGH_TRAFFIC_ZONES_PATH = PROJECT_ROOT / _paths["high_traffic_zones"]

# --- DVC -----------------------------------------------------------------
DVC_REMOTE = _config["dvc"]["remote"]

# --- Data schema ---------------------------------------------------------
TARGET_COLUMN = _data["target_column"]
ID_COLUMN = _data["id_column"]
DATETIME_COLUMNS = _data["datetime_columns"]
DROP_COLUMNS_FOR_TRAINING = _data["drop_columns_for_training"]

# --- Training --------------------------------------------------------------
TEST_SIZE = _training["test_size"]
RANDOM_STATE = _training["random_state"]
LINEAR_REGRESSION = _training["linear_regression"]
GRADIENT_BOOSTING = _training["gradient_boosting"]

# --- Serving ---------------------------------------------------------------
DEFAULT_SERVING_MODEL = _serving["default_model"]
SERVING_MODELS = {name: PROJECT_ROOT / path for name, path in _serving["models"].items()}

# --- MLflow ------------------------------------------------------------------
# MLflow parses tracking URIs as actual URIs (scheme://...). A bare Windows
# path like "C:\...\mlruns" gets misread as scheme "c", so this must be a
# real file:// URI (Path.as_uri()), not a plain path string, to work on
# every OS.
MLFLOW_TRACKING_URI = (PROJECT_ROOT / _mlflow["tracking_uri"]).as_uri()
MLFLOW_EXPERIMENT_NAME = _mlflow["experiment_name"]
