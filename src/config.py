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

# --- Paths -------------------------------------------------------------
TRAIN_RAW_PATH = PROJECT_ROOT / _paths["train_raw"]
TEST_RAW_PATH = PROJECT_ROOT / _paths["test_raw"]
TRAIN_PROCESSED_PATH = PROJECT_ROOT / _paths["train_processed"]
TEST_PROCESSED_PATH = PROJECT_ROOT / _paths["test_processed"]
EDA_OUTPUT_DIR = PROJECT_ROOT / _paths["eda_output_dir"]
MODELS_DIR = PROJECT_ROOT / _paths["models_dir"]

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
