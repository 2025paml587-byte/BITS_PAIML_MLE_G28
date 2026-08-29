# ML Engineering Mini-Project: Delivery / Ride ETA Prediction

This project aims to build an end-to-end Machine Learning pipeline for predicting Delivery / Ride Estimated Time of Arrival (ETA) based on various factors. It is structured to demonstrate key MLOps principles including data acquisition, feature engineering, model training, deployment, monitoring, and documentation.

**Team:** BITS PILANI AI/ML Group 28  
**Project Focus:** NYC Taxi Trip Duration Prediction  
**Models:** Linear Regression, Gradient Boosting

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Core Modules](#core-modules-src)
- [ML Pipeline Architecture](#ml-pipeline-architecture)
- [Running Experiments](#running-experiments)
- [Training Models](#training-models)
- [Dataset & Features](#dataset--features)
- [Data Download](#data-download)
- [Configuration](#configuration)
- [Running the API](#running-the-api)
- [Docker Deployment](#docker-deployment)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [MLflow Integration](#mlflow-integration)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [DVC Data Setup](#dvc-data-setup-for-team-members)

## Prerequisites

- **Python:** 3.8 or higher
- **pip:** Package manager for Python
- **Git:** Version control
- **DVC:** Data versioning (included in requirements)
- **Google Account:** For accessing shared DVC remote (Google Drive)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd BITS_PAIML_MLE_G28
```

### 2. Create a Virtual Environment
```bash
# Using venv
python -m venv .venv

# Activate the environment
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup DVC and Download Data
```bash
dvc pull
```

## Project Structure:
- `configs/`: Configuration files for the project.
- `data/`: Raw and processed data files.
- `docs/`: Project documentation and reports.
- `models/`: Trained machine learning models.
- `notebooks/`: Jupyter/Colab notebooks for experimentation and analysis.
- `src/`: Source code for the ML pipeline components (e.g., data processing, model training, API).

## Project Structure:
- `configs/`: Configuration files (YAML) for paths, models, and hyperparameters
- `data/`: Raw and processed data files managed by DVC
- `docs/`: Project documentation, EDA reports, and analysis charts
- `models/`: Trained machine learning models (joblib format)
- `notebooks/`: Jupyter/Colab notebooks for experimentation and analysis
- `src/`: Source code for ML pipeline components
  - `api/`: FastAPI service for model serving
  - `data/`: Data loading and staging
  - `eda/`: Exploratory Data Analysis and quality checks
  - `features/`: Feature engineering and data cleaning
  - `models/`: Model training and tracking
- `tests/`: Unit and integration tests (pytest)
- `pyproject.toml`: Project configuration and pytest settings
- `requirements.txt`: Python package dependencies

### Core Modules (src/)

#### **Configuration & Entry Point**
- **`src/config.py`**: Centralized configuration loader
  - Loads `configs/config.yaml` with absolute path resolution
  - Exports: Training parameters, model paths, MLflow settings
  - Used by all modules for consistent configuration

- **`src/main.py`**: Project entry point (placeholder for future orchestration)

#### **API & Inference** (`src/api/`)
- **`src/api/schemas.py`**: Pydantic request/response models
  - `TripRequest`: Input validation (vendor_id, coords, datetime, etc.)
  - `PredictionResponse`: Standardized output format

- **`src/api/inference.py`**: Model loading and prediction logic
  - Lazy-loads models from joblib (cached in memory)
  - Applies feature engineering to requests
  - Reorders features to match model's training order

- **`src/api/app.py`**: FastAPI server
  - `GET /`: Interactive HTML UI
  - `GET /health`: Server status check
  - `GET /models`: List available models
  - `POST /predict`: Make predictions with validated input
  - Auto-generated Swagger docs at `/docs`

#### **Data Loading** (`src/data/`)
- **`src/data/load.py`**: Unified data loading with automatic DVC sync
  - `dvc_pull()`: Fetches data from Google Drive if needed
  - `load_dvc_tracked_csv()`: Handles ZIP files transparently
  - `load_cleaned_train_test()`: Used by training scripts
  - `read_dvc_md5()`: Reads data version hash for reproducibility

- **`src/data/stage_raw.py`**: DVC initialization and staging
  - Copies raw files to DVC-tracked locations
  - Runs `dvc add` for version control
  - Used in initial data pipeline setup

#### **EDA & Quality Checks** (`src/eda/`)
- **`src/eda/quality_checks.py`**: Data validation functions
  - 6 checks: missing values, duplicates, negative durations, invalid times, coordinates, flags
  - Pure functions for easy testing

- **`src/eda/plots.py`**: 7 visualization functions
  - Passenger count, categorical distributions, pickup/dropoff map (Folium)
  - Distance distributions, time-based patterns
  - Saves to `docs/EDA_chart_outputs/` (300 DPI PNG)

- **`src/eda/report.py`**: Full EDA orchestration
  - Loads data, runs quality checks, generates visualizations
  - Saves processed data and tracks with DVC
  - Entry point: `python -m src.eda.report`

#### **Feature Engineering** (`src/features/`)
- **`src/features/build_features.py`**: Basic features (used by API)
  - Vectorized: Haversine distance, temporal features (hour, day, month)
  - Lightweight, production-ready

- **`src/features/cleaned_features.py`**: Advanced features (for retraining)
  - Enhanced: Holidays, seasons, bearing, grid zones, interactions
  - Memory-efficient chunked processing support
  - Optional external data integration (weather, traffic)

- **`src/features/cleaning_pipeline.py`**: Chunked data processing
  - Processes 1.4M rows in 100K-row chunks
  - Finds high-traffic zones, applies feature engineering
  - Memory-efficient streaming read-process-write

- **`src/features/cleaning_plots.py`**: Feature diagnostic charts
  - Distribution plots, correlation heatmap, categorical counts
  - Quality assurance for engineered features

#### **Model Training & Tracking** (`src/models/`)
- **`src/models/common.py`**: Shared utilities
  - `split_features_target()`: Separates features from target
  - `build_preprocessor()`: Creates sklearn pipeline (impute, scale, encode)
  - `evaluate_regression()`: Computes MAE, RMSE, R²
  - `save_model()`: Persists model to joblib

- **`src/models/tracking.py`**: MLflow integration
  - `configure_mlflow()`: Sets tracking server and experiment
  - `start_run_if_enabled()`: Conditional MLflow context manager
  - `log_dataset_tags()`: Logs data version and feature set
  - Enables complete reproducibility tracking

- **`src/models/train_linear_regression.py`**: Linear regression training (production)
  - Config-driven, no feature scaling
  - Full MLflow logging support

- **`src/models/train_gradient_boosting.py`**: Gradient boosting training (production)
  - Warm-start training with progress logging every 25 trees
  - Logs MAE/RMSE/R² at each stage to MLflow

- **`src/models/backfill_existing_models.py`**: Register legacy models
  - Loads pre-existing joblib models into MLflow
  - Enables historical experiment comparison

## Project Structure:

The ML pipeline follows these sequential stages:

```
Data Ingestion → Raw Data Storage (DVC)
        ↓
Exploratory Data Analysis (EDA)
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Engineering
        ↓
Model Training (Linear Regression, Gradient Boosting)
        ↓
Model Evaluation & Tracking (MLflow)
        ↓
Model Serving (FastAPI)
        ↓
Monitoring & Inference
```

## Dataset & Features

**Dataset:** NYC Taxi Trip Duration Dataset  
**Target Variable:** `trip_duration` (minutes)  
**Training Set:** ~1.46M records  
**Test Set:** ~625K records  

**Key Features:**
- Pickup & Dropoff coordinates (latitude, longitude)
- Pickup datetime and time-based features
- Trip type and vendor information
- Passenger count
- Distance metrics
- Temporal features (hour, day, month, etc.)

## Configuration

The project uses a centralized configuration file at `configs/config.yaml`:

```yaml
paths:
  train_raw: data/data_folder/train/raw/train.zip
  test_raw: data/data_folder/test/raw/test.csv
  models_dir: models

dvc:
  remote: cloud

mlflow:
  tracking_uri: mlruns
  experiment_name: trip-duration-prediction

serving:
  default_model: gradient_boosting
  models:
    gradient_boosting: models/gradient_boosting_trip_duration.joblib
    linear_regression: models/linear_regression_trip_duration.joblib
```

All paths are relative to the project root and are automatically resolved by `src/config.py`.

## Running the API

The project includes a FastAPI server for real-time ETA predictions.

### Start the API Server
```bash
# Activate virtual environment first
.\.venv\Scripts\activate

# Run the server
uvicorn src.api.app:app --reload
```

**Access:**
- **UI:** Open http://127.0.0.1:8000/ in your browser
- **API Docs:** http://127.0.0.1:8000/docs (Swagger UI)
- **Alternative Docs:** http://127.0.0.1:8000/redoc (ReDoc)

## Docker Deployment

The prediction API (FastAPI + UI) can be packaged as a single, self-contained
Docker image with the two pre-trained models baked in - no DVC, MLflow, or
Python setup needed to run it. This is the easiest way to hand someone (e.g.
an invigilator) a working demo without sharing the whole repo or having them
install anything beyond Docker itself.

The image is built from `Dockerfile` at the repo root (multi-stage: deps are
installed in a builder stage, the final image is `python:3.12-slim`, runs as
a non-root user, and only contains `src/`, `configs/`, and `models/*.joblib`
- no raw/processed data, no notebooks, no `.git`). Built size is ~165 MB.

### Build

```bash
docker build -t trip-duration-api .
```

### Run

```bash
docker run -p 8000:8000 trip-duration-api
```

Then open:
- **UI:** http://127.0.0.1:8000/
- **Swagger docs:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

### Sharing the image

**Option A - via a container registry (recommended):** push once, anyone can pull.

```bash
docker tag trip-duration-api <your-dockerhub-username>/trip-duration-api:latest
docker push <your-dockerhub-username>/trip-duration-api:latest
```

The recipient then just needs:

```bash
docker pull <your-dockerhub-username>/trip-duration-api:latest
docker run -p 8000:8000 <your-dockerhub-username>/trip-duration-api:latest
```

**Option B - as a portable file (no registry account needed):**

```bash
docker save -o trip-duration-api.tar trip-duration-api
```

Share `trip-duration-api.tar` directly (drive/email/USB). The recipient loads
and runs it with:

```bash
docker load -i trip-duration-api.tar
docker run -p 8000:8000 trip-duration-api
```

## API Endpoints

### `GET /health`
Check if the API is running.
```bash
curl http://127.0.0.1:8000/health
```
**Response:**
```json
{"status": "ok"}
```

### `GET /models`
List available models and the default model.
```bash
curl http://127.0.0.1:8000/models
```
**Response:**
```json
{
  "available_models": ["gradient_boosting", "linear_regression"],
  "default_model": "gradient_boosting"
}
```

### `POST /predict`
Predict trip duration for a given trip.

**Request Body:**
```json
{
  "pickup_datetime": "2023-01-15 14:30:00",
  "dropoff_datetime": "2023-01-15 14:45:00",
  "pickup_longitude": -73.97,
  "pickup_latitude": 40.77,
  "dropoff_longitude": -73.98,
  "dropoff_latitude": 40.76,
  "passenger_count": 1,
  "vendor_id": 2,
  "trip_type": 1,
  "algorithm": "gradient_boosting"
}
```

**Response:**
```json
{
  "predicted_trip_duration": 15.4,
  "model_used": "gradient_boosting",
  "unit": "minutes"
}
```

## Training Models

### Overview
The project provides two model implementations:

1. **Notebook Models** (`notebooks/`): Exploration-focused, minimal dependencies
2. **Production Models** (`src/models/`): Full MLflow tracking, config-driven, modular

### Production Training (Recommended)

#### Train Linear Regression Model
```bash
python -m src.models.train_linear_regression
```

**What it does:**
- Loads cleaned/feature-engineered data via DVC
- Builds sklearn pipeline with imputation and preprocessing
- Trains on 80% of data, validates on 20%
- Logs metrics to MLflow (MAE, RMSE, R²)
- Saves model to `models/linear_regression_trip_duration.joblib`

**Key Parameters (from `configs/config.yaml`):**
- No feature scaling (raw features used)
- Random state: 42 (reproducibility)
- Test size: 20%

#### Train Gradient Boosting Model
```bash
python -m src.models.train_gradient_boosting
```

**What it does:**
- Loads cleaned data and applies feature engineering
- Creates a warm-start Gradient Boosting regressor
- Logs progress every 25 trees (300 trees total)
- Tracks MAE/RMSE/R² at each stage
- Saves final model to `models/gradient_boosting_trip_duration.joblib`

**Key Parameters (from `configs/config.yaml`):**
```yaml
GRADIENT_BOOSTING:
  n_estimators: 300
  learning_rate: 0.1
  max_depth: 5
  loss: "huber"  # Robust to outliers
  progress_every: 25  # Log metrics every 25 trees
```

**Training Progress Output:**
```
Loading cleaned training data...
Splitting data into train and test sets...
Starting model training...
Training progress: 25/300 trees | MAE=185.32 | RMSE=320.15 | R2=0.7023
Training progress: 50/300 trees | MAE=178.91 | RMSE=310.45 | R2=0.7156
...
Training progress: 300/300 trees | MAE=165.25 | RMSE=295.67 | R2=0.7412
Training complete.
```

### Notebook Models (Exploration)

#### Gradient Boosting Notebook
```bash
python notebooks/gradientBoosting.py
```

**Features:**
- Warm-start training with progress output
- Saves model to disk
- No MLflow integration (standalone)
- Processes EDA-processed data

#### Linear Regression Notebook
```bash
python notebooks/linearRegressionmodel.py
```

**Features:**
- Single-pass training
- No feature scaling
- Baseline performance metrics
- Minimal dependencies

### Model Comparison

| Aspect | Linear Regression | Gradient Boosting |
|--------|-------------------|-------------------|
| **Training Time** | ~1 minute | ~5-10 minutes (warm-start) |
| **R² Score** | ~0.65-0.68 | ~0.74-0.76 |
| **MAE (seconds)** | ~200-220 | ~160-180 |
| **Interpretability** | High (coefficients) | Low (tree ensemble) |
| **Hyperparameters** | 0 (default sklearn) | ~4 (n_est, lr, depth, loss) |
| **Use Case** | Baseline, fast inference | Production, best accuracy |

### MLflow Experiment Tracking

Both production models automatically log to MLflow:

**Tracked Information:**
- **Parameters**: Hyperparameters (learning_rate, n_estimators, etc.)
- **Metrics**: MAE, RMSE, R² at various stages
- **Tags**: Data version (DVC MD5), feature set, dataset path
- **Artifacts**: Trained model pickle file

**View Experiments:**
```bash
mlflow ui
```
Open http://127.0.0.1:5000 to compare runs

### Training Data Versions

Models are trained on:
- **Cleaned Dataset**: Enhanced features (temporal, spatial, interaction)
- **Location**: `data/data_folder/train/processed/train_cleaned.zip`
- **Size**: Feature-engineered subset of original data
- **DVC Tracked**: Version hash logged for reproducibility

### Custom Training

For advanced use cases, train models programmatically:

```python
from src.models.train_gradient_boosting import train_model
from pathlib import Path

# Train with custom data
model = train_model(
    train_df=my_dataframe,
    model_path=Path("models/custom_gb.joblib"),
    log_to_mlflow=True
)

# Predictions
predictions = model.predict(X_test)
```

Models are saved to the `models/` directory and tracked in MLflow.

## ML Pipeline Architecture

### Complete Data & Training Flow

The project implements a comprehensive ML pipeline with clear separation between exploration (notebooks/) and production (src/):

```
EXPLORATION PHASE (Notebooks)
├─ notebooks/EDA.py
│  └─ Loads raw data → Quality checks → Visualizations
├─ notebooks/featureengineering.py
│  └─ Explores feature engineering techniques
├─ notebooks/gradientBoosting.py
│  └─ Experiments with GB model (warm-start)
└─ notebooks/linearRegressionmodel.py
   └─ Baseline linear regression

         ↓ (Validated approaches)

PRODUCTION PHASE (src/)
├─ src/data/
│  ├─ load.py → DVC auto-sync, transparent ZIP handling
│  └─ stage_raw.py → DVC initialization
├─ src/eda/
│  ├─ quality_checks.py → 6 data validation functions
│  ├─ plots.py → 7 visualization functions
│  └─ report.py → Full EDA orchestration
├─ src/features/
│  ├─ build_features.py → API-ready features
│  ├─ cleaned_features.py → Advanced features (training)
│  ├─ cleaning_pipeline.py → Chunked processing
│  └─ cleaning_plots.py → Feature diagnostics
├─ src/models/
│  ├─ common.py → Preprocessing, evaluation utilities
│  ├─ tracking.py → MLflow integration
│  ├─ train_linear_regression.py → Production training
│  ├─ train_gradient_boosting.py → Production training
│  └─ backfill_existing_models.py → Legacy registration
└─ src/api/
   ├─ app.py → FastAPI server (GET /health, /models, POST /predict)
   ├─ inference.py → Model loading & prediction
   └─ schemas.py → Pydantic validation
```

### Data Pipeline Stages

**Stage 1: Raw Data (DVC Storage)**
- Location: Google Drive (remote) + `data/data_folder/train/raw/` (local)
- Files: train.zip (1.4M rows), test.csv (625K rows)
- Versioning: DVC + `.dvc` metadata files

**Stage 2: EDA-Processed Data**
- Applied basic haversine distance and temporal features
- Location: `data/data_folder/train/processed/train_eda_processed.zip`
- Used by: notebooks/EDA.py, initial exploration

**Stage 3: Feature-Engineered (Cleaned) Data**
- Applied 40+ advanced features (holidays, seasons, grid zones, interactions)
- Location: `data/data_folder/train/processed/train_cleaned.zip`
- Used by: Model training (both linear & GB)

### Inference Pipeline

When a prediction request arrives at the API:
```
HTTP Request (TripRequest)
    ↓ [src/api/app.py]
Validate & Parse JSON
    ↓ [src/api/inference.py]
Load Model (cached in memory)
    ↓ [src/features/build_features.py]
Engineer Features
    - Haversine distance
    - Pickup hour, day of week, month
    ↓ [Model Pipeline]
Preprocessing:
    - Impute missing values
    - Scale numeric features
    - Encode categorical features
    ↓ [Regressor]
Linear Regression OR Gradient Boosting
    ↓
HTTP Response (PredictionResponse)
    └─ predicted_trip_duration_seconds
    └─ model_used
```

### Configuration System

All settings are centralized in `configs/config.yaml` and loaded by `src/config.py`:

**Paths Section:**
```yaml
paths:
  train_raw: data/data_folder/train/raw/train.zip
  train_processed: data/data_folder/train/processed/train_eda_processed.zip
  train_cleaned: data/data_folder/train/processed/train_cleaned.zip
  models_dir: models
  eda_output: docs/EDA_chart_outputs
  feature_engineering_output: docs/feature_engg_chart_outputs
```

**Model Parameters Section:**
```yaml
LINEAR_REGRESSION:
  model_class: LinearRegression
  scale_numeric: false  # No feature scaling

GRADIENT_BOOSTING:
  n_estimators: 300       # Total trees
  learning_rate: 0.1      # Step size
  max_depth: 5            # Tree depth
  loss: huber             # Robust to outliers
  progress_every: 25      # Log metrics every 25 trees
```

**Experiment Tracking:**
```yaml
mlflow:
  tracking_uri: mlruns/
  experiment_name: trip-duration-prediction
```

## Running Experiments

### Exploratory Data Analysis
```bash
python notebooks/EDA.py
```
Outputs: HTML charts and visualizations in `docs/EDA_chart_outputs/`

### Feature Engineering
```bash
python notebooks/featureengineering.py
```

### Data Extraction with DVC
```bash
python notebooks/ExtractData_Using_DVC.py
```

### Data Fetching
```bash
python notebooks/FetchingData.py
```

### Pre-EDA with DVC
```bash
python notebooks/pre-EDA-dvc.py
```

### Notebook Scripts Reference

#### **EDA.py** - Exploratory Data Analysis
Comprehensive analysis of the NYC taxi dataset:
- **Data Loading**: Loads training and test sets from ZIP/CSV
- **Inspection**: DataFrame info, descriptive statistics, missing values
- **Univariate Analysis**: Passenger count, categorical distributions, haversine distance
- **Bivariate Analysis**: Distance vs duration, temporal patterns, vendor/payment comparisons
- **Temporal Features**: Pickup patterns by hour, day of week, and month
- **Geographic Mapping**: Interactive Folium maps of pickup/dropoff locations
- **Output**: Charts saved to `docs/EDA_chart_outputs/` (HTML, PNG)

#### **featureengineering.py** - Advanced Feature Engineering
Develops rich feature set for model training:
- **Time Features**: Holidays (US Federal), seasons, part of day, temporal interactions
- **Location Features**: Manhattan/Haversine distance, bearing, grid zones, route zones
- **Trip Features**: Normalization, interaction terms
- **High-Traffic Detection**: Identifies frequently-used zones (90th percentile)
- **External Data**: Integrates optional weather/traffic data
- **Output**: Feature-engineered dataset with 40+ computed features

#### **pre-EDA-dvc.py** - DVC Pipeline Setup
Initializes data versioning:
- **Directory Setup**: Creates data folder structure
- **File Staging**: Copies raw files to `data/data_folder/`
- **DVC Initialization**: Runs `dvc init` and `dvc add`
- **Version Control**: Enables data tracking and remote storage
- **Output**: `.dvc` metadata files for reproducibility

#### **ExtractData_Using_DVC.py** - DVC Data Extraction
Demonstrates transparent data handling:
- **Automatic Sync**: Checks if files exist, pulls from Google Drive if needed
- **ZIP Handling**: Decompresses data transparently to pandas
- **MD5 Versioning**: Reads data version hash for reproducibility tracking
- **Output**: DataFrames loaded directly into memory

#### **FetchingData.py** - Data Fetching (Template)
Kaggle API template (currently inactive):
- Replaced by Google Drive + DVC for this project
- Kept for reference in case Kaggle data is needed
- Requires Kaggle API credentials setup

#### **gradientBoosting.py** - Model Training (Notebook Version)
Trains gradient boosting model for experimentation:
- **Warm-Start Training**: Incrementally adds trees (300 total), logs progress every 25 trees
- **Feature Preparation**: Numeric scaling, categorical encoding
- **Evaluation Metrics**: MAE, RMSE, R² on validation set
- **Model Persistence**: Saves to `models/gradient_boosting_trip_duration.joblib`
- **Output**: Trained model + console metrics

#### **linearRegressionmodel.py** - Linear Regression (Notebook Version)
Baseline linear regression model:
- **Simple Preprocessing**: No feature scaling, straightforward pipeline
- **Fast Training**: Single-pass training on full dataset
- **Baseline Metrics**: Establishes performance baseline
- **Model Persistence**: Saves to `models/linear_regression_trip_duration.joblib`
- **Use Case**: Quick experimentation, interpretable coefficients

### Notebook vs Production Code Comparison

| Aspect | Notebooks | Production (src/) |
|--------|-----------|-------------------|
| **Purpose** | Exploration & experimentation | Reproducible pipelines & serving |
| **Data Versioning** | Manual or standalone DVC | Integrated with src/data/load.py |
| **Configuration** | Hardcoded paths & params | `configs/config.yaml` + `src/config.py` |
| **Logging** | Console prints | MLflow + console |
| **Error Handling** | Basic | Comprehensive validation |
| **Testing** | Manual | Pytest + test suite |
| **Reproducibility** | Dependent on user input | Deterministic (RANDOM_STATE) |
| **Deployment** | Not suitable | Production-ready |
| **Feature Engineering** | Exploratory | Optimized, chunked processing |
| **Model Persistence** | joblib (standalone) | joblib + MLflow artifacts |
| **Scalability** | Limited by memory | Handles 1.4M+ rows |

**When to use Notebooks:**
- Rapid prototyping and idea testing
- EDA and data exploration
- Trying new algorithms quickly
- One-off analysis tasks

**When to use Production Code:**
- Training production models
- Running scheduled pipelines
- Serving real-time predictions
- Reproducible experiments
- Team collaboration

## Testing

Run all tests using pytest:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src
```

Run specific test file:
```bash
pytest tests/test_api_app.py -v
```

**Test Coverage:**
- API endpoint tests
- Model training and inference tests
- Data loading and preprocessing tests
- Feature engineering tests
- EDA and quality checks tests

## MLflow Integration

Model training is automatically tracked in MLflow. Experiments can be viewed using the MLflow UI.

### Start MLflow UI
```bash
mlflow ui
```

Then open http://127.0.0.1:5000 in your browser.

**Tracked Metrics:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Training time
- Model parameters

## Getting Started (Quick Start)

## Getting Started (Quick Start)

1. Clone the repository
2. Create virtual environment: `python -m venv .venv` & activate it
3. Install dependencies: `pip install -r requirements.txt`
4. Download data: `dvc pull` (authenticate with your Google account)
5. Start API: `uvicorn src.api.app:app --reload`
6. Open UI: http://127.0.0.1:8000/

For detailed instructions, see [Installation](#installation) section above.

## Contributing

### Workflow for Team Members
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make changes and commit: `git commit -m "Description of changes"`
3. Push to remote: `git push origin feature/your-feature-name`
4. Submit a pull request for review

### Code Standards
- Follow PEP 8 style guidelines
- Write tests for new features
- Ensure all tests pass: `pytest`
- Update documentation as needed
- Commit DVC changes along with code changes

### Data Changes
- Always use DVC for data modifications: `dvc add <data-file>` then `git add <file>.dvc`
- Push changes: `dvc push`
- Never commit large data files directly to git

## Troubleshooting

### Issue: `dvc pull` fails with authentication error
**Solution:**
- Ensure you've signed in with the correct Google account
- Run: `dvc pull` again and complete the OAuth flow
- Verify the shared Google Drive folder has been shared with your account
- Check `.dvc/config` to ensure the remote path is correct

### Issue: ModuleNotFoundError when running scripts
**Solution:**
```bash
# Ensure virtual environment is activated
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Verify requirements are installed
pip install -r requirements.txt
```

### Issue: API won't start (port already in use)
**Solution:**
```bash
# Use a different port
uvicorn src.api.app:app --reload --port 8001

# Or kill existing process on port 8000
# Windows: netstat -ano | findstr :8000
# macOS/Linux: lsof -i :8000
```

### Issue: Models not found during inference
**Solution:**
- Run model training scripts first: `python -m src.models.train_gradient_boosting`
- Verify model files exist in `models/` directory
- Check `configs/config.yaml` for correct model paths

### Issue: Tests fail with import errors
**Solution:**
```bash
# Ensure you're in project root directory
cd BITS_PAIML_MLE_G28

# Run pytest from project root
pytest

# Or with verbose output
pytest -v
```

### Issue: DVC remote not configured
**Solution:**
```bash
# Check current remote
dvc remote list

# Configure Google Drive remote (if needed)
dvc remote add -d cloud gdrive://<folder-id>

# Verify connection
dvc status
```

## License & Authors

**Project Team:** BITS PILANI AI/ML Group 28  
**Institution:** BITS Pilani  
**Course:** Machine Learning Engineering  
**Project Type:** Mini Project  

## Support & Documentation

- Detailed documentation: See `docs/Project Documentation_v0_G28.docx`
- EDA outputs: Check `docs/EDA_chart_outputs/`
- Feature engineering charts: Check `docs/feature_engg_chart_outputs/`
- Experiment tracking: Run `mlflow ui` to view all experiments

## Data Download

The dataset is stored on Google Drive and can be accessed using DVC (recommended) or downloaded directly.

### Google Drive Link
**Direct Download:** [NYC Taxi Data on Google Drive](https://drive.google.com/drive/folders/1C9CN2xgSkCYEk_8p-0IWYyDmncjTAjmH?usp=sharing)

> **Note:** You may need to request access to the folder. Contact the project owner if you don't have access.

## DVC Data Setup for Team Members

The data files are stored in the shared Google Drive DVC remote configured in
`.dvc/config`; they are not downloaded by `git clone` or `git pull`.

### Recommended Method: Using DVC (Automatic)

**Steps:**
1. Install the dependencies from `requirements.txt`.
2. Ensure the remote owner has shared the Google Drive folder with your Google account used for DVC.
3. From the repository root, run:

	```bash
	dvc pull
	```

4. Complete the Google OAuth sign-in in the browser when prompted. Use the
	account that has access to the shared folder.

If `dvc pull` reports that the remote cannot be accessed, verify the signed-in
Google account has access to the Drive folder and run `dvc pull` again. Do not
commit `.dvc/config.local` or OAuth credential files.

### Alternative Method: Manual Download

If you prefer to download the data manually:

1. Access the Google Drive folder: [NYC Taxi Data](https://drive.google.com/drive/folders/1C9CN2xgSkCYEk_8p-0IWYyDmncjTAjmH?usp=sharing)
2. Download the required data files
3. Extract files to `data/data_folder/` directory structure:
   ```
   data/
   └── data_folder/
       ├── train/
       │   ├── raw/
       │   │   └── train.zip
       │   └── processed/
       │       ├── train_eda_processed.zip
       │       └── train_cleaned.zip
       └── test/
           ├── raw/
           │   └── test.csv
           └── processed/
               ├── test_eda_processed.csv
               └── test_cleaned.csv
   ```
4. No need to run `dvc pull` if files are already present locally

**Advantages of DVC method:**
- Automatic version control and tracking
- Ensures data consistency across team
- Easy updates when data is refreshed
- Reproducible data pipelines
