# ML Engineering Mini-Project: Delivery / Ride ETA Prediction

This project aims to build an end-to-end Machine Learning pipeline for predicting Delivery / Ride Estimated Time of Arrival (ETA) based on various factors. It is structured to demonstrate key MLOps principles including data acquisition, feature engineering, model training, deployment, monitoring, and documentation.

**Team:** BITS PILANI AI/ML Group 28  
**Project Focus:** NYC Taxi Trip Duration Prediction  
**Models:** Linear Regression, Gradient Boosting

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Project Workflow](#project-workflow)
- [Dataset & Features](#dataset--features)
- [Configuration](#configuration)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Training Models](#training-models)
- [Running Experiments](#running-experiments)
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

## Project Workflow

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

### Train Linear Regression Model
```bash
python -m src.models.train_linear_regression
```

### Train Gradient Boosting Model
```bash
python -m src.models.train_gradient_boosting
```

Models are saved to the `models/` directory and tracked in MLflow.

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

## DVC Data Setup for Team Members

The data files are stored in the shared Google Drive DVC remote configured in
`.dvc/config`; they are not downloaded by `git clone` or `git pull`.

### Steps:
1. Install the dependencies from `requirements.txt`.
2. Ask the remote owner to share the configured Google Drive folder with the
	Google account you will use for DVC.
3. From the repository root, run:

	```bash
	dvc pull
	```

4. Complete the Google OAuth sign-in in the browser when prompted. Use the
	account that has access to the shared folder.

If `dvc pull` reports that the remote cannot be accessed, verify the signed-in
Google account has access to the Drive folder and run `dvc pull` again. Do not
commit `.dvc/config.local` or OAuth credential files.
