# Script Explanations - NYC Taxi Trip Duration Prediction

This document provides detailed explanations of all the exploratory and preprocessing scripts in the `notebooks/` directory.

---

## 1. **EDA.py** - Exploratory Data Analysis

### Purpose
Comprehensive exploratory data analysis (EDA) to understand the structure, distribution, and relationships in the NYC taxi dataset.

### Key Components

#### A. **Data Loading**
```python
train_df = pd.read_csv("data/data_folder/train/raw/train.zip")
test_df = pd.read_csv("data/data_folder/test/raw/test.csv")
```
- Loads training and test datasets
- Handles zip files containing CSV data

#### B. **Initial Data Inspection**
- **DataFrame Info**: Prints data types and non-null counts
- **Descriptive Statistics**: Basic statistics (mean, std, min, max, quartiles)
- **Missing Values Analysis**: Identifies columns with null values
- **Value Counts**: Frequency of categorical values

#### C. **Datetime Conversion**
```python
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])
```
- Converts string datetime columns to proper datetime objects
- Enables time-based feature extraction

#### D. **Univariate Analysis**

**Passenger Count Distribution:**
- Count plot showing frequency of trips by passenger count
- Box plot to identify outliers
- Value counts for quick reference

**Categorical Features:**
- Vendor ID distribution (which taxi company)
- Store and Forward Flag (indicates connection issues)

**Haversine Distance Calculation:**
- Computes geographic distance between pickup and dropoff using GPS coordinates
- Uses the Haversine formula (accounts for Earth's spherical shape)
- Formula: `distance = 2 * R * arcsin(sqrt(sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)))`
- Result: Distance in kilometers

#### E. **Bivariate Analysis**

**Haversine Distance vs. Trip Duration:**
- Scatter plot showing correlation between distance and duration
- Uses log scale to better visualize wide ranges
- Shows trip duration increases with distance

#### F. **Temporal Analysis**

Extracts time-based features:
- **pickup_hour**: Hour of day (0-23)
- **pickup_day_of_week**: Day of week (0=Monday, 6=Sunday)
- **pickup_month**: Month of year
- **pickup_day**: Day of month
- **pickup_quarter**: Quarter (Q1-Q4)

**Temporal Distributions:**
- Pickups by hour of day
- Pickups by day of week
- Pickups by month

**Temporal vs. Trip Duration:**
- Median trip duration by hour
- Median trip duration by day of week
- Median trip duration by month

#### G. **Spatial Analysis**

**Interactive Map Visualization:**
- Creates an interactive Folium map centered on NYC
- Blue markers: Pickup locations (sample of 1000 to avoid overcrowding)
- Red markers: Dropoff locations
- Saved as HTML file for interactive exploration
- Helps identify geographic hotspots and traffic patterns

### Output
- Multiple PNG files with visualizations saved to `docs/EDA_chart_outputs/`
- Interactive HTML map showing pickup/dropoff locations
- Console output with detailed statistics

### When to Use
- Initial project setup for data understanding
- Sharing findings with team members
- Identifying data quality issues or outliers

---

## 2. **featureengineering.py** - Advanced Feature Engineering

### Purpose
Create sophisticated features for model training by extracting temporal, spatial, and interaction features from raw data.

### Key Functions

#### A. **extract_time_features(dataframe)**
Extracts comprehensive time-based features:

```python
For each datetime column (pickup_datetime, dropoff_datetime):
  - hour: Hour of the day (0-23)
  - day_of_week: Day of week (0-6)
  - month: Month (1-12)
  - season: Winter/Spring/Summer/Autumn
  - day_of_year: Julian day (1-366)
  - week_of_year: ISO week number (1-52)
  - is_holiday: US Federal Holiday (1=holiday, 0=regular day)
  - part_of_day: night/morning/afternoon/evening
```

**Holiday Detection:**
- Uses `USFederalHolidayCalendar` to identify federal holidays
- Helps capture special traffic patterns on holidays

#### B. **extract_location_features(dataframe)**
Creates spatial features from GPS coordinates:

**Distance Calculations:**
- **Manhattan Distance**: Sum of absolute differences in latitude and longitude
  - Formula: `|lat2 - lat1| + |lon2 - lon1|`
  - Fast approximate distance measure
  
- **Haversine Distance**: Great-circle distance on Earth
  - More accurate than Manhattan distance
  - Accounts for Earth's curvature

**Direction/Bearing:**
- Calculates compass bearing from pickup to dropoff
- Range: 0-360 degrees
- Useful for understanding trip direction patterns

**Zone Features:**
- **Grid Zones**: Creates 0.01-degree grid cells from coordinates
  - Format: `lat_lon` (e.g., "40.75_-73.98")
  - Enables zone-based analysis
  
- **Route Zones**: Combination of pickup and dropoff zones
  - Format: `pickup_zone_to_dropoff_zone`
  - Identifies common routes
  
- **High Traffic Zones**: Flags frequently used zones
  - Identifies major business districts, airports, etc.
  - Based on 90th percentile of zone frequency

#### C. **extract_trip_features(dataframe)**
Normalizes existing trip attributes:

```python
- passenger_count: Converts to numeric
- store_and_fwd_flag: Standardizes format (N/Y to uppercase)
```

#### D. **extract_interaction_features(dataframe)**
Creates feature interactions (combinations):

```python
- distance_hour_interaction: distance × pickup_hour
  - Captures time-dependent distance effects
  - Example: Rush hour trips might be shorter but take longer
  
- day_of_week_holiday_interaction: day_of_week × is_holiday
  - Identifies special day patterns
```

#### E. **prepare_model_features(dataframe)**
Cleans features for model input:

**Data Validation:**
- Clips passenger count to 1-6 (removes invalid values)
- Clips geographic coordinates to NYC bounds:
  - Latitude: 40.5° to 41.0°
  - Longitude: -74.25° to -73.7°
- Clips distances to realistic ranges

**Categorical Encoding:**
- **store_and_fwd_flag**: N→0, Y→1
- **season & part_of_day**: Fills missing with "unknown"

**Missing Value Handling:**
- Numeric columns: Filled with median
- Categorical columns: Filled with "unknown"

**Important:** Removes all dropoff features before model training:
- Reason: Dropoff data isn't available at prediction time
- This ensures training data matches inference data

#### F. **integrate_external_features(dataframe, external_data)**
Merges optional external data (weather, traffic):

**Supported External Features:**
- temperature, precipitation, wind_speed
- weather_type, average_speed, traffic_congestion

**Merge Strategy:**
- Joins on available keys (datetime, zones)
- Keeps original data structure
- Prevents target leakage (removes trip_duration from external data)

#### G. **find_high_traffic_zones(dataframe, quantile=0.9)**
Identifies busy zones from training data:

```python
1. Extracts location features from dataframe
2. Counts frequency of pickup and dropoff zones
3. Returns zones in top 10% (90th percentile)
4. Used to flag high-traffic routes
```

**Chunk-based Version:**
- `find_high_traffic_zones_from_csv()`: Process large files in chunks
- Prevents loading entire dataset into memory
- Used for datasets too large to fit in RAM

### Output
- Enhanced dataframe with additional features
- Maintains data integrity (no data leakage)
- Ready for model training

### When to Use
- Preprocessing train/test data before model training
- Creating features for ensemble models
- Improving model predictive power

---

## 3. **pre-EDA-dvc.py** - DVC Initialization for Raw Data

### Purpose
Add raw data files to DVC tracking for version control and team collaboration.

### Key Steps

#### A. **Directory Setup**
```python
os.makedirs("data/data_folder/train/raw", exist_ok=True)
os.makedirs("data/data_folder/test/raw", exist_ok=True)
```
- Creates standard directory structure
- Ensures consistent paths across team

#### B. **Copy Raw Files**
```python
shutil.copy2(raw_train_path, "data/data_folder/train/raw/train.zip")
shutil.copy2(raw_test_path, "data/data_folder/test/raw/test.csv")
```
- Copies files to DVC-tracked directories
- Preserves metadata (timestamps, permissions)
- Enables version control

#### C. **DVC Initialization**
```python
if not os.path.isdir(".dvc"):
    subprocess.run([dvc_executable, "dvc", "init"], check=True)
```
- Initializes DVC in repository (if not already done)
- Creates `.dvc/` configuration directory

#### D. **Add Files to DVC**
```python
dvc add data/data_folder/train/raw/train.zip
dvc add data/data_folder/test/raw/test.csv
```
- Tracks files with DVC
- Creates `.dvc` metadata files (train.zip.dvc, test.csv.dvc)
- Files listed in `.gitignore` to prevent large files in Git

#### E. **Status Verification**
```python
dvc status data/data_folder/train/raw/train.zip
```
- Checks if files are properly tracked
- Verifies no corruption or changes

#### F. **Git Integration**
The script includes git commands (as comments):
```bash
git add data/data_folder/train/raw/train.zip.dvc
git commit -m "Track raw data with DVC"
git push origin main
```

### Data Flow
```
Raw Data Files
    ↓
Copy to DVC Directory
    ↓
Run `dvc add`
    ↓
Creates .dvc metadata files
    ↓
Commit .dvc files to Git
    ↓
Push to remote (Google Drive)
    ↓
Team members run `dvc pull`
```

### When to Use
- First time setting up project with DVC
- After downloading new data files
- When team members need to sync data

### Benefits
- **Version Control**: Track data changes like code
- **Storage Efficiency**: Store large files on cloud, not Git
- **Team Collaboration**: Share data access securely
- **Reproducibility**: Same data version for all team members

---

## 4. **ExtractData_Using_DVC.py** - Load DVC-Tracked Data

### Purpose
Load data from DVC remote storage and prepare it for analysis/modeling.

### Key Components

#### A. **Path Configuration**
```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ZIP = PROJECT_ROOT / "data/data_folder/train/processed/train_eda_processed.zip"
PROCESSED_TEST_CSV = PROJECT_ROOT / "data/data_folder/test/processed/test_eda_processed.csv"
```
- Uses relative paths (works on any machine)
- Targets processed data (already cleaned)

#### B. **load_processed_data(data_path)**
Intelligent data loading function:

**Logic:**
1. Check if data file exists locally
2. If not, check for `.dvc` metadata file
3. If metadata exists, run `dvc pull` to download from remote
4. Load data from CSV or ZIP

**ZIP Handling:**
```python
with zipfile.ZipFile(data_path) as archive:
    csv_files = [name for name in archive.namelist() if name.endswith(".csv")]
    with archive.open(csv_files[0]) as csv_file:
        return pd.read_csv(csv_file)
```
- Automatically extracts CSV from ZIP
- Validates that ZIP contains exactly one CSV
- Reads into DataFrame without extracting to disk

#### C. **Data Loading**
```python
train_df = load_processed_data(PROCESSED_ZIP)
test_df = load_processed_data(PROCESSED_TEST_CSV)
```
- Loads both training and test datasets
- Prints row and column counts
- Automatically pulls from DVC if needed

### Output
- `train_df`: DataFrame with ~1.46M rows and processed features
- `test_df`: DataFrame with ~625K rows
- Ready for model training or further analysis

### When to Use
- Loading processed data for model training
- When DVC remote is configured
- Team collaboration (auto-syncs from cloud)

### Advantages
- Automatic DVC synchronization
- No manual downloads needed
- Handles both ZIP and CSV formats
- Relative paths work across systems

---

## 5. **FetchingData.py** - Kaggle Data Download (Template)

### Purpose
Download NYC taxi dataset from Kaggle API (commented out - template for reference).

### Key Features

#### A. **Project Setup**
```python
project_root = r"C:\Users\Kiran\BITS PILANI AI_ML\MiniProject\BITS_PAIML_MLE_G28"
os.chdir(project_root)
os.makedirs('data_folder', exist_ok=True)
```
- Sets working directory to project root
- Creates data folder structure

#### B. **Kaggle API Authentication**
```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
```
- Requires Kaggle API credentials (`~/.kaggle/kaggle.json`)
- Downloads credentials locally from Kaggle website

#### C. **Download Methods** (Commented - for reference)

**For Kaggle Datasets:**
```python
api.dataset_download_files("owner/dataset-name", path=data_folder, unzip=True)
```

**For Kaggle Competitions:**
```python
api.competition_download_files("competition-name", path=data_folder, unzip=True)
```

### Current Status
- **Not actively used** (all code is commented out)
- **Why?** Data already available via Google Drive
- **Purpose**: Reference template for future Kaggle downloads

### When to Use
- If dataset needs to be refreshed from Kaggle
- For other Kaggle-based projects
- As template for downloading competition data

### Prerequisites for Activation
1. Install Kaggle API:
   ```bash
   pip install kaggle
   ```
2. Download credentials from https://www.kaggle.com/settings/account
3. Save to `~/.kaggle/kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

---

---

## 6. **gradientBoosting.py** - Gradient Boosting Model (Notebook Version)

### Purpose
Train and save a gradient boosting regressor for predicting NYC taxi trip duration using iterative training with progress logging.

### Model Architecture

**Gradient Boosting Regressor:**
- **Algorithm**: Ensemble of decision trees trained sequentially
- **Mechanism**: Each tree corrects errors made by previous trees
- **Key Parameters**:
  ```python
  n_estimators=300           # 300 decision trees
  learning_rate=0.05         # Controls contribution of each tree (0.05 = 5%)
  max_depth=4                # Maximum depth of each tree
  loss='huber'               # Huber loss (robust to outliers)
  random_state=42            # For reproducibility
  warm_start=True            # Enable incremental training
  ```

### Data Pipeline

#### A. **Data Loading & Preparation**
```python
data = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)
```
- Loads training data
- Replaces infinite values with NaN for proper handling

#### B. **Target & Feature Separation**
```python
y = pd.to_numeric(data.pop(TARGET), errors="coerce")  # Trip duration target
X = data.drop(columns=["id", "pickup_datetime", "dropoff_datetime"], errors="ignore")
```
- Removes non-predictive columns
- Extracts target variable `trip_duration`
- Validates target values (non-null, numeric)

#### C. **Feature Type Detection**
```python
numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = X.select_dtypes(exclude=np.number).columns.tolist()
```
- Identifies numeric vs categorical features
- Enables appropriate preprocessing for each type

#### D. **Preprocessing Pipeline**

**Numeric Features:**
```
Numeric Data
    ↓
SimpleImputer (strategy='median')  → Fill missing values with median
    ↓
StandardScaler                     → Normalize to mean=0, std=1
```

**Categorical Features:**
```
Categorical Data
    ↓
SimpleImputer (strategy='most_frequent')  → Fill with most common value
    ↓
OneHotEncoder (handle_unknown='ignore')   → Convert categories to binary columns
```

**Combined Pipeline:**
```python
model = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])),
    ("regressor", GradientBoostingRegressor(...))
])
```

### Training Process

**Incremental Training with Progress Logging:**
```python
for stage in range(progress_every, final_stage + 1, progress_every):
    model.named_steps["regressor"].set_params(n_estimators=stage)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # Log metrics every 25 trees
```

- Trains in increments (default: every 25 trees)
- Logs MAE, RMSE, R² at each stage
- Shows convergence behavior
- `warm_start=True` allows this incremental approach

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **MAE** | Average of \|predicted - actual\| | Average error in seconds |
| **RMSE** | √(mean of squared errors) | Penalizes large errors more |
| **R²** | 1 - (SS_res / SS_tot) | % variance explained (0-1) |

### Model Persistence
```python
joblib.dump(model, MODEL_PATH)
```
- Saves entire pipeline (preprocessing + model)
- Can be loaded later for inference
- Binary format for efficient storage

### When to Use
- Baseline notebook experimentation
- Understanding model behavior through incremental logging
- Direct data loading (hardcoded paths)

### Limitations
- Hardcoded file paths (not flexible across systems)
- No MLflow integration (no experiment tracking)
- No hyperparameter tuning
- Not production-ready

---

## 7. **linearRegressionmodel.py** - Linear Regression Model (Notebook Version)

### Purpose
Train a simple linear regression baseline for trip duration prediction.

### Model Algorithm

**Linear Regression:**
- **Equation**: `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`
- **Goal**: Find optimal coefficients (β) minimizing error
- **Advantages**: 
  - Simple, interpretable
  - Fast to train
  - Good baseline for comparison
- **Limitations**:
  - Assumes linear relationships
  - Sensitive to outliers

### Data Preprocessing

**Numeric Features:**
```python
Pipeline([("imputer", SimpleImputer(strategy="median"))])
```
- Fills missing values with median
- No scaling (Linear Regression is scale-invariant for coefficients)

**Categorical Features:**
```python
Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
```
- Imputes with most frequent category
- Converts to one-hot encoding (binary columns)

### Model Architecture
```python
Pipeline([
    ("preprocessor", ColumnTransformer(...)),
    ("regressor", LinearRegression())
])
```

### Training & Evaluation

**Single-Stage Training:**
```python
model.fit(X_train, y_train)  # Fit once with all data
predictions = model.predict(X_test)
```

**Metrics Reported:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

### Output
```
Training metrics:
MAE (seconds): XX.XX
RMSE (seconds): XX.XX
R2 score: 0.XXXX
```

### When to Use
- Initial baseline establishment
- Identifying which features are predictive
- Comparing with more complex models
- Interpretability focus

### Comparison with Gradient Boosting
| Aspect | Linear Regression | Gradient Boosting |
|--------|-------------------|-------------------|
| Complexity | Simple | Complex |
| Training Speed | Very Fast | Slower |
| Non-linearity | No | Yes |
| Interpretability | High | Low |
| Typical R² | 0.50-0.70 | 0.70-0.85+ |

---

## 8. **src/models/train_gradient_boosting.py** - Gradient Boosting (Production Version)

### Purpose
Production-ready gradient boosting model with MLflow integration, modular code, and configuration management.

### Key Improvements over Notebook Version

| Feature | Notebook | Production |
|---------|----------|-----------|
| **Config Management** | Hardcoded paths/params | `src.config.py` |
| **Preprocessing** | Inline code | `src.models.common.build_preprocessor()` |
| **Data Loading** | Manual `pd.read_csv()` | `src.data.load.load_cleaned_train_test()` |
| **Metrics** | Manual calculation | `src.models.common.evaluate_regression()` |
| **MLflow Integration** | None | Full tracking |
| **Model Saving** | `joblib.dump()` | `src.models.common.save_model()` |
| **Modularity** | Monolithic | Separated concerns |

### Configuration-Driven Training

**From src/config.py:**
```python
GRADIENT_BOOSTING = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "loss": "huber",
    "model_filename": "gradient_boosting_trip_duration.joblib",
    "progress_every": 25,
}

MODELS_DIR = Path("models/")
TRAIN_CLEANED_PATH = Path("data/data_folder/train/processed/train_cleaned.zip")
```

**Benefits:**
- Change hyperparameters without touching code
- Consistent across team members
- Easy experimentation (adjust config, rerun)

### Enhanced Data Loading

```python
train_df, _ = load_cleaned_train_test()
```
- Uses DVC-managed processed data
- Automatically pulls from remote if needed
- Handles ZIP files transparently

### Preprocessing with Helper Functions

```python
preprocessor = build_preprocessor(X, scale_numeric=True)
```
- Centralized preprocessing logic
- Reusable across models
- Consistent feature handling

### MLflow Integration

**Automatic Experiment Tracking:**
```python
with start_run_if_enabled("gradient_boosting", log_to_mlflow, tracking_uri):
    # Log hyperparameters
    mlflow.log_params({
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "loss": "huber",
        ...
    })
    
    # Log training progress
    for stage in range(...):
        mlflow.log_metrics(metrics, step=stage)
    
    # Log final model
    mlflow.sklearn.log_model(model, "model", ...)
```

**What Gets Tracked:**
- **Parameters**: All hyperparameters
- **Metrics**: MAE, RMSE, R² at each stage
- **Dataset Tags**: Data version and path
- **Artifacts**: Trained model pickle
- **Run Info**: Start time, duration, status

**View Results:**
```bash
mlflow ui
# Open http://localhost:5000 in browser
```

### Progressive Training with Metrics Logging

```python
print(f"Training progress: {stage}/{final_stage} trees | {format_metrics(metrics)}")
if run is not None:
    mlflow.log_metrics(metrics, step=stage)
```

**Example Output:**
```
Training progress: 25/300 trees | MAE=200.5 | RMSE=350.2 | R2=0.652
Training progress: 50/300 trees | MAE=185.3 | RMSE=320.1 | R2=0.701
Training progress: 75/300 trees | MAE=175.8 | RMSE=305.4 | R2=0.725
...
```

### Function Signature
```python
def train_model(
    train_df: pd.DataFrame | None = None,
    model_path: Path | None = None,
    log_to_mlflow: bool = True,
    tracking_uri: str | None = None,
) -> Pipeline:
```

**Parameters:**
- `train_df`: Override data loading with custom dataframe
- `model_path`: Override default save location
- `log_to_mlflow`: Enable/disable experiment tracking
- `tracking_uri`: MLflow server location

### When to Use
- Production model training
- Experiment tracking and comparison
- Hyperparameter tuning (with config changes)
- Model versioning and management
- Team collaboration

### Execution
```bash
# From project root
python -m src.models.train_gradient_boosting
```

---

## 9. **src/models/train_linear_regression.py** - Linear Regression (Production Version)

### Purpose
Production-ready linear regression baseline with MLflow tracking and configuration management.

### Configuration-Driven Approach

**From src/config.py:**
```python
LINEAR_REGRESSION = {
    "model_filename": "linear_regression_trip_duration.joblib",
}
```

### Preprocessing Differences

**Unique Feature:**
```python
preprocessor = build_preprocessor(X, scale_numeric=False)
```

- **No scaling** for numeric features
- **Reason**: Linear regression doesn't require scaled features
- **Gradient boosting**: Requires scaling for better performance

### Simplified Training

**Single-Stage Fit:**
```python
model.fit(X_train, y_train)
```

- No incremental training (unlike gradient boosting)
- Linear regression trains in one pass
- Faster than gradient boosting

### MLflow Tracking

**Simpler Logging:**
```python
mlflow.log_params({"test_size": TEST_SIZE, "random_state": RANDOM_STATE})
mlflow.log_metrics(metrics)
mlflow.sklearn.log_model(model, "model", ...)
```

**Example MLflow Experiment Entry:**
```
Experiment: linear_regression
Parameters:
  - test_size: 0.2
  - random_state: 42
Metrics:
  - MAE: 185.32 seconds
  - RMSE: 320.15 seconds
  - R2: 0.7023
```

### Model Persistence

```python
save_model(model, model_path)
```

- Uses centralized save function
- Consistent with gradient boosting
- Enables comparison and deployment

### Function Signature
```python
def train_model(
    train_df: pd.DataFrame | None = None,
    model_path: Path | None = None,
    log_to_mlflow: bool = True,
    tracking_uri: str | None = None,
) -> Pipeline:
```

Same flexibility as gradient boosting version.

### When to Use
- Baseline model establishment
- Interpretability requirements
- Quick model training for comparisons
- Starting point for hyperparameter tuning

### Execution
```bash
python -m src.models.train_linear_regression
```

---

## Model Comparison Matrix

| Aspect | Linear Regression | Gradient Boosting |
|--------|-------------------|-------------------|
| **File (Notebook)** | `linearRegressionmodel.py` | `gradientBoosting.py` |
| **File (Production)** | `src/models/train_linear_regression.py` | `src/models/train_gradient_boosting.py` |
| **Training Time** | ~30 seconds | ~5-10 minutes |
| **Typical R² Score** | 0.55-0.70 | 0.75-0.85 |
| **MAE (seconds)** | ~200-220 | ~150-180 |
| **Complexity** | Simple | Complex |
| **Interpretability** | High | Low |
| **Hyperparameters** | Minimal | Many |
| **Use Case** | Baseline | Production |
| **MLflow Support** | Yes (prod) | Yes (prod) |

---

## Production vs. Notebook Scripts

### Key Differences

**1. Import Organization**
- Notebook: Direct sklearn imports
- Production: Uses modular src packages

**2. Data Loading**
- Notebook: Hardcoded `pd.read_csv()`
- Production: `src.data.load.load_cleaned_train_test()`

**3. Configuration**
- Notebook: Inline parameters
- Production: `src.config` module

**4. Preprocessing**
- Notebook: Inline pipelines
- Production: `src.models.common.build_preprocessor()`

**5. Evaluation**
- Notebook: Manual metrics calculation
- Production: `src.models.common.evaluate_regression()`

**6. Experiment Tracking**
- Notebook: No tracking
- Production: Full MLflow integration

**7. Error Handling**
- Notebook: Basic validation
- Production: Comprehensive error messages

### Why Two Versions?

| Aspect | Notebook | Production |
|--------|----------|-----------|
| Purpose | Experimentation | Deployment |
| Audience | Individual developer | Team/DevOps |
| Flexibility | High | Configured |
| Reproducibility | Variable | Guaranteed |
| Maintenance | Simple | Structured |

---

## Complete Training Workflow

```
1. Data Preparation
   ├─ featureengineering.py
   └─ create train_cleaned.csv

2. Model Training (Production)
   ├─ train_linear_regression.py
   │  ├─ Load cleaned data
   │  ├─ Preprocess
   │  ├─ Train (single pass)
   │  ├─ Evaluate
   │  ├─ Log to MLflow
   │  └─ Save model
   │
   └─ train_gradient_boosting.py
      ├─ Load cleaned data
      ├─ Preprocess (with scaling)
      ├─ Train (300 trees, log progress)
      ├─ Evaluate at each stage
      ├─ Log to MLflow
      └─ Save model

3. Experiment Comparison
   └─ MLflow UI (http://localhost:5000)

4. Model Deployment
   ├─ Select best model
   ├─ API server loads model
   └─ Inference ready
```

---

## Hyperparameter Tuning Strategy

### Current Settings
```yaml
Gradient Boosting:
  n_estimators: 300
  learning_rate: 0.05
  max_depth: 4
  loss: huber

Linear Regression:
  No tunable hyperparameters
```

### To Experiment
1. Edit `configs/config.yaml`
2. Rerun training script
3. Compare results in MLflow UI

### Typical Adjustments
```python
# For more complex patterns
n_estimators: 500          # More trees
max_depth: 6               # Deeper trees

# For faster training
n_estimators: 100          # Fewer trees
learning_rate: 0.1         # Larger steps
```

---

## Script Execution Order (Updated)

### Recommended Workflow:
```
1. pre-EDA-dvc.py              → Initialize DVC tracking for raw data
2. EDA.py                      → Explore data, understand patterns
3. featureengineering.py       → Create advanced features
4. ExtractData_Using_DVC.py    → Load processed data for modeling
5. train_linear_regression.py  → Baseline model (production version)
6. train_gradient_boosting.py  → Advanced model (production version)
```

### Alternative (Notebook Experimentation):
```
1. EDA.py                      → Initial exploration
2. featureengineering.py       → Feature creation
3. linearRegressionmodel.py    → Baseline model
4. gradientBoosting.py         → Advanced model
```

---

## Key Concepts Summary

| Script | Type | Focus | Input | Output |
|--------|------|-------|-------|--------|
| EDA.py | Notebook | Data Understanding | Raw data | Visualizations & Statistics |
| featureengineering.py | Notebook | Feature Creation | Raw/Processed data | Enhanced dataframe |
| pre-EDA-dvc.py | Notebook | Version Control | Raw files | DVC metadata files |
| ExtractData_Using_DVC.py | Notebook | Data Loading | DVC metadata | Processed dataframes |
| linearRegressionmodel.py | Notebook | Baseline Model | Processed data | Linear model + metrics |
| gradientBoosting.py | Notebook | Advanced Model | Processed data | GB model + metrics |
| train_linear_regression.py | Production | Baseline Model | Cleaned data | Model + MLflow logs |
| train_gradient_boosting.py | Production | Advanced Model | Cleaned data | Model + MLflow logs |

---

## Common Issues & Solutions

### Issue: DVC pull fails
**Solution:** Check Google Drive access, run `dvc status`, ensure credentials are set

### Issue: Memory error with large datasets
**Solution:** Use chunked processing or `find_high_traffic_zones_from_csv()`

### Issue: Missing datetime features
**Solution:** Ensure datetime columns are converted with `pd.to_datetime()`

### Issue: Coordinate values out of range
**Solution:** Scripts automatically clip to NYC bounds (40.5-41.0°N, 74.25-73.7°W)

### Issue: Model training fails - hardcoded paths
**Solution:** Use production scripts (`src/models/`) which use config-based paths

### Issue: Can't see MLflow metrics
**Solution:** Run `mlflow ui` from project root, ensure `log_to_mlflow=True`

---

---

## 10. **src/config.py** - Centralized Configuration

### Purpose
Central configuration module that loads `configs/config.yaml` and exposes all paths and settings as absolute Path objects.

### Key Features

**Configuration Loading:**
```python
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
_config = yaml.safe_load(CONFIG_PATH)
```

**Exported Paths (Absolute):**
```python
# Data paths
TRAIN_RAW_PATH              # data/data_folder/train/raw/train.zip
TEST_RAW_PATH               # data/data_folder/test/raw/test.csv
TRAIN_PROCESSED_PATH        # EDA-processed data
TEST_PROCESSED_PATH
TRAIN_CLEANED_PATH          # Feature-engineered data
TEST_CLEANED_PATH
EDA_OUTPUT_DIR              # docs/EDA_chart_outputs/
FEATURE_ENGINEERING_OUTPUT_DIR  # docs/feature_engg_chart_outputs/
MODELS_DIR                  # models/
EXTERNAL_DATA_DIR           # data/data_folder/external/
```

**Training Parameters:**
```python
TEST_SIZE = 0.2            # 80/20 train/test split
RANDOM_STATE = 42          # Reproducibility
LINEAR_REGRESSION = {...}  # Model config
GRADIENT_BOOSTING = {...}  # Model hyperparameters
```

**Serving Configuration:**
```python
DEFAULT_SERVING_MODEL = "gradient_boosting"
SERVING_MODELS = {
    "linear_regression": Path("models/linear_regression_trip_duration.joblib"),
    "gradient_boosting": Path("models/gradient_boosting_trip_duration.joblib"),
}
```

**MLflow Settings:**
```python
MLFLOW_TRACKING_URI = "mlruns/"
MLFLOW_EXPERIMENT_NAME = "trip-duration-prediction"
```

### Benefits
- **Single Source of Truth**: All paths and configs in one place
- **Portability**: Works regardless of working directory
- **Consistency**: Ensures all modules use same settings
- **Testability**: Easy to mock/override in tests

### Usage Example
```python
from src.config import TRAIN_CLEANED_PATH, MODELS_DIR
df = pd.read_csv(TRAIN_CLEANED_PATH)
model_path = MODELS_DIR / "my_model.joblib"
```

---

## 11. **src/main.py** - Entry Point

### Purpose
Simple placeholder entry point for the project.

### Current Content
```python
def hello_world():
    return "Hello from src/main.py!"

if __name__ == "__main__":
    print(hello_world())
```

### Use Case
- Can be extended for pipeline orchestration
- Placeholder for future main execution flow

---

## 12. **src/api/schemas.py** - Pydantic Request/Response Models

### Purpose
Define request and response data structures for the FastAPI server with validation.

### Request Schema: `TripRequest`

**Fields:**
```python
vendor_id: int              # Taxi company (1 or 2)
passenger_count: int        # 0-9 passengers
pickup_datetime: datetime   # Trip start time
pickup_longitude: float     # Pickup GPS longitude
pickup_latitude: float      # Pickup GPS latitude
dropoff_longitude: float    # Dropoff GPS longitude
dropoff_latitude: float     # Dropoff GPS latitude
store_and_fwd_flag: Literal["Y", "N"]  # Connection flag
algorithm: Optional[str]    # "gradient_boosting" or "linear_regression"
```

**Example Request:**
```json
{
  "vendor_id": 1,
  "passenger_count": 1,
  "pickup_datetime": "2016-03-14T09:30:00",
  "pickup_longitude": -73.9855,
  "pickup_latitude": 40.7580,
  "dropoff_longitude": -73.9654,
  "dropoff_latitude": 40.7829,
  "store_and_fwd_flag": "N",
  "algorithm": "gradient_boosting"
}
```

### Response Schema: `PredictionResponse`

**Fields:**
```python
predicted_trip_duration_seconds: float  # Trip duration in seconds
model_used: str                         # Which model made prediction
```

**Example Response:**
```json
{
  "predicted_trip_duration_seconds": 850.25,
  "model_used": "gradient_boosting"
}
```

### Benefits
- **Type Safety**: Automatic validation and type checking
- **Documentation**: Auto-generated Swagger/OpenAPI docs
- **Error Handling**: Clear error messages for invalid input
- **Serialization**: Automatic JSON encoding/decoding

---

## 13. **src/api/inference.py** - Model Inference Logic

### Purpose
Load pre-trained models and make predictions with feature engineering.

### Key Functions

**`load_model(name: str)`**
- Lazy-loads models from joblib files
- Caches models in memory (one-time load)
- Returns cached model on subsequent calls
- Raises `KeyError` if model name not found
- Raises `FileNotFoundError` if model file missing

**`clear_model_cache()`**
- Clears all cached models from memory
- Useful for testing or freeing memory

**`build_feature_row(request: TripRequest) -> pd.DataFrame`**
- Converts request to DataFrame
- Applies feature engineering (`engineer_features()`)
- Returns one-row DataFrame with all computed features

**`predict_trip_duration(model, request: TripRequest) -> float`**
- Builds feature row from request
- Reorders features to match model's training order
- Runs prediction using scikit-learn model
- Returns predicted duration in seconds

### Feature Engineering Pipeline

```
TripRequest
    ↓
build_feature_row()
    ├─ Create raw DataFrame from request fields
    └─ Apply engineer_features() (from src.features.build_features)
        ├─ add_distance_feature() → haversine_distance
        └─ add_temporal_features() → pickup_hour, day_of_week, etc.
    ↓
Raw Feature DataFrame
    ↓
model.predict()
    ├─ Pipeline preprocessor (impute, scale, encode)
    └─ Regressor (Linear/Gradient Boosting)
    ↓
Prediction (seconds)
```

### Model Caching Benefit
```
Request 1 → Load model from disk (slow)
Request 2 → Use cached model (fast)
Request 3 → Use cached model (fast)
```

---

## 14. **src/api/app.py** - FastAPI Server

### Purpose
Production-ready REST API for trip duration predictions.

### Endpoints

**`GET /`** - Web UI
- Returns interactive HTML form
- Loads from `src/api/static/index.html`
- User-friendly interface for testing

**`GET /health`** - Health Check
```json
{"status": "ok"}
```
- Used by load balancers
- Indicates server is running

**`GET /models`** - List Available Models
```json
{
  "available_models": ["gradient_boosting", "linear_regression"],
  "default_model": "gradient_boosting"
}
```

**`POST /predict`** - Make Prediction
- Input: `TripRequest` (validated JSON)
- Output: `PredictionResponse`
- Runs feature engineering + model inference
- Error handling:
  - 400: Invalid model name
  - 503: Model file not found

### Automatic API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- Auto-generated from Pydantic schemas

### Running the Server
```bash
uvicorn src.api.app:app --reload
# Production: uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

---

## 15. **src/data/load.py** - DVC Data Loading

### Purpose
Unified data loading with automatic DVC synchronization.

### Key Functions

**`dvc_pull(data_path: Path) -> None`**
- Checks if file exists locally
- If not, runs `dvc pull` to download from remote
- Handles Google Drive synchronization automatically

**`load_dvc_tracked_csv(data_path: Path) -> pd.DataFrame`**
- Loads DVC-tracked CSV or ZIP file
- Automatically calls `dvc_pull()` if needed
- Handles ZIP decompression transparently
- Returns DataFrame

**`load_raw_train_test() -> tuple[DataFrame, DataFrame]`**
- Loads raw training and test data
- Returns: (train_df, test_df)

**`load_processed_train_test() -> tuple[DataFrame, DataFrame]`**
- Loads EDA-processed data
- Used by feature engineering scripts

**`load_cleaned_train_test() -> tuple[DataFrame, DataFrame]`**
- Loads feature-engineered cleaned data
- Used by model training scripts

**`read_dvc_md5(tracked_path: Path) -> str | None`**
- Reads data version hash from `.dvc` metadata
- Used for tagging MLflow experiments
- Enables reproducibility tracking

### Data Loading Flow

```
load_cleaned_train_test()
    ↓
For TRAIN_CLEANED_PATH:
    ├─ Check if file exists
    ├─ If no → run dvc pull (fetch from Google Drive)
    ├─ Open ZIP → extract CSV
    └─ Load into DataFrame
    ↓
Return (train_df, test_df)
```

### Benefits
- **Automatic Sync**: No manual `dvc pull` needed
- **Transparent Handling**: Zip files seamless
- **Versioning**: MD5 tracking for reproducibility
- **Scalability**: Handles large files efficiently

---

## 16. **src/data/stage_raw.py** - DVC Setup

### Purpose
Stage raw data files into DVC-tracked locations and initialize version control.

### Key Functions

**`stage_file(source: Path, destination: Path) -> None`**
- Copies file from legacy location to DVC location
- Idempotent (skips if already staged)
- Creates parent directories automatically

**`dvc_add(path: Path) -> None`**
- Runs `dvc add` command
- Creates `.dvc` metadata file
- Makes file version-controllable

**`stage_raw_data() -> None`**
- Main workflow:
  1. Initialize DVC if not already done
  2. Copy train.zip to `data/data_folder/train/raw/`
  3. Copy test.csv to `data/data_folder/test/raw/`
  4. Run `dvc add` on both files
  5. Print next steps for git commit

### Execution
```bash
python -m src.data.stage_raw
```

### Output
```
Copied ... -> data/data_folder/train/raw/train.zip
Copied ... -> data/data_folder/test/raw/test.csv

Raw data staged and added to DVC. Next steps:
  git add data/data_folder/train/raw/train.zip.dvc ...
  git commit -m 'Track raw data with DVC'
  dvc push
```

---

## 17. **src/eda/quality_checks.py** - Data Quality Validation

### Purpose
Pure functions to validate data integrity and identify anomalies.

### Validation Functions

**`missing_value_report(df: DataFrame) -> DataFrame`**
- Counts missing values per column
- Shows percentage
- Returns sorted report of columns with missing data

**`count_duplicate_rows(df: DataFrame) -> int`**
- Identifies exact row duplicates
- Returns count

**`count_negative_durations(df: DataFrame) -> int`**
- Finds trips with negative duration (impossible)
- Returns count

**`count_invalid_time_sequence(df: DataFrame) -> int`**
- Identifies dropoff_datetime < pickup_datetime
- Indicates data entry errors
- Returns count

**`count_invalid_coordinates(df: DataFrame) -> dict`**
- Validates GPS coordinates within NYC bounds
- Bounds: Latitude [40.5, 41.0], Longitude [-74.25, -73.7]
- Returns dictionary with counts per coordinate type

**`unexpected_store_and_fwd_flag_values(df: DataFrame) -> list`**
- Checks for invalid flag values (should be "Y" or "N")
- Returns list of unexpected values

**`run_quality_checks(train_df: DataFrame) -> dict`**
- Runs all checks and returns combined report
- Output example:
  ```python
  {
      "duplicate_rows": 0,
      "negative_duration_trips": 5,
      "invalid_time_sequence": 12,
      "unexpected_store_and_fwd_flag_values": [],
      "invalid_pickup_lat": 0,
      "invalid_pickup_lon": 3,
      ...
  }
  ```

### Usage
```python
from src.eda.quality_checks import run_quality_checks
report = run_quality_checks(train_df)
print(report)
```

### Benefits
- **Pure Functions**: No I/O, easily testable
- **Comprehensive**: Checks multiple data quality dimensions
- **Actionable**: Identifies specific row counts
- **Modular**: Individual checks can be used independently

---

## 18. **src/eda/plots.py** - EDA Visualization Functions

### Purpose
Generate publication-quality charts for exploratory data analysis.

### Plotting Functions

**`plot_passenger_count(train_df, output_dir)`**
- Count plot: frequency distribution
- Box plot: identify outliers

**`plot_categorical_distributions(train_df, output_dir)`**
- Vendor ID distribution
- Store/Forward flag distribution

**`plot_pickup_dropoff_map(train_df, output_dir, sample_size=1000)`**
- Interactive Folium map
- Blue markers: Pickups
- Red markers: Dropoffs
- Sample 1000 points to avoid overcrowding

**`plot_haversine_distance_distribution(train_df, output_dir)`**
- Histogram with KDE
- Box plot
- Shows trip distance patterns

**`plot_distance_vs_duration(train_df, output_dir)`**
- Scatter plot (sampled 100K points)
- Log scales for better visualization
- Shows correlation

**`plot_pickup_distribution_by_time(train_df, output_dir)`**
- Distribution by hour of day
- Distribution by day of week
- Distribution by month

**`plot_duration_by_time(train_df, output_dir)`**
- Median duration by hour
- Median duration by day of week
- Median duration by month

### Output
- All plots saved as PNG (300 DPI)
- Interactive maps saved as HTML
- Location: `docs/EDA_chart_outputs/`

### Technical Details
- Uses Agg backend (headless, no display needed)
- All plots use consistent styling
- High resolution for publication

---

## 19. **src/eda/report.py** - Full EDA Report Generation

### Purpose
Orchestrates complete EDA pipeline: data inspection, quality checks, visualizations, and output.

### Key Functions

**`inspect(df: DataFrame, name: str) -> None`**
- Prints DataFrame head, info, description
- Shows missing value report
- Output to console

**`generate_plots(train_df: DataFrame) -> None`**
- Calls all plotting functions
- Saves to `EDA_OUTPUT_DIR`
- Prints confirmation messages

**`save_and_track_processed_data(train_df, test_df) -> None`**
- Saves processed data to disk:
  - train: Zipped (large file)
  - test: Plain CSV
- Runs `dvc add` on both
- Prints git commands for next steps

**`main()` - Full Report Execution**
1. Load raw data from DVC
2. Inspect both datasets (console output)
3. Run quality checks
4. Generate feature engineering (haversine, temporal)
5. Generate all visualizations
6. Save processed datasets
7. Track with DVC

### Execution
```bash
python -m src.eda.report
```

### Output Structure
```
docs/EDA_chart_outputs/
├── passenger_count_distribution.png
├── categorical_feature_distributions.png
├── pickup_dropoff_locations.html
├── haversine_distance_distributions.png
├── distance_vs_duration.png
├── pickup_distribution_by_time.png
└── duration_by_time.png

data/data_folder/
├── train/processed/train_eda_processed.zip
└── test/processed/test_eda_processed.csv
```

---

## 20. **src/features/build_features.py** - Basic Feature Engineering

### Purpose
Vectorized feature engineering used by both EDA and inference.

### Functions

**`haversine_distance_km(lat1, lon1, lat2, lon2) -> array`**
- Calculates great-circle distance
- Fully vectorized (fast for DataFrames)
- Returns distance in kilometers

**`add_distance_feature(df: DataFrame) -> DataFrame`**
- Adds `haversine_distance` column
- Returns copy of dataframe

**`add_temporal_features(df, datetime_col="pickup_datetime", prefix="pickup") -> DataFrame`**
- Extracts time-based features from datetime:
  - `pickup_hour`: 0-23
  - `pickup_day_of_week`: 0-6 (Monday-Sunday)
  - `pickup_month`: 1-12
  - `pickup_day`: 1-31
  - `pickup_quarter`: 1-4
- Returns copy with new columns

**`engineer_features(df: DataFrame) -> DataFrame`**
- Applies full feature pipeline
- Calls both distance and temporal functions
- Used by inference pipeline

### Usage
```python
from src.features.build_features import engineer_features
df = pd.DataFrame([{...trip data...}])
df_engineered = engineer_features(df)  # Adds features for inference
```

### Performance
- Vectorized operations (no row-wise loops)
- Efficient for large DataFrames
- Used in production API

---

## 21. **src/features/cleaned_features.py** - Advanced Feature Engineering

### Purpose
Richer feature set for future model retraining (not compatible with current API models).

### Key Functions

**`extract_time_features(dataframe) -> DataFrame`**
- Enhanced version of build_features
- Adds all time features:
  - Season mapping (winter/spring/summer/autumn)
  - Day of year, week of year
  - **Holiday detection**: US Federal holidays
  - Part of day: night/morning/afternoon/evening
- Handles both pickup and dropoff datetime

**`extract_location_features(dataframe, high_traffic_zones=None) -> DataFrame`**
- Comprehensive spatial features:
  - **Manhattan distance**: Sum of |Δlat| + |Δlon|
  - **Haversine distance**: Great-circle distance
  - **Bearing**: Compass direction (0-360°)
  - **Grid zones**: 0.01° cells (consistent between train/test)
  - **Route zones**: Pickup-to-dropoff zone pairs
  - **High-traffic flags**: Boolean indicators

**`extract_trip_features(dataframe) -> DataFrame`**
- Normalizes existing trip attributes
- Type conversions, format standardization

**`extract_interaction_features(dataframe) -> DataFrame`**
- Creates feature combinations:
  - `distance_hour_interaction`: Distance × pickup_hour
  - `day_of_week_holiday_interaction`: Day × holiday

**`prepare_model_features(dataframe) -> DataFrame`**
- Cleans data for training:
  - Drops dropoff columns (not available at inference)
  - Clips values to valid ranges
  - Fills missing values appropriately
  - Encodes categorical variables

**`integrate_external_features(dataframe, external_data) -> DataFrame`**
- Merges optional external data (weather, traffic)
- Prevents target leakage

**`find_high_traffic_zones(dataframe, quantile=0.9) -> set`**
- Identifies frequently-used zones
- Used for high-traffic feature generation

**`find_high_traffic_zones_from_csv(path, chunksize=100_000, quantile=0.9) -> set`**
- Memory-efficient version for large files
- Processes in chunks

### Key Difference from build_features.py

| Aspect | build_features.py | cleaned_features.py |
|--------|-------------------|----------------------|
| Purpose | Inference | Training (future) |
| Dropoff Features | Kept (raw coords) | Dropped |
| Holidays | No | Yes |
| Grid Zones | No | Yes |
| Bearing | No | Yes |
| Interactions | No | Yes |
| External Data | No | Yes |
| Complexity | Simple | Advanced |

---

## 22. **src/features/cleaning_pipeline.py** - Data Processing Pipeline

### Purpose
Chunked read-process-write pipeline for feature engineering at scale.

### Key Functions

**`find_high_traffic_zones_from_csv(path, chunksize=100_000, quantile=0.9) -> set`**
- Finds zones without loading entire file
- Processes in 100K-row chunks
- Returns top 10% zones (90th percentile)

**`write_processed_csv(input_path, output_path, high_traffic_zones, external_data, compression=None, chunksize=100_000) -> None`**
- Streams data through processing pipeline
- Handles both plain CSV and ZIP output
- Incremental writing (low memory usage)

**`load_external_data(path: Path) -> DataFrame | None`**
- Loads optional external data (weather, traffic)
- Returns None if file doesn't exist

**`track_cleaned_files_with_dvc(paths: list[Path]) -> None`**
- Runs `dvc add` on output files
- Makes outputs version-controlled

**`run_feature_cleaning(chunksize=100_000) -> None`**
- Main execution:
  1. Find high-traffic zones
  2. Load optional external data
  3. Process train data (chunked)
  4. Process test data (chunked)
  5. Generate diagnostic charts
  6. Add to DVC tracking

### Execution
```bash
python -m src.features.cleaning_pipeline
```

### Memory Efficiency
```
Traditional (all-in-memory):
Load entire CSV → Process → Write
Problem: 1.4M rows × features = large memory

Chunked (streaming):
Loop through 100K-row chunks:
    Load chunk → Process → Write chunk
Memory: Only 100K rows at a time
```

---

## 23. **src/features/cleaning_plots.py** - Feature Engineering Diagnostics

### Purpose
Visualize engineered features for quality assurance.

### Function

**`generate_feature_engineering_charts(dataframe, output_dir) -> None`**
- Creates 3 diagnostic charts:

1. **Feature Distributions**
   - 6 subplots for key engineered features
   - Haversine distance, Manhattan distance, bearing
   - Distance × hour interaction, passenger count, duration

2. **Correlation Matrix**
   - Heatmap of all numeric features
   - Identifies multicollinearity
   - Shows relationships with target

3. **Categorical Distributions**
   - Bar plots for categorical features
   - Vendor ID, store/fwd flag, season, part of day

### Output
- Saved to `FEATURE_ENGINEERING_OUTPUT_DIR`
- 200 DPI for documentation

---

## 24. **src/models/common.py** - Shared Model Utilities

### Purpose
Reusable preprocessing, evaluation, and persistence functions.

### Key Functions

**`split_features_target(df: DataFrame) -> tuple[DataFrame, Series]`**
- Separates features from target
- Drops non-feature columns (id, datetime, target)
- Removes rows with invalid targets
- Returns: (X, y)

**`build_preprocessor(X: DataFrame, scale_numeric: bool=False) -> ColumnTransformer`**
- Creates sklearn preprocessing pipeline:

**Numeric Branch:**
- Impute missing with median
- Optionally scale to mean=0, std=1

**Categorical Branch:**
- Impute missing with most frequent
- One-hot encode

**`evaluate_regression(y_true, y_pred) -> dict`**
- Calculates metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² (coefficient of determination)
- Returns dictionary

**`format_metrics(metrics: dict) -> str`**
- Pretty-prints metrics
- Example: `"MAE=185.32 | RMSE=320.15 | R2=0.7023"`

**`save_model(model, path: Path) -> None`**
- Saves sklearn pipeline to joblib
- Creates parent directories
- Prints confirmation

### Benefits
- **Reusability**: Used by both linear_regression and gradient_boosting
- **Consistency**: Same preprocessing across models
- **Testability**: Pure functions with inputs/outputs
- **Maintainability**: Changes in one place affect both models

---

## 25. **src/models/tracking.py** - MLflow Integration

### Purpose
Experiment tracking and reproducibility management.

### Key Functions

**`configure_mlflow(tracking_uri: str | None=None) -> None`**
- Sets MLflow tracking server
- Sets experiment name
- Called before starting run

**`start_run_if_enabled(run_name, enabled, tracking_uri=None) -> context manager`**
- Conditional MLflow context manager
- Yields active run if enabled, else None
- Enables:
  ```python
  with start_run_if_enabled(...) as run:
      if run is not None:
          mlflow.log_metrics(...)
  ```

**`log_dataset_tags(feature_set: str, dataset_path: Path) -> None`**
- Tags run with:
  - Feature set version (raw/processed/cleaned)
  - Dataset path
  - DVC MD5 hash (data version)
- Enables complete reproducibility tracking

### MLflow Recording
- **Parameters**: Hyperparameters (learning_rate, n_estimators, etc.)
- **Metrics**: Performance (MAE, RMSE, R²) at each stage
- **Tags**: Metadata (data version, feature set)
- **Artifacts**: Trained model pickle
- **Run Info**: Duration, status, timestamps

### Tracking URI
```python
MLFLOW_TRACKING_URI = "mlruns/"  # Local file storage
# Or: "sqlite:///mlruns.db" for SQL backend
# Or: "http://mlflow-server:5000" for remote server
```

### Benefits
- **Reproducibility**: Track which data/code/params produced which results
- **Comparison**: MLflow UI to compare experiments
- **Governance**: Who trained what model when
- **Versioning**: Complete experiment history

---

## 26. **src/models/backfill_existing_models.py** - Legacy Model Registration

### Purpose
Register pre-existing trained models into MLflow without retraining.

### Key Functions

**`backfill_model(name: str, path=None) -> None`**
- Loads joblib model from disk
- Creates MLflow run tagged as "backfilled"
- Logs model artifact
- No parameters/metrics (trained before tracking)

**`backfill_all() -> None`**
- Registers all models in SERVING_MODELS
- Called once to backfill historical models

### Execution
```bash
python -m src.models.backfill_existing_models
```

### Output
```
Loading 'gradient_boosting' from models/...
Backfilled 'gradient_boosting' into MLflow.
Loading 'linear_regression' from models/...
Backfilled 'linear_regression' into MLflow.
```

### Use Case
- Models trained with older notebooks (pre-MLflow)
- Want to register them in MLflow for comparison
- Tagged distinctly from new training runs

---

## Summary: Production Architecture

### Data Pipeline
```
Raw Data (Google Drive)
    ↓
src.data.load (DVC pull if needed)
    ↓
src.eda.report (Inspect, validate, visualize)
    ↓
src.features.cleaning_pipeline (Engineer features, chunked)
    ↓
src.data.load (load_cleaned_train_test)
    ↓
Model Training (Linear/Gradient Boosting)
```

### Inference Pipeline
```
HTTP Request (TripRequest)
    ↓
src.api.app (/predict endpoint)
    ↓
src.api.inference (load_model from cache)
    ↓
src.features.build_features (engineer_features)
    ↓
Model.predict()
    ↓
HTTP Response (PredictionResponse)
```

### Experiment Tracking
```
Training Script
    ├─ src.models.common (preprocess, evaluate)
    ├─ src.models.tracking (MLflow logging)
    └─ src.config (load config)
    ↓
MLflow Experiment Storage (mlruns/)
    ↓
MLflow UI (localhost:5000)
```

### Module Dependencies
```
src.api.app
    └─ src.api.inference
        ├─ src.features.build_features
        └─ src.config

src.models.train_*
    ├─ src.data.load
    ├─ src.features.build_features / cleaned_features
    ├─ src.models.common
    ├─ src.models.tracking
    └─ src.config

src.eda.report
    ├─ src.data.load
    ├─ src.eda.plots
    ├─ src.eda.quality_checks
    ├─ src.features.build_features
    └─ src.config

src.features.cleaning_pipeline
    ├─ src.features.cleaned_features
    ├─ src.features.cleaning_plots
    └─ src.config
```

---

For detailed documentation, see `docs/Project Documentation_G28.docx`

