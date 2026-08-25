import pandas as pd
import numpy as np
import os as os
import shutil as shutil
import subprocess
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from math import radians, sin, cos, sqrt, atan2

print("Loading train data...")
train_df = pd.read_csv("data/data_folder/train/raw/train.zip")
print("Training data loaded successfully.")

print("Loading test data...")
test_df = pd.read_csv("data/data_folder/test/raw/test.csv")
print("Testing data loaded successfully.")

print("\nInitial Data Inspection:")

if not train_df.empty:
    print("\n--- Train DataFrame Head ---")
    print(train_df.head())
    print("\n--- Train DataFrame Info ---")
    train_df.info()
    print("\n--- Train DataFrame Description ---")
    print(train_df.describe())
    print("\n--- Train DataFrame Missing Values ---")
    print(train_df.isnull().sum()[train_df.isnull().sum() > 0].sort_values(ascending=False))

if not test_df.empty:
    print("\n--- Test DataFrame Head ---")
    print(test_df.head())
    print("\n--- Test DataFrame Info ---")
    test_df.info()
    print("\n--- Test DataFrame Description ---")
    print(test_df.describe())
    print("\n--- Test DataFrame Missing Values ---")
    print(test_df.isnull().sum()[test_df.isnull().sum() > 0].sort_values(ascending=False))

print("\nData inspection completed.")

print("\nData Preparation for Temporal and Spatial EDA...")

# Convert datetime columns to datetime objects
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])

test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])
## Note: test_df does not have dropoff_datetime or trip_duration

print("Datetime columns converted.")
print("\n--- Train DataFrame Info after datetime conversion ---")
train_df.info()

# Univariate Analysis: Continuous Features
eda_output_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs",
    "EDA_chart_outputs",
)
os.makedirs(eda_output_dir, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.countplot(data=train_df, x='passenger_count', ax=axes[0])
axes[0].set_title('Passenger Count Distribution (Train)')
axes[0].set_xlabel('Passenger Count')
axes[0].set_ylabel('Number of Trips')

sns.boxplot(data=train_df, x='passenger_count', ax=axes[1])
axes[1].set_title('Passenger Count Box Plot (Train)')
axes[1].set_xlabel('Passenger Count')

plt.tight_layout()
plt.savefig(os.path.join(eda_output_dir, "passenger_count_distribution.png"), dpi=300)
plt.close(fig)

print("Passenger count value counts:")
print(train_df['passenger_count'].value_counts().sort_index())

# Univariate Analysis: Categorical Features
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Vendor ID Distribution
sns.countplot(data=train_df, x='vendor_id', ax=axes[0])
axes[0].set_title('Distribution of Vendor ID (Train)')
axes[0].set_xlabel('Vendor ID')
axes[0].set_ylabel('Number of Trips')

# Store and Fwd Flag Distribution
sns.countplot(data=train_df, x='store_and_fwd_flag', ax=axes[1])
axes[1].set_title('Distribution of Store and Fwd Flag (Train)')
axes[1].set_xlabel('Store and Fwd Flag')
axes[1].set_ylabel('Number of Trips')

plt.tight_layout()
plt.savefig(os.path.join(eda_output_dir, "categorical_feature_distributions.png"), dpi=300)
plt.close(fig)

print("\n--- Vendor ID Value Counts ---")
print(train_df['vendor_id'].value_counts())

print("\n--- Store and Fwd Flag Value Counts ---")
print(train_df['store_and_fwd_flag'].value_counts())

# Spatial Analysis: Pickup and Dropoff Locations
print("\n--- Spatial Analysis: Pickup and Dropoff Locations ---")
def plot_pickup_dropoff_locations(df, title='Pickup and Dropoff Locations'):
    # Create a base map centered around New York City
    nyc_center = [40.75, -73.98]
    m = folium.Map(location=nyc_center, zoom_start=12)

    # Add pickup locations (a sample to avoid overcrowding)
    # Sample to avoid exceeding Colab's rendering limits and to keep it performant
    sample_df = df.sample(n=min(len(df), 1000), random_state=42)
    for index, row in sample_df.iterrows():
        folium.CircleMarker(
            location=[row['pickup_latitude'], row['pickup_longitude']],
            radius=3,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            tooltip=f"Pickup: {row['pickup_datetime']}"
        ).add_to(m)

        # Add dropoff locations
        # Ensure dropoff_datetime exists for train_df. For test_df, it won't be there.
        if 'dropoff_datetime' in row.index:
            folium.CircleMarker(
                location=[row['dropoff_latitude'], row['dropoff_longitude']],
                radius=3,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6,
                tooltip=f"Dropoff: {row['dropoff_datetime']}"
            ).add_to(m)

    return m

print("Generating map for a sample of pickup (blue) and dropoff (red) locations. This may take a moment.")

# Display the map for a sample of the training data
pickup_dropoff_map = plot_pickup_dropoff_locations(train_df, title='Sample of Pickup (Blue) and Dropoff (Red) Locations')
#print(pickup_dropoff_map)
pickup_dropoff_map.save(os.path.join(eda_output_dir, "pickup_dropoff_locations.html"))
print("\nMap generated successfully.")

# Feature Engineering (for EDA): Haversine Distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Apply Haversine distance calculation to the training data
train_df['haversine_distance'] = train_df.apply(
    lambda row: haversine_distance(
        row['pickup_latitude'], row['pickup_longitude'],
        row['dropoff_latitude'], row['dropoff_longitude']
    ),
    axis=1
)

# Apply Haversine distance calculation to the test data
test_df['haversine_distance'] = test_df.apply(
    lambda row: haversine_distance(
        row['pickup_latitude'], row['pickup_longitude'],
        row['dropoff_latitude'], row['dropoff_longitude']
    ),
    axis=1
)

print("\nHaversine distance calculated for both train_df and test_df.")
print("\n--- Train DataFrame Head with Haversine Distance ---")
print(train_df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'haversine_distance']].head())

print("\n--- Haversine Distance Descriptive Statistics ---")
print(train_df['haversine_distance'].describe())

# Univariate Analysis: Haversine Distance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.histplot(train_df['haversine_distance'], bins=50, kde=True, ax=axes[0])
axes[0].set_title('Haversine Distance Distribution (Train)')
axes[0].set_xlabel('Haversine Distance (km)')
axes[0].set_ylabel('Frequency')

sns.boxplot(x=train_df['haversine_distance'], ax=axes[1])
axes[1].set_title('Haversine Distance Box Plot (Train)')
axes[1].set_xlabel('Haversine Distance (km)')

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(eda_output_dir, "haversine_distance_distributions.png"), dpi=300)
plt.close(fig)
print("\nHaversine Distance Distribution generated successfully.")

print("\nHaversine Distance Descriptive Statistics:")
print(train_df['haversine_distance'].describe())

print("\nTrips with zero Haversine distance:")
print(train_df[train_df['haversine_distance'] == 0].shape[0])

# Bivariate Analysis: Haversine Distance vs. Trip Duration
# To avoid plotting millions of points, let's sample the data for the scatter plot
sample_for_plot = train_df.sample(n=100000, random_state=42)

plt.figure(figsize=(12, 8))
sns.scatterplot(x='haversine_distance', y='trip_duration', data=sample_for_plot, alpha=0.5)
plt.title('Haversine Distance vs. Trip Duration (Sampled)')
plt.xlabel('Haversine Distance (km)')
plt.ylabel('Trip Duration (seconds)')
plt.xscale('log') # Use log scale for distance due to potential outliers
plt.yscale('log') # Use log scale for duration due to wide range
plt.grid(True, which="both", ls="--", c='0.7')
#plt.show()
plt.savefig(os.path.join(eda_output_dir, "haversine_distance_v_trip_duration.png"), dpi=300)
plt.close(fig)
print("\nHaversine Distance v Trip Duration Distribution generated successfully.")

print("\nCorrelation between Haversine Distance and Trip Duration:")
print(train_df[['haversine_distance', 'trip_duration']].corr())

# Temporal Analysis: Extracting Time-Based Features
# Extract temporal features for training data
train_df['pickup_hour'] = train_df['pickup_datetime'].dt.hour
train_df['pickup_day_of_week'] = train_df['pickup_datetime'].dt.dayofweek # Monday=0, Sunday=6
train_df['pickup_month'] = train_df['pickup_datetime'].dt.month
train_df['pickup_day'] = train_df['pickup_datetime'].dt.day
train_df['pickup_quarter'] = train_df['pickup_datetime'].dt.quarter

# Extract temporal features for test data (only pickup_datetime is available)
test_df['pickup_hour'] = test_df['pickup_datetime'].dt.hour
test_df['pickup_day_of_week'] = test_df['pickup_datetime'].dt.dayofweek
test_df['pickup_month'] = test_df['pickup_datetime'].dt.month
test_df['pickup_day'] = test_df['pickup_datetime'].dt.day
test_df['pickup_quarter'] = test_df['pickup_datetime'].dt.quarter

print("Temporal features extracted from pickup_datetime.")
print("\n--- Train DataFrame Head with new temporal features ---")
print(train_df[['pickup_datetime', 'pickup_hour', 'pickup_day_of_week', 'pickup_month', 'pickup_day', 'pickup_quarter']].head())

# Univariate Analysis: Temporal Features
fig, axes = plt.subplots(3, 1, figsize=(16, 18))

# Pickup Hour Distribution
sns.countplot(data=train_df, x='pickup_hour', ax=axes[0])
axes[0].set_title('Distribution of Pickups by Hour of Day (Train)')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Number of Trips')

# Pickup Day of Week Distribution
sns.countplot(data=train_df, x='pickup_day_of_week', ax=axes[1])
axes[1].set_title('Distribution of Pickups by Day of Week (Train)')
axes[1].set_xlabel('Day of Week (0=Monday, 6=Sunday)')
axes[1].set_ylabel('Number of Trips')

# Pickup Month Distribution
sns.countplot(data=train_df, x='pickup_month', ax=axes[2])
axes[2].set_title('Distribution of Pickups by Month (Train)')
axes[2].set_xlabel('Month')
axes[2].set_ylabel('Number of Trips')

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(eda_output_dir, "pickup_distribution_by_hourofday_dayofweek_month.png"), dpi=300)
plt.close(fig)
print("\nPickup Distribution By Hour of Day/Day of Week/Month generated successfully.")

# Bivariate Analysis: Trip Duration vs. Temporal Features
fig, axes = plt.subplots(3, 1, figsize=(16, 18))

# Trip Duration vs. Pickup Hour
sns.barplot(data=train_df, x='pickup_hour', y='trip_duration', ax=axes[0], estimator=np.median)
axes[0].set_title('Median Trip Duration by Hour of Day (Train)')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Median Trip Duration (seconds)')

# Trip Duration vs. Pickup Day of Week
sns.barplot(data=train_df, x='pickup_day_of_week', y='trip_duration', ax=axes[1], estimator=np.median)
axes[1].set_title('Median Trip Duration by Day of Week (Train)')
axes[1].set_xlabel('Day of Week (0=Monday, 6=Sunday)')
axes[1].set_ylabel('Median Trip Duration (seconds)')

# Trip Duration vs. Pickup Month
sns.barplot(data=train_df, x='pickup_month', y='trip_duration', ax=axes[2], estimator=np.median)
axes[2].set_title('Median Trip Duration by Month (Train)')
axes[2].set_xlabel('Month')
axes[2].set_ylabel('Median Trip Duration (seconds)')

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(eda_output_dir, "median_trip_duration_distribution_by_hourofday_dayofweek_month.png"), dpi=300)
plt.close(fig)
print("\nMedian Trip Duration By Hour of Day/Day of Week/Month generated successfully.")

# Missing Value Analysis
def check_missing_values(df, df_name):
    missing_counts = df.isnull().sum()
    missing_percentages = 100 * df.isnull().sum() / len(df)
    missing_info = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage': missing_percentages
    })
    missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

    print(f"\n--- Missing Values in {df_name} ---")
    if missing_info.empty:
        print("No missing values found.")
    else:
        print(missing_info)

check_missing_values(train_df, 'Train DataFrame')
check_missing_values(test_df, 'Test DataFrame')

print("\n--- Checking for Duplicate Records ---")
duplicate_rows = train_df.duplicated().sum()
print(f"Number of duplicate rows in train_df: {duplicate_rows}")

# Check for logical inconsistencies in trip duration (e.g., negative duration)
print("\n--- Checking Trip Duration Logical Consistency ---")
negative_duration_trips = train_df[train_df['trip_duration'] < 0].shape[0]
print(f"Number of trips with negative duration: {negative_duration_trips}")

# Check if dropoff_datetime is before pickup_datetime
if 'dropoff_datetime' in train_df.columns:
    invalid_time_sequence = train_df[train_df['dropoff_datetime'] < train_df['pickup_datetime']].shape[0]
    print(f"Number of trips where dropoff_datetime is before pickup_datetime: {invalid_time_sequence}")

# Check for geographical coordinate validity for NYC
print("\n--- Checking Geographical Coordinate Validity (NYC) ---")
NYC_LAT_MIN, NYC_LAT_MAX = 40.5, 41.0
NYC_LON_MIN, NYC_LON_MAX = -74.25, -73.7

invalid_pickup_lat = train_df[(train_df['pickup_latitude'] < NYC_LAT_MIN) | (train_df['pickup_latitude'] > NYC_LAT_MAX)].shape[0]
invalid_pickup_lon = train_df[(train_df['pickup_longitude'] < NYC_LON_MIN) | (train_df['pickup_longitude'] > NYC_LON_MAX)].shape[0]
invalid_dropoff_lat = train_df[(train_df['dropoff_latitude'] < NYC_LAT_MIN) | (train_df['dropoff_latitude'] > NYC_LAT_MAX)].shape[0]
invalid_dropoff_lon = train_df[(train_df['dropoff_longitude'] < NYC_LON_MIN) | (train_df['dropoff_longitude'] > NYC_LON_MAX)].shape[0]

print(f"Invalid pickup latitudes: {invalid_pickup_lat}")
print(f"Invalid pickup longitudes: {invalid_pickup_lon}")
print(f"Invalid dropoff latitudes: {invalid_dropoff_lat}")
print(f"Invalid dropoff longitudes: {invalid_dropoff_lon}")

# Check for consistency in 'store_and_fwd_flag'
print("\n--- Checking 'store_and_fwd_flag' Consistency ---")
unique_flags = train_df['store_and_fwd_flag'].unique()
print(f"Unique values in 'store_and_fwd_flag': {unique_flags}")
if not all(f in ['Y', 'N'] for f in unique_flags):
    print("Warning: 'store_and_fwd_flag' contains unexpected values.")
else:
    print("'store_and_fwd_flag' values are consistent (Y/N).")

#Post EDA DVC
processed_train_path = f'data/data_folder/train/processed/train_eda_processed.csv'
processed_test_path = f'data/data_folder/test/processed//test_eda_processed.csv'

# Create a directory for processed data within DVC tracking if it doesn't exist
os.makedirs("data/data_folder/train/processed", exist_ok=True)

os.makedirs("data/data_folder/test/processed", exist_ok=True)

# Save the currently processed DataFrames
train_df.to_csv(processed_train_path, index=False)
print(f"Processed train_df saved to: {processed_train_path}")
processed_train_zip_path = shutil.make_archive(
    processed_train_path.removesuffix('.csv'),
    'zip',
    root_dir=os.path.dirname(processed_train_path),
    base_dir=os.path.basename(processed_train_path),
)
os.remove(processed_train_path)
print(f"Processed train_df zipped to: {processed_train_zip_path}")
test_df.to_csv(processed_test_path, index=False)
print(f"Processed test_df saved to: {processed_test_path}")

# Verify the processed files were saved successfully
print("Verifying processed data files...")
if os.path.exists(processed_train_zip_path):
    print("Processed Train data ZIP saved successfully.")
if os.path.exists("data/data_folder/test/processed/test_eda_processed.csv"):
    print("Processed Test data saved successfully.")

#Versioning the raw data files using DVC
print("\nAdding processed data files to DVC tracking...")
dvc_executable = shutil.which("dvc")
if dvc_executable is None:
    dvc_executable = os.path.join(os.path.dirname(sys.executable), "dvc.exe")
if not os.path.isfile(dvc_executable):
    raise RuntimeError(
        "DVC is not installed for the active Python environment. "
        "Install it with 'python -m pip install dvc' and rerun EDA.py."
    )
if not os.path.isdir(".dvc"):
    subprocess.run([dvc_executable, "init"], check=True)

subprocess.run(
    [dvc_executable, "add", processed_train_zip_path],
    check=True,
)
subprocess.run(
    [dvc_executable, "add", "data/data_folder/test/processed/test_eda_processed.csv"],
    check=True,
)
subprocess.run(
    [dvc_executable, "status", processed_train_zip_path],
    check=True,
)
subprocess.run(
    [dvc_executable, "status", "data/data_folder/test/processed/test_eda_processed.csv"],
    check=True,
)

print("\nAdded processed data files to DVC tracking successfully...")

print("\nEDA and data preparation completed successfully.")