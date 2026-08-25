import pandas as pd
import numpy as np
import os as os
import shutil as shutil
import subprocess
import sys
import matplotlib.pyplot as plt
import seaborn as sns

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
