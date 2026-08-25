import pandas as pd
import numpy as np
import os as os
import shutil as shutil
import subprocess
import sys

#Add raw data files to dvc
raw_train_path = "data/data_folder/train/train.zip"
raw_test_path = "data/data_folder/test/test.csv"

# Create a directory for raw data within DVC tracking if it doesn't exist
os.makedirs("data/data_folder/train/raw", exist_ok=True)

os.makedirs("data/data_folder/test/raw", exist_ok=True)

# Copy raw files to a DVC-tracked directory for consistency (optional but good practice)
print("Copying raw data files to DVC-tracked directories...")
print("Copying train.zip ...")
shutil.copy2(raw_train_path, "data/data_folder/train/raw/train.zip")
#check if the train.zip file is copied successfully
if os.path.exists("data/data_folder/train/raw/train.zip"):
    print("Train data copied successfully.")
print("Copying test.csv ...")
shutil.copy2(raw_test_path, "data/data_folder/test/raw/test.csv")
#Check if the test.csv file is copied successfully
if os.path.exists("data/data_folder/test/raw/test.csv"):
    print("Test data copied successfully.")

#Versioning the raw data files using DVC
print("\nAdding raw data files to DVC tracking...")
dvc_executable = os.path.join(os.path.dirname(sys.executable), "dvc.exe")
if not os.path.isdir(".dvc"):
    subprocess.run([dvc_executable, "init"], check=True)

subprocess.run(
    [dvc_executable, "add", "data/data_folder/train/raw/train.zip"],
    check=True,
)
subprocess.run(
    [dvc_executable, "add", "data/data_folder/test/raw/test.csv"],
    check=True,
)
subprocess.run(
    [dvc_executable, "status", "data/data_folder/train/raw/train.zip"],
    check=True,
)
subprocess.run(
    [dvc_executable, "status", "data/data_folder/test/raw/test.csv"],
    check=True,
)
dvc_train_push_command = "dvc push data/data_folder/train/raw/train.zip"
dvc_test_push_command = "dvc push data/data_folder/test/raw/test.csv"
dvc_train_commit_command = "git add data/data_folder/train/raw/train.zip.dvc"
dvc_test_commit_command = "git add data/data_folder/test/raw/test.csv.dvc"
dvc_train_commit_message = "git commit -m 'Add raw train data to DVC tracking'"
dvc_test_commit_message = "git commit -m 'Add raw test data to DVC tracking'"
dvc_train_push_git_command = "git push origin main"
dvc_test_push_git_command = "git push origin main"
dvc_train_pull_command = "dvc pull data/data_folder/train/raw/train.zip"
dvc_test_pull_command = "dvc pull data/data_folder/test/raw/test.csv"
dvc_train_remove_command = "dvc remove data/data_folder/train/raw/train.zip"
dvc_test_remove_command = "dvc remove data/data_folder/test/raw/test.csv"

#git add .dvc data/data_folder/**/*.dvc
#git commit -m "Track raw data with DVC"