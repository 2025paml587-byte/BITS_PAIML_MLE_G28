#import pandas as pd
import os
import shutil # For moving filesimport os

project_root = r"C:\Users\Kiran\BITS PILANI AI_ML\MiniProject\BITS_PAIML_MLE_G28"
print("Current directory:", os.getcwd())

os.chdir(project_root)
print("Project root:", os.getcwd())

os.chdir(os.path.join(project_root, "data"))
print("Data folder:", os.getcwd())

print("Current directory:", os.getcwd())

os.makedirs('data_folder', exist_ok=True)

#from kaggle.api.kaggle_api_extended import KaggleApi
import os

#data_folder = r"C:\Users\Kiran\BITS PILANI AI_ML\MiniProject\BITS_PAIML_MLE_G28\data"
#os.makedirs(data_folder, exist_ok=True)

#api = KaggleApi()
#api.authenticate()

# For a dataset
#api.dataset_download_files("owner/dataset-name", path=data_folder, unzip=True)

# For a competition
# api.competition_download_files("competition-name", path=data_folder, unzip=True)