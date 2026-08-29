import joblib
import json
from pathlib import Path
import os
import pickle

print("="*80)
print("GATHERING ACTUAL PROJECT DATA")
print("="*80)

# Check for trained models
print("\n1. TRAINED MODELS:")
model_dir = Path('models')
if model_dir.exists():
    for model_file in sorted(model_dir.glob('*.joblib')):
        size_kb = model_file.stat().st_size / 1024
        print(f"  - {model_file.name} ({size_kb:.1f} KB)")
        
        # Try to load and inspect the model
        try:
            model = joblib.load(model_file)
            print(f"    Type: {type(model).__name__}")
            if hasattr(model, 'n_estimators'):
                print(f"    n_estimators: {model.n_estimators}")
            if hasattr(model, 'learning_rate'):
                print(f"    learning_rate: {model.learning_rate}")
            if hasattr(model, 'max_depth'):
                print(f"    max_depth: {model.max_depth}")
        except Exception as e:
            print(f"    Error loading: {e}")

# Check for trained models in data/data_folder/train/processed
print("\n2. PROCESSED MODELS:")
train_processed_dir = Path('data/data_folder/train/processed')
if train_processed_dir.exists():
    for model_file in sorted(train_processed_dir.glob('*.joblib')):
        size_kb = model_file.stat().st_size / 1024
        print(f"  - {model_file.name} ({size_kb:.1f} KB)")

# Check for test data
print("\n3. TEST DATA:")
test_dir = Path('data/data_folder/test')
if test_dir.exists():
    for test_file in sorted(test_dir.rglob('*.csv')):
        size_mb = test_file.stat().st_size / (1024*1024)
        print(f"  - {test_file.relative_to(test_dir)} ({size_mb:.2f} MB)")

# Check MLflow
print("\n4. MLFLOW TRACKING:")
mlruns_dir = Path('mlruns')
if mlruns_dir.exists():
    for exp_dir in sorted(mlruns_dir.iterdir()):
        if exp_dir.is_dir() and exp_dir.name != '0':
            print(f"  Experiment {exp_dir.name}:")
            for run_dir in sorted(exp_dir.iterdir()):
                if run_dir.is_dir():
                    meta_file = run_dir / 'meta.yaml'
                    if meta_file.exists():
                        print(f"    Run {run_dir.name}:")
                        # Try to read params and metrics
                        params_file = run_dir / 'params.yaml'
                        metrics_file = run_dir / 'metrics.yaml'
                        if params_file.exists():
                            print(f"      Has params")
                        if metrics_file.exists():
                            print(f"      Has metrics")

# Check DVC files
print("\n5. DVC TRACKED FILES:")
for dvc_file in sorted(Path('data').rglob('*.dvc')):
    print(f"  - {dvc_file.relative_to(Path('data'))}")

# Check for results/test outputs
print("\n6. RESULTS/OUTPUTS:")
results_dir = Path('results')
if results_dir.exists():
    for result_file in sorted(results_dir.glob('*')):
        print(f"  - {result_file.name}")

# Check notebooks for execution outputs
print("\n7. NOTEBOOK OUTPUTS:")
notebooks_dir = Path('notebooks')
if notebooks_dir.exists():
    print(f"  Notebooks: {len(list(notebooks_dir.glob('*.py')))}")

print("\nData gathering complete.")
