#!/usr/bin/env python
"""Extract actual model metrics and data for documentation."""

import joblib
from pathlib import Path
import sys
import traceback

output_file = Path("model_metrics_extracted.txt")

with open(output_file, 'w') as out:
    out.write("="*80 + "\n")
    out.write("MODEL METRICS EXTRACTION\n")
    out.write("="*80 + "\n\n")
    
    try:
        # Load Linear Regression model
        lr_path = Path('models/linear_regression_trip_duration.joblib')
        if lr_path.exists():
            out.write(f"1. LINEAR REGRESSION MODEL\n")
            out.write(f"   File: {lr_path}\n")
            lr_model = joblib.load(lr_path)
            out.write(f"   Type: {type(lr_model).__name__}\n")
            out.write(f"   Model: {lr_model}\n\n")
        else:
            out.write(f"LINEAR REGRESSION MODEL NOT FOUND\n\n")
        
        # Load Gradient Boosting model
        gb_path = Path('models/gradient_boosting_trip_duration.joblib')
        if gb_path.exists():
            out.write(f"2. GRADIENT BOOSTING MODEL\n")
            out.write(f"   File: {gb_path}\n")
            gb_model = joblib.load(gb_path)
            out.write(f"   Type: {type(gb_model).__name__}\n")
            
            # Extract hyperparameters
            if hasattr(gb_model, 'n_estimators'):
                out.write(f"   n_estimators: {gb_model.n_estimators}\n")
            if hasattr(gb_model, 'learning_rate'):
                out.write(f"   learning_rate: {gb_model.learning_rate}\n")
            if hasattr(gb_model, 'max_depth'):
                out.write(f"   max_depth: {gb_model.max_depth}\n")
            if hasattr(gb_model, 'min_samples_split'):
                out.write(f"   min_samples_split: {gb_model.min_samples_split}\n")
            if hasattr(gb_model, 'min_samples_leaf'):
                out.write(f"   min_samples_leaf: {gb_model.min_samples_leaf}\n")
            
            # Extract feature importances if available
            if hasattr(gb_model, 'feature_importances_'):
                out.write(f"\n   Feature Importances:\n")
                importances = gb_model.feature_importances_
                for i, imp in enumerate(sorted(enumerate(importances), key=lambda x: x[1], reverse=True)[:10]):
                    out.write(f"     Feature {imp[0]}: {imp[1]:.6f}\n")
            
            out.write(f"\n")
        else:
            out.write(f"GRADIENT BOOSTING MODEL NOT FOUND\n\n")
        
        # Check test data
        test_data_path = Path('data/data_folder/test/processed/test_cleaned.csv')
        if test_data_path.exists():
            out.write(f"3. TEST DATA\n")
            out.write(f"   File: {test_data_path}\n")
            out.write(f"   Size: {test_data_path.stat().st_size / (1024*1024):.2f} MB\n\n")
    
    except Exception as e:
        out.write(f"ERROR: {e}\n")
        out.write(traceback.format_exc())

print(f"Results written to {output_file}")
