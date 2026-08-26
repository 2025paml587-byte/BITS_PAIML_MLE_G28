"""Train and save a gradient boosting model for trip duration prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(
	r"C:\Users\rakes\BITS_PAIML_MLE_G28\data\data_folder\train\processed\train_eda_processed.csv"
)
MODEL_PATH = DATA_PATH.parents[4] / "models" / "gradient_boosting_trip_duration.joblib"
TARGET = "trip_duration"


def train_model(n_estimators: int = 300, progress_every: int = 25):
	print(f"Loading training data from: {DATA_PATH}")
	data = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)
	if TARGET not in data.columns:
		raise ValueError(f"Missing target column: {TARGET}")

	print("Preparing target and feature matrix...")
	y = pd.to_numeric(data.pop(TARGET), errors="coerce")
	X = data.drop(columns=["id", "pickup_datetime", "dropoff_datetime"], errors="ignore")
	valid = y.notna()
	X, y = X.loc[valid], y.loc[valid]
	print(f"Rows after removing invalid targets: {len(X)}")
	numeric = X.select_dtypes(include=np.number).columns.tolist()
	categorical = X.select_dtypes(exclude=np.number).columns.tolist()
	print(f"Numeric features: {len(numeric)}")
	print(f"Categorical features: {len(categorical)}")
	if categorical:
		print(f"Categorical columns: {categorical}")

	transformers = []
	if numeric:
		transformers.append(
			("numeric", Pipeline([
				("imputer", SimpleImputer(strategy="median")),
				("scaler", StandardScaler()),
			]), numeric)
		)
	if categorical:
		transformers.append(
			("categorical", Pipeline([
				("imputer", SimpleImputer(strategy="most_frequent")),
				("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
			]), categorical)
		)

	model = Pipeline([
		("preprocessor", ColumnTransformer(transformers)),
		("regressor", GradientBoostingRegressor(
			n_estimators=n_estimators,
			learning_rate=0.05,
			max_depth=4,
			loss="huber",
			random_state=42,
			warm_start=True,
		)),
	])

	print("Splitting data into train and test sets...")
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)
	print(f"Training set size: {len(X_train)}")
	print(f"Test set size: {len(X_test)}")
	print(f"Training progress will be logged every {progress_every} trees.")

	print("Starting model training...")
	final_stage = n_estimators
	for stage in range(progress_every, final_stage + 1, progress_every):
		model.named_steps["regressor"].set_params(n_estimators=stage)
		model.fit(X_train, y_train)
		predictions = model.predict(X_test)
		mae = mean_absolute_error(y_test, predictions)
		rmse = mean_squared_error(y_test, predictions) ** 0.5
		r2 = r2_score(y_test, predictions)
		print(
			f"Training progress: {stage}/{final_stage} trees | "
			f"MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}"
		)

	if final_stage % progress_every != 0:
		model.named_steps["regressor"].set_params(n_estimators=final_stage)
		model.fit(X_train, y_train)
		predictions = model.predict(X_test)
		mae = mean_absolute_error(y_test, predictions)
		rmse = mean_squared_error(y_test, predictions) ** 0.5
		r2 = r2_score(y_test, predictions)
		print(
			f"Training progress: {final_stage}/{final_stage} trees | "
			f"MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}"
		)

	print("Training complete. Final evaluation:")
	predictions = model.predict(X_test)
	print(f"MAE: {mean_absolute_error(y_test, predictions):.3f}")
	print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.3f}")
	print(f"R2: {r2_score(y_test, predictions):.3f}")

	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	print(f"Saving model to: {MODEL_PATH}")
	joblib.dump(model, MODEL_PATH)
	print("Model saved successfully.")
	return model


if __name__ == "__main__":
	train_model()
