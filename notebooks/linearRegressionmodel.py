"""Train a linear regression model to predict trip duration in seconds."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(
	r"C:\Users\rakes\BITS_PAIML_MLE_G28\data\data_folder\train\processed\train_eda_processed.csv"
)
MODEL_PATH = DATA_PATH.with_name("duration_linear_regression.joblib")


def main() -> None:
	data = pd.read_csv(DATA_PATH)
	target = "trip_duration"

	if target not in data.columns:
		raise ValueError(f"Target column '{target}' was not found in the CSV file.")

	# Duration is the response variable and is measured in seconds.
	y = pd.to_numeric(data[target], errors="coerce")
	X = data.drop(columns=[target, "id", "pickup_datetime", "dropoff_datetime"], errors="ignore")
	valid_rows = y.notna()
	X, y = X.loc[valid_rows], y.loc[valid_rows]

	if X.empty:
		raise ValueError("The dataset contains no valid rows for training.")

	numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
	categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

	numeric_pipeline = Pipeline(
		steps=[("imputer", SimpleImputer(strategy="median"))]
	)
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("encoder", OneHotEncoder(handle_unknown="ignore")),
		]
	)
	preprocessor = ColumnTransformer(
		transformers=[
			("numeric", numeric_pipeline, numeric_columns),
			("categorical", categorical_pipeline, categorical_columns),
		]
	)
	model = Pipeline(
		steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())]
	)

	print("Training linear regression model...")
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)

	mae = mean_absolute_error(y_test, predictions)
	rmse = mean_squared_error(y_test, predictions) ** 0.5
	r2 = r2_score(y_test, predictions)

	print("Training metrics:")
	print(f"MAE (seconds): {mae:.2f}")
	print(f"RMSE (seconds): {rmse:.2f}")
	print(f"R2 score: {r2:.4f}")

	joblib.dump(model, MODEL_PATH)
	print(f"Model saved to: {MODEL_PATH}")



if __name__ == "__main__":
	main()