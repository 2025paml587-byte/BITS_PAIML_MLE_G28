from pathlib import Path
import subprocess
import zipfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ZIP = (
	PROJECT_ROOT
	/ "data"
	/ "data_folder"
	/ "train"
	/ "processed"
	/ "train_eda_processed.zip"
)
PROCESSED_TEST_CSV = (
	PROJECT_ROOT
	/ "data"
	/ "data_folder"
	/ "test"
	/ "processed"
	/ "test_eda_processed.csv"
)


def load_processed_data(data_path: Path) -> pd.DataFrame:
	"""Pull a DVC-tracked CSV or ZIP and load its data into a DataFrame."""
	dvc_file = Path(f"{data_path}.dvc")
	if not data_path.exists():
		if not dvc_file.exists():
			raise FileNotFoundError(f"DVC metadata file not found: {dvc_file}")
		subprocess.run(
			["dvc", "pull", str(dvc_file.relative_to(PROJECT_ROOT))],
			cwd=PROJECT_ROOT,
			check=True,
		)

	if data_path.suffix != ".zip":
		return pd.read_csv(data_path)

	with zipfile.ZipFile(data_path) as archive:
		csv_files = [name for name in archive.namelist() if name.endswith(".csv")]
		if len(csv_files) != 1:
			raise ValueError(
				f"Expected exactly one CSV in {data_path.name}, found {csv_files}"
			)
		with archive.open(csv_files[0]) as csv_file:
			return pd.read_csv(csv_file)


train_df = load_processed_data(PROCESSED_ZIP)
test_df = load_processed_data(PROCESSED_TEST_CSV)
print(f"Loaded train_df: {len(train_df):,} rows and {len(train_df.columns):,} columns.")
print(f"Loaded test_df: {len(test_df):,} rows and {len(test_df.columns):,} columns.")
