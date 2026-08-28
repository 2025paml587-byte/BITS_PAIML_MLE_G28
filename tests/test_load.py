import zipfile

import pandas as pd
import pytest

from src.data.load import load_dvc_tracked_csv


def test_loads_plain_csv_without_pulling(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv_path, index=False)

    result = load_dvc_tracked_csv(csv_path)

    assert list(result.columns) == ["a"]
    assert result["a"].tolist() == [1, 2]


def test_loads_single_csv_from_zip(tmp_path):
    csv_name = "inner.csv"
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(csv_name, "a,b\n1,2\n3,4\n")

    result = load_dvc_tracked_csv(zip_path)

    assert list(result.columns) == ["a", "b"]
    assert len(result) == 2


def test_raises_when_zip_has_no_csv(tmp_path):
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("readme.txt", "not a csv")

    with pytest.raises(ValueError, match="Expected exactly one CSV"):
        load_dvc_tracked_csv(zip_path)


def test_raises_when_zip_has_multiple_csvs(tmp_path):
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("a.csv", "x\n1\n")
        archive.writestr("b.csv", "y\n2\n")

    with pytest.raises(ValueError, match="Expected exactly one CSV"):
        load_dvc_tracked_csv(zip_path)


def test_raises_when_file_and_dvc_pointer_both_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError, match="DVC metadata file not found"):
        load_dvc_tracked_csv(missing_path)
