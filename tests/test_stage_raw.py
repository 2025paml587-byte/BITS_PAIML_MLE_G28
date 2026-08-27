import pytest

from src.data.stage_raw import stage_file


def test_stage_file_copies_when_destination_missing(tmp_path):
    source = tmp_path / "source.zip"
    source.write_bytes(b"fake zip contents")
    destination = tmp_path / "raw" / "source.zip"

    stage_file(source, destination)

    assert destination.read_bytes() == b"fake zip contents"


def test_stage_file_skips_copy_when_already_staged(tmp_path):
    source = tmp_path / "source.zip"
    source.write_bytes(b"new contents")
    destination = tmp_path / "raw" / "source.zip"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"already staged contents")

    stage_file(source, destination)

    assert destination.read_bytes() == b"already staged contents"


def test_stage_file_raises_when_neither_source_nor_destination_exists(tmp_path):
    source = tmp_path / "missing_source.zip"
    destination = tmp_path / "raw" / "missing_source.zip"

    with pytest.raises(FileNotFoundError, match="Fetch the raw data first"):
        stage_file(source, destination)
