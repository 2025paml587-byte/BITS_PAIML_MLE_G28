"""Stage raw data files into their DVC-tracked locations.

Replaces notebooks/pre-EDA-dvc.py: copies train.zip/test.csv into
data/data_folder/{train,test}/raw/ (their DVC-tracked home) and runs
`dvc add` on them. Idempotent - skips the copy if the file is already
staged there.
"""

import shutil
import subprocess
from pathlib import Path

from src.config import PROJECT_ROOT, TEST_RAW_PATH, TRAIN_RAW_PATH

# Where the raw files lived before DVC tracking was introduced. Staging
# copies them from here into their tracked location under raw/.
LEGACY_TRAIN_RAW_PATH = PROJECT_ROOT / "data" / "data_folder" / "train" / "train.zip"
LEGACY_TEST_RAW_PATH = PROJECT_ROOT / "data" / "data_folder" / "test" / "test.csv"


def stage_file(source: Path, destination: Path) -> None:
    """Copy source to destination unless destination is already there."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already staged: {destination}")
        return
    if not source.exists():
        raise FileNotFoundError(
            f"Cannot stage {destination.name}: neither {destination} nor "
            f"{source} exists. Fetch the raw data first."
        )
    shutil.copy2(source, destination)
    print(f"Copied {source} -> {destination}")


def dvc_add(path: Path) -> None:
    subprocess.run(
        ["dvc", "add", str(path.relative_to(PROJECT_ROOT))],
        cwd=PROJECT_ROOT,
        check=True,
    )


def stage_raw_data() -> None:
    if not (PROJECT_ROOT / ".dvc").is_dir():
        subprocess.run(["dvc", "init"], cwd=PROJECT_ROOT, check=True)

    stage_file(LEGACY_TRAIN_RAW_PATH, TRAIN_RAW_PATH)
    stage_file(LEGACY_TEST_RAW_PATH, TEST_RAW_PATH)

    dvc_add(TRAIN_RAW_PATH)
    dvc_add(TEST_RAW_PATH)

    print(
        "\nRaw data staged and added to DVC. Next steps:\n"
        "  git add data/data_folder/train/raw/train.zip.dvc "
        "data/data_folder/test/raw/test.csv.dvc\n"
        "  git commit -m 'Track raw data with DVC'\n"
        "  dvc push"
    )


if __name__ == "__main__":
    stage_raw_data()
