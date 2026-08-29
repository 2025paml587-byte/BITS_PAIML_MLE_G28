"""Persistence for the "high traffic zones" set used by the cleaned
feature schema (src.features.cleaned_features).

Split out from src.features.cleaning_pipeline so that importing it -
e.g. from src.api.inference, at serving time - doesn't drag in that
module's matplotlib/seaborn charting dependency. This module only
needs the standard library plus src.config.
"""

import json
from pathlib import Path

from src.config import HIGH_TRAFFIC_ZONES_PATH


def save_high_traffic_zones(zones: set, path: Path = HIGH_TRAFFIC_ZONES_PATH) -> None:
    """Persist the zones set so serving can reuse the exact same
    definition of "high traffic" the model was trained on, instead of
    guessing at inference time (a single request has no dataset to
    compute a frequency quantile from)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sorted(zones), f)


def load_high_traffic_zones(path: Path = HIGH_TRAFFIC_ZONES_PATH) -> set:
    """Load a previously persisted zones set, or an empty set if none
    has been saved yet."""
    if not path.exists():
        return set()
    with open(path, "r") as f:
        return set(json.load(f))
