"""SQLite-backed log of predictions (live and replayed), used to
compare predicted vs. actual trip duration and to detect drift.

Uses the stdlib sqlite3 module - no new dependency, and it's easily
queryable for the monitoring endpoints/UI.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.config import MONITORING_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL,
    batch_label TEXT NOT NULL,
    model_used TEXT NOT NULL,
    predicted_seconds REAL NOT NULL,
    actual_seconds REAL,
    absolute_error REAL,
    vendor_id INTEGER,
    passenger_count INTEGER,
    pickup_datetime TEXT,
    pickup_latitude REAL,
    pickup_longitude REAL,
    dropoff_latitude REAL,
    dropoff_longitude REAL,
    store_and_fwd_flag TEXT
);
"""


@contextmanager
def _connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_prediction(
    *,
    batch_label: str,
    model_used: str,
    predicted_seconds: float,
    vendor_id: int,
    passenger_count: int,
    pickup_datetime,
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    store_and_fwd_flag: str,
    actual_seconds: float | None = None,
    db_path: Path | None = None,
) -> dict:
    """Log one prediction. actual_seconds is None for a live prediction
    (no future ground truth yet) or a value when replaying historical/
    simulated trips whose real duration is already known."""
    db_path = db_path if db_path is not None else MONITORING_DB_PATH
    absolute_error = (
        abs(predicted_seconds - actual_seconds) if actual_seconds is not None else None
    )
    row = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "batch_label": batch_label,
        "model_used": model_used,
        "predicted_seconds": predicted_seconds,
        "actual_seconds": actual_seconds,
        "absolute_error": absolute_error,
        "vendor_id": vendor_id,
        "passenger_count": passenger_count,
        "pickup_datetime": str(pickup_datetime),
        "pickup_latitude": pickup_latitude,
        "pickup_longitude": pickup_longitude,
        "dropoff_latitude": dropoff_latitude,
        "dropoff_longitude": dropoff_longitude,
        "store_and_fwd_flag": store_and_fwd_flag,
    }
    with _connect(db_path) as conn:
        columns = ", ".join(row)
        placeholders = ", ".join(f":{key}" for key in row)
        conn.execute(f"INSERT INTO predictions ({columns}) VALUES ({placeholders})", row)
    return row


def get_recent_predictions(limit: int = 50, db_path: Path | None = None) -> list[dict]:
    db_path = db_path if db_path is not None else MONITORING_DB_PATH
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_batch_summary(db_path: Path | None = None) -> dict:
    """Per-batch_label count, average predicted/actual, and MAE (over
    rows where the actual value is known)."""
    db_path = db_path if db_path is not None else MONITORING_DB_PATH
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT batch_label,
                   COUNT(*) AS count,
                   AVG(predicted_seconds) AS avg_predicted,
                   AVG(actual_seconds) AS avg_actual,
                   AVG(absolute_error) AS mae
            FROM predictions
            GROUP BY batch_label
            """
        ).fetchall()
    return {row["batch_label"]: dict(row) for row in rows}


def clear_all(db_path: Path | None = None) -> None:
    """Wipe the log. Used by tests and to reset the drift demo."""
    db_path = db_path if db_path is not None else MONITORING_DB_PATH
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM predictions")
