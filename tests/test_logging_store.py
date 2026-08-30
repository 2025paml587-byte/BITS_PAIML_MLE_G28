from src.monitoring.logging_store import (
    clear_all,
    get_batch_summary,
    get_recent_predictions,
    log_prediction,
)

SAMPLE_FIELDS = dict(
    vendor_id=1,
    passenger_count=1,
    pickup_datetime="2016-03-14T09:30:00",
    pickup_latitude=40.75,
    pickup_longitude=-73.98,
    dropoff_latitude=40.76,
    dropoff_longitude=-73.99,
    store_and_fwd_flag="N",
)


def test_log_prediction_computes_absolute_error_when_actual_known(tmp_path):
    db_path = tmp_path / "predictions.db"
    row = log_prediction(
        batch_label="baseline",
        model_used="gradient_boosting",
        predicted_seconds=500.0,
        actual_seconds=600.0,
        db_path=db_path,
        **SAMPLE_FIELDS,
    )
    assert row["absolute_error"] == 100.0


def test_log_prediction_leaves_error_none_without_actual(tmp_path):
    db_path = tmp_path / "predictions.db"
    row = log_prediction(
        batch_label="live",
        model_used="gradient_boosting",
        predicted_seconds=500.0,
        db_path=db_path,
        **SAMPLE_FIELDS,
    )
    assert row["actual_seconds"] is None
    assert row["absolute_error"] is None


def test_get_recent_predictions_returns_most_recent_first(tmp_path):
    db_path = tmp_path / "predictions.db"
    for predicted in (100.0, 200.0, 300.0):
        log_prediction(
            batch_label="live",
            model_used="gradient_boosting",
            predicted_seconds=predicted,
            db_path=db_path,
            **SAMPLE_FIELDS,
        )
    recent = get_recent_predictions(limit=2, db_path=db_path)
    assert len(recent) == 2
    assert recent[0]["predicted_seconds"] == 300.0
    assert recent[1]["predicted_seconds"] == 200.0


def test_get_batch_summary_aggregates_per_batch_label(tmp_path):
    db_path = tmp_path / "predictions.db"
    log_prediction(
        batch_label="baseline", model_used="m", predicted_seconds=100.0,
        actual_seconds=110.0, db_path=db_path, **SAMPLE_FIELDS,
    )
    log_prediction(
        batch_label="baseline", model_used="m", predicted_seconds=200.0,
        actual_seconds=180.0, db_path=db_path, **SAMPLE_FIELDS,
    )
    log_prediction(
        batch_label="festival_surge", model_used="m", predicted_seconds=100.0,
        actual_seconds=400.0, db_path=db_path, **SAMPLE_FIELDS,
    )

    summary = get_batch_summary(db_path=db_path)

    assert summary["baseline"]["count"] == 2
    assert summary["baseline"]["mae"] == 15.0  # mean(|100-110|, |200-180|) = mean(10, 20)
    assert summary["festival_surge"]["mae"] == 300.0


def test_clear_all_removes_every_row(tmp_path):
    db_path = tmp_path / "predictions.db"
    log_prediction(
        batch_label="live", model_used="m", predicted_seconds=100.0,
        db_path=db_path, **SAMPLE_FIELDS,
    )
    clear_all(db_path=db_path)
    assert get_recent_predictions(db_path=db_path) == []
    assert get_batch_summary(db_path=db_path) == {}
