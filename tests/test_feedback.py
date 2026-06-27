import os
import pytest
from feedback import (
    init_db, insert_feedback, get_feedback_count,
    insert_model_run, get_latest_model_run,
)


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


def test_init_creates_both_tables(db):
    import sqlite3
    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert "feedback" in tables
    assert "model_runs" in tables


def test_insert_feedback_delta_pct(db):
    insert_feedback(
        db,
        summary="Fix login bug",
        region="MOSCOW",
        subsystem="Auth",
        commitments="Q3",
        predicted_days=1.5,
        actual_days=2.0,
    )
    assert get_feedback_count(db) == 1
    import sqlite3
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT delta_pct FROM feedback WHERE id=1").fetchone()
    conn.close()
    # (2.0 - 1.5) / 2.0 * 100 = 25.0
    assert abs(row[0] - 25.0) < 0.01


def test_model_run_roundtrip(db):
    insert_model_run(
        db,
        run_id="abc123def456",
        trigger="manual",
        model_type="estimate",
        r2=0.144,
        mae=1.23,
        rmse=1.85,
        overrun_recall=None,
        fold_scores=[0.14, 0.15, 0.13, 0.14, 0.15],
    )
    row = get_latest_model_run(db, model_type="estimate")
    assert row is not None
    assert row["run_id"] == "abc123def456"
    assert abs(row["r2"] - 0.144) < 0.001
    assert row["trigger"] == "manual"
    assert len(row["fold_scores"]) == 5
