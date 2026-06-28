import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional


def init_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at            TEXT    NOT NULL,
                summary               TEXT    NOT NULL,
                description           TEXT    DEFAULT '',
                region                TEXT,
                subsystem             TEXT,
                commitments           TEXT,
                predicted_days        REAL    NOT NULL,
                actual_days           REAL    NOT NULL,
                delta_pct             REAL,
                is_used_for_training  INTEGER DEFAULT 0
            )
        """)
        # Migrate existing DBs that predate the description column
        try:
            conn.execute("ALTER TABLE feedback ADD COLUMN description TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT    NOT NULL UNIQUE,
                created_at      TEXT    NOT NULL,
                trigger         TEXT    NOT NULL,
                model_type      TEXT    NOT NULL,
                r2              REAL,
                mae             REAL,
                rmse            REAL,
                overrun_recall  REAL,
                fold_scores     TEXT
            )
        """)
        conn.commit()


def insert_feedback(
    db_path: str,
    *,
    summary: str,
    description: str = "",
    region: str,
    subsystem: str,
    commitments: str,
    predicted_days: float,
    actual_days: float,
) -> int:
    delta_pct = (actual_days - predicted_days) / actual_days * 100 if actual_days else None
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO feedback
               (created_at, summary, description, region, subsystem, commitments,
                predicted_days, actual_days, delta_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                summary, description, region, subsystem, commitments,
                predicted_days, actual_days, delta_pct,
            ),
        )
        conn.commit()
        return cur.lastrowid or 0


def get_feedback_count(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]


def get_unused_feedback_count(db_path: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE is_used_for_training=0"
        ).fetchone()[0]


def get_feedback_rows_for_training(db_path: str) -> list:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT id, summary, description, region, subsystem, commitments, actual_days
               FROM feedback WHERE is_used_for_training=0"""
        ).fetchall()
    return [dict(r) for r in rows]


def mark_feedback_used(db_path: str, ids: list) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "UPDATE feedback SET is_used_for_training=1 WHERE id=?",
            [(i,) for i in ids],
        )
        conn.commit()


def get_feedback_metrics(db_path: str) -> dict:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT delta_pct FROM feedback"
        ).fetchall()
    if not rows:
        return {
            "feedbacks_collected": 0,
            "feedback_mean_delta_pct": None,
            "feedback_accuracy_within_25pct": None,
        }
    delta_pcts = [r[0] for r in rows if r[0] is not None]
    within_25 = [abs(d) <= 25.0 for d in delta_pcts]
    return {
        "feedbacks_collected": len(rows),
        "feedback_mean_delta_pct": round(sum(delta_pcts) / len(delta_pcts), 1) if delta_pcts else None,
        "feedback_accuracy_within_25pct": round(sum(within_25) / len(within_25), 3) if within_25 else None,
    }


def insert_model_run(
    db_path: str,
    *,
    run_id: str,
    trigger: str,
    model_type: str,
    r2: Optional[float],
    mae: Optional[float],
    rmse: Optional[float],
    overrun_recall: Optional[float],
    fold_scores: list,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO model_runs
               (run_id, created_at, trigger, model_type, r2, mae, rmse, overrun_recall, fold_scores)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                trigger, model_type, r2, mae, rmse, overrun_recall,
                json.dumps(fold_scores),
            ),
        )
        conn.commit()


def get_latest_model_run(db_path: str, model_type: str) -> Optional[dict]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """SELECT run_id, created_at, trigger, model_type, r2, mae, rmse, overrun_recall, fold_scores
               FROM model_runs
               WHERE model_type=?
               ORDER BY created_at DESC LIMIT 1""",
            (model_type,),
        ).fetchone()
    if not row:
        return None
    return {
        "run_id": row[0],
        "created_at": row[1],
        "trigger": row[2],
        "model_type": row[3],
        "r2": row[4],
        "mae": row[5],
        "rmse": row[6],
        "overrun_recall": row[7],
        "fold_scores": json.loads(row[8]) if row[8] else [],
    }
