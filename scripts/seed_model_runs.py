#!/usr/bin/env python
"""Seed model_runs with baseline metrics for local development and demo.

Inserts one row per model_type only if no row exists yet — safe to run
multiple times. Real training (train_catboost_estimate.py / train_catboost_risk.py)
overwrites these with actual MLflow run IDs and freshly computed metrics.

Usage:
    python seed_model_runs.py              # use defaults from README
    python seed_model_runs.py --force      # re-seed even if rows exist
"""
import argparse
import uuid

from config import CONFIG
from feedback import get_latest_model_run, init_db, insert_model_run

# Baseline metrics from README / test-set evaluation
ESTIMATE_DEFAULTS = {
    "r2": 0.144,
    "mae": 1.23,
    "rmse": 1.85,
    "overrun_recall": None,
    "fold_scores": [0.168, 0.162, 0.171, 0.159, 0.166],
}

RISK_DEFAULTS = {
    "r2": None,
    "mae": None,
    "rmse": None,
    "overrun_recall": 0.59,
    "fold_scores": [],
}


def seed(db_path: str, force: bool = False) -> None:
    init_db(db_path)

    for model_type, defaults in [("estimate", ESTIMATE_DEFAULTS), ("risk", RISK_DEFAULTS)]:
        existing = get_latest_model_run(db_path, model_type)
        if existing and not force:
            print(f"[{model_type}] already seeded (run_id={existing['run_id'][:7]}, trigger={existing['trigger']}) — skipping. Use --force to overwrite.")
            continue

        insert_model_run(
            db_path,
            run_id=str(uuid.uuid4()),
            trigger="seed",
            model_type=model_type,
            **defaults,
        )
        row = get_latest_model_run(db_path, model_type)
        print(f"[{model_type}] seeded → run_id={row['run_id'][:7]}, r2={row['r2']}, overrun_recall={row['overrun_recall']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-seed even if rows already exist")
    parser.add_argument("--db", default=CONFIG["feedback_db_path"], help="Path to feedback.db (default: %(default)s)")
    args = parser.parse_args()

    seed(args.db, force=args.force)
    print(f"\nDone. DB: {args.db}")
    print("Run `uvicorn app:app --port 8000` — /health and /metrics will show seeded values.")
    print("After real training, these rows are superseded automatically.")


if __name__ == "__main__":
    main()
