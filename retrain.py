#!/usr/bin/env python
"""Feedback-triggered retraining for the effort estimator.

Usage: python retrain.py [--dry-run]

Checks if unused feedback count >= CONFIG['retrain_threshold'].
If so, augments the training data with feedback rows and retrains
the estimate model, logging a new MLflow run with trigger='feedback'.
Only the estimate model is retrained (risk model requires story_points).
"""
import argparse
import os
import pickle

import mlflow
import numpy as np
import pandas as pd

from config import CONFIG
from feature_engineering import compute_text_features
from feedback import (
    get_feedback_rows_for_training,
    get_unused_feedback_count,
    insert_model_run,
    mark_feedback_used,
)
from load_catboost_data import load_catboost_data
from train_catboost_estimate import evaluate_ensemble, train_cv


def build_feedback_features(rows: list) -> tuple:
    """Convert feedback DB rows into (X_df, y_array) matching load_catboost_data output."""
    pipeline_path = os.path.join(CONFIG['embeddings_save_path'], "bm25_lsa_pipeline.pkl")
    with open(pipeline_path, "rb") as f:
        artifacts = pickle.load(f)
    vectorizer = artifacts["vectorizer"]
    svd = artifacts["svd_transformer"]

    feature_rows = []
    y_vals = []
    for fb in rows:
        full_text = str(fb["summary"])
        structural = compute_text_features(full_text, "")
        sparse = vectorizer.transform([full_text])
        lsa = svd.transform(sparse).flatten().astype(np.float32)

        row = {f"emb_{i}": float(lsa[i]) for i in range(len(lsa))}
        row.update({k: float(v) for k, v in structural.items()})
        row["region"] = str(fb["region"] or "Unknown")
        row["subsystem"] = str(fb["subsystem"] or "Unknown")
        row["commitments"] = str(fb["commitments"] or "Unknown")
        feature_rows.append(row)
        y_vals.append(float(fb["actual_days"]))

    return pd.DataFrame(feature_rows), np.array(y_vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Check threshold but do not retrain")
    args = parser.parse_args()

    db_path = CONFIG["feedback_db_path"]
    threshold = CONFIG["retrain_threshold"]
    unused_count = get_unused_feedback_count(db_path)

    print(f"Unused feedback rows: {unused_count} / threshold: {threshold}")

    if unused_count < threshold:
        print("Not enough feedback to retrain. Exiting.")
        return

    if args.dry_run:
        print("--dry-run: would retrain now. Exiting.")
        return

    print(f"Threshold reached. Loading {unused_count} feedback rows...")
    feedback_rows = get_feedback_rows_for_training(db_path)
    feedback_ids = [r["id"] for r in feedback_rows]
    X_feedback, y_feedback = build_feedback_features(feedback_rows)

    print("Loading existing training data...")
    X_cv, y_cv, X_test, y_test_raw = load_catboost_data("est")

    X_cv_aug = pd.concat([X_cv, X_feedback], ignore_index=True)
    y_cv_aug = np.concatenate([y_cv, y_feedback])
    print(f"Augmented CV set: {len(X_cv_aug)} rows (+{len(X_feedback)} feedback)")

    mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
    mlflow.set_experiment(CONFIG["mlflow_experiment_name"])

    with mlflow.start_run(tags={"model_type": "estimate", "trigger": "feedback"}) as run:
        mlflow.log_params({
            "iterations": 15000,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 6.0,
            "n_folds": 5,
            "feedback_rows_added": len(X_feedback),
        })

        model_paths, cv_scores = train_cv(X_cv_aug, y_cv_aug, CONFIG["catboost_estimate_model_save_path"])
        r2, mae, rmse = evaluate_ensemble(model_paths, X_test, y_test_raw)

        for path in model_paths:
            mlflow.log_artifact(path)

        run_id = run.info.run_id

    insert_model_run(
        db_path,
        run_id=run_id,
        trigger="feedback",
        model_type="estimate",
        r2=r2,
        mae=mae,
        rmse=rmse,
        overrun_recall=None,
        fold_scores=[float(s) for s in cv_scores],
    )
    mark_feedback_used(db_path, feedback_ids)
    print(f"Retraining complete. MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()
