#!/usr/bin/env python
"""Error analysis: per-subsystem and per-region MAE for the CatBoost ensemble.

Loads the trained models and test set, runs inference, and produces a
ranked table of where the model makes its largest errors. Useful for:
  - Understanding model limitations
  - Identifying data quality issues by subsystem
  - Presentation: "here's where and why the model struggles"

Usage:
    python analyze_errors.py
    python analyze_errors.py --test data/test.csv --top 10
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from config import CONFIG
from feature_engineering import compute_text_features


def load_models(model_dir: str = "models") -> list[CatBoostRegressor]:
    models = []
    for fold in range(5):
        path = os.path.join(model_dir, f"estimate_fold_{fold}.cbm")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path} — run train_catboost_estimate.py first.")
        m = CatBoostRegressor()
        m.load_model(path)
        models.append(m)
    return models


def build_features(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    cat_features = ["region", "subsystem", "commitments"]

    emb_list = []
    struct_list = []
    for _, row in df.iterrows():
        summary = str(row.get("summary", "") or "")
        description = str(row.get("description", "") or "")
        full_text = f"{summary} {description}".strip()

        emb = pipeline.transform([full_text])[0]
        struct = compute_text_features(full_text, description)
        emb_list.append(emb)
        struct_list.append(struct)

    emb_df = pd.DataFrame(emb_list, columns=[f"emb_{i}" for i in range(len(emb_list[0]))])
    struct_df = pd.DataFrame(struct_list)
    meta_df = df[cat_features].reset_index(drop=True).fillna("Unknown")

    return pd.concat([emb_df, struct_df, meta_df], axis=1)


def run_ensemble(models: list, X: pd.DataFrame) -> np.ndarray:
    cat_features = ["region", "subsystem", "commitments"]
    preds = []
    for m in models:
        pool = Pool(X, cat_features=cat_features)
        preds.append(m.predict(pool))
    return np.mean(preds, axis=0)


def analyze(test_path: str, top_n: int) -> None:
    test = pd.read_csv(test_path)
    if "logged_days" not in test.columns:
        raise ValueError("test.csv missing 'logged_days' — run prepare_data.py first.")

    pipeline_path = os.path.join(CONFIG["embeddings_save_path"], "bm25_lsa_pipeline.pkl")
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    models = load_models()
    X = build_features(test, pipeline)
    y_true_raw = test["logged_days"].values
    y_pred_log = run_ensemble(models, X)
    y_pred_raw = np.expm1(y_pred_log)

    abs_err = np.abs(y_true_raw - y_pred_raw)
    test = test.copy()
    test["abs_error_days"] = abs_err
    test["pred_days"] = y_pred_raw
    test["actual_days"] = y_true_raw

    print(f"\n=== Overall test MAE: {abs_err.mean():.3f} days ===\n")

    # ── Per subsystem ──────────────────────────────────────────────────────────
    if "subsystem" in test.columns:
        sub_stats = (
            test.groupby("subsystem")
            .agg(
                count=("abs_error_days", "count"),
                mae=("abs_error_days", "mean"),
                median_actual=("actual_days", "median"),
            )
            .sort_values("mae", ascending=False)
            .head(top_n)
            .reset_index()
        )
        sub_stats["mae"] = sub_stats["mae"].round(2)
        sub_stats["median_actual"] = sub_stats["median_actual"].round(2)
        print(f"=== Top-{top_n} Subsystems by MAE ===")
        print(sub_stats.to_string(index=False))

    # ── Per region ────────────────────────────────────────────────────────────
    if "region" in test.columns:
        reg_stats = (
            test.groupby("region")
            .agg(
                count=("abs_error_days", "count"),
                mae=("abs_error_days", "mean"),
                median_actual=("actual_days", "median"),
            )
            .sort_values("mae", ascending=False)
            .reset_index()
        )
        reg_stats["mae"] = reg_stats["mae"].round(2)
        reg_stats["median_actual"] = reg_stats["median_actual"].round(2)
        print("\n=== MAE by Region ===")
        print(reg_stats.to_string(index=False))

    # ── Worst individual predictions ──────────────────────────────────────────
    worst = (
        test[["summary", "subsystem", "region", "actual_days", "pred_days", "abs_error_days"]]
        .sort_values("abs_error_days", ascending=False)
        .head(10)
    )
    worst["abs_error_days"] = worst["abs_error_days"].round(1)
    worst["pred_days"] = worst["pred_days"].round(1)
    print("\n=== Top-10 Worst Individual Predictions ===")
    print(worst.to_string(index=False))

    print("\nInsight: large errors typically occur on tasks with hidden complexity")
    print("(cross-team dependencies, undefined scope, or mislabelled estimate).")
    print("The model cannot see what the ticket doesn't say.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--top", type=int, default=10, help="Number of subsystems to show")
    args = parser.parse_args()

    try:
        analyze(args.test, args.top)
    except FileNotFoundError as e:
        print(f"Missing file: {e}")
    except ValueError as e:
        print(f"Data error: {e}")


if __name__ == "__main__":
    main()
