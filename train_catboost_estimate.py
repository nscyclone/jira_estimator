import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from config import CONFIG
from feedback import insert_model_run
from load_catboost_data import load_catboost_data

CAT_FEATURE_COLS = ['region', 'subsystem', 'commitments']


def train_cv(X_cv, y_cv, base_path):
    print("CatBoost estimate regression")

    base_dir = os.path.dirname(base_path)
    os.makedirs(base_dir, exist_ok=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    saved_model_paths = []
    cv_scores = []
    last_X_val = None
    last_y_val_raw = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv, y_cv)):
        print(f"\n--- Training Fold {fold + 1}/5 ---")

        X_train_fold = X_cv.iloc[train_idx].copy()
        X_val_fold = X_cv.iloc[val_idx].copy()
        y_train_fold_raw = y_cv[train_idx]
        y_val_fold_raw = y_cv[val_idx]

        lower_bound = np.percentile(y_train_fold_raw, 1)
        upper_bound = np.percentile(y_train_fold_raw, 99)
        train_mask = (y_train_fold_raw >= lower_bound) & (y_train_fold_raw <= upper_bound)
        X_train_fold = X_train_fold.iloc[train_mask].reset_index(drop=True)
        y_train_fold_raw = y_train_fold_raw[train_mask]

        y_train_fold = np.log1p(y_train_fold_raw)
        y_val_fold = np.log1p(y_val_fold_raw)

        train_pool = Pool(data=X_train_fold, label=y_train_fold, cat_features=CAT_FEATURE_COLS)
        val_pool = Pool(data=X_val_fold, label=y_val_fold, cat_features=CAT_FEATURE_COLS)

        model = CatBoostRegressor(
            iterations=15000,
            learning_rate=0.03,
            depth=8,
            loss_function='RMSE',
            l2_leaf_reg=6.0,
            eval_metric='RMSE',
            random_seed=42 + fold,
            task_type="CPU",
            cat_features=CAT_FEATURE_COLS,
        )

        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=150, logging_level='Verbose')

        val_pred = np.clip(np.expm1(model.predict(val_pool)), a_min=0.0, a_max=None)
        fold_r2 = r2_score(y_val_fold_raw, val_pred)
        print(f"Fold {fold + 1} Validation R² Score: {fold_r2:.4f}")
        cv_scores.append(fold_r2)
        mlflow.log_metric(f"fold_{fold}_r2", fold_r2, step=fold)

        fold_path = os.path.join(base_dir, f"estimate_fold_{fold}.cbm")
        model.save_model(fold_path)
        saved_model_paths.append(fold_path)

        if fold == 4:
            last_X_val = X_val_fold.copy()
            last_y_val_raw = y_val_fold_raw.copy()

        del model, train_pool, val_pool

    mlflow.log_metric("cv_mean_r2", float(np.mean(cv_scores)))
    mlflow.log_metric("cv_std_r2", float(np.std(cv_scores)))
    print(f"\nMean CV R²: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # SHAP summary on last fold
    last_model = CatBoostRegressor().load_model(saved_model_paths[-1])
    last_pool = Pool(data=last_X_val, cat_features=CAT_FEATURE_COLS)
    shap_vals = last_model.get_feature_importance(last_pool, type='ShapValues')
    importances = np.abs(shap_vals[:, :-1]).mean(axis=0)
    feature_names = last_X_val.columns.tolist()
    idx = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 8))
    plt.barh([feature_names[i] for i in idx], importances[idx])
    plt.xlabel('Mean |SHAP value| (log-space)')
    plt.title('SHAP Feature Importance — Estimate Model (Fold 4)')
    plt.tight_layout()
    shap_path = 'shap_summary_estimate.png'
    plt.savefig(shap_path, dpi=150)
    plt.close()
    mlflow.log_artifact(shap_path)
    os.remove(shap_path)
    del last_model

    return saved_model_paths, cv_scores


def evaluate_ensemble(model_paths, X_test, y_test_raw):
    print("Evaluating ensemble on test data")

    test_pool = Pool(data=X_test, cat_features=CAT_FEATURE_COLS)
    preds_log_list = []
    for path in model_paths:
        model = CatBoostRegressor().load_model(path)
        preds_log_list.append(model.predict(test_pool))
        del model

    y_pred = np.clip(np.expm1(np.mean(preds_log_list, axis=0)), a_min=0.0, a_max=None)
    y_true = y_test_raw

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mae)

    abs_errors = np.abs(y_pred - y_true)
    print(f"  MAE: {mae:.3f} FTE | RMSE: {rmse:.3f} | R²: {r2:.4f}")
    print(f"  ≤ 0.50 FTE: {np.mean(abs_errors <= 0.50) * 100:.1f}% | > 2.00 FTE: {np.mean(abs_errors > 2.00) * 100:.1f}%")

    mlflow.log_metric("test_r2", r2)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_rmse", rmse)

    return r2, mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trigger', default='manual', choices=['manual', 'feedback'])
    args = parser.parse_args()

    mlflow.set_tracking_uri(CONFIG['mlflow_tracking_uri'])
    mlflow.set_experiment(CONFIG['mlflow_experiment_name'])

    with mlflow.start_run(tags={'model_type': 'estimate', 'trigger': args.trigger}) as run:
        mlflow.log_params({
            'iterations': 15000,
            'learning_rate': 0.03,
            'depth': 8,
            'l2_leaf_reg': 6.0,
            'n_folds': 5,
            'loss_function': 'RMSE',
        })

        X_cv, y_cv, X_test, y_test_raw = load_catboost_data('est')
        model_paths, cv_scores = train_cv(X_cv, y_cv, CONFIG['catboost_estimate_model_save_path'])
        r2, mae, rmse = evaluate_ensemble(model_paths, X_test, y_test_raw)

        for path in model_paths:
            mlflow.log_artifact(path)

        run_id = run.info.run_id

    insert_model_run(
        CONFIG['feedback_db_path'],
        run_id=run_id,
        trigger=args.trigger,
        model_type='estimate',
        r2=r2,
        mae=mae,
        rmse=rmse,
        overrun_recall=None,
        fold_scores=[float(s) for s in cv_scores],
    )
    print(f"MLflow run_id: {run_id}")


if __name__ == '__main__':
    main()
