import os
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import CONFIG
from load_catboost_data import load_catboost_data

CAT_FEATURE_COLS = ['region', 'subsystem', 'commitments']


def train_cv(X_cv, y_cv, base_path):
    print("CatBoost estimate regression")

    base_dir = os.path.dirname(base_path)
    os.makedirs(base_dir, exist_ok=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    saved_model_paths = []
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv, y_cv)):
        print(f"\n--- Training Fold {fold + 1}/5 ---")

        X_train_fold = X_cv.iloc[train_idx].copy()
        X_val_fold = X_cv.iloc[val_idx].copy()
        y_train_fold_raw = y_cv[train_idx]
        y_val_fold_raw = y_cv[val_idx]

        lower_bound, upper_bound = np.percentile(y_train_fold_raw, 1), np.percentile(y_train_fold_raw, 99)
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

        fold_path = os.path.join(base_dir, f"estimate_fold_{fold}.cbm")
        model.save_model(fold_path)
        saved_model_paths.append(fold_path)

        del model, train_pool, val_pool

    print(f"\nMean CV R² Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    return saved_model_paths


def evaluate_ensemble(model_paths, X_test, y_test_raw):
    print("Evaluating ensemble on test data (Target: Actual Logged Days)")

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
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    abs_errors = np.abs(y_pred - y_true)
    errors_95_pct = np.percentile(abs_errors, 95)

    print("CatBoost regression evaluation results (Logged Days in FTE):")
    print(f"  MAE: {mae:.3f} FTE")
    print(f"  RMSE: {rmse:.3f} FTE")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Median errors: {np.median(abs_errors):.3f} FTE")
    print(f"  95% of errors are below: {errors_95_pct:.3f} FTE")
    print(f"  Max errors: {np.max(abs_errors):.3f} FTE")

    print("\nAccuracy (Logged Days vs Predicted):")
    print(f"  ≤ 0.25 FTE (2h): {(np.mean(abs_errors <= 0.25) * 100):.2f}%")
    print(f"  ≤ 0.50 FTE (4h): {(np.mean(abs_errors <= 0.50) * 100):.2f}%")
    print(f"  ≤ 1.00 FTE: {(np.mean(abs_errors <= 1.00) * 100):.2f}%")
    print(f"  > 2.00 FTE: {(np.mean(abs_errors > 2.00) * 100):.2f}%")


def main():
    X_cv, y_cv, X_test, y_test_raw = load_catboost_data('est')
    model_paths = train_cv(X_cv, y_cv, CONFIG['catboost_estimate_model_save_path'])
    evaluate_ensemble(model_paths, X_test, y_test_raw)


if __name__ == '__main__':
    main()
