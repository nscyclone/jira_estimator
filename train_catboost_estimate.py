import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import CONFIG


def load_data():
    print(f"Loading embeddings from {CONFIG['embeddings_save_path']}")
    X_train_emb = np.load(f"{CONFIG['embeddings_save_path']}/train_X.npy")
    y_train_raw = np.load(f"{CONFIG['embeddings_save_path']}/train_y_est.npy")

    X_val_emb = np.load(f"{CONFIG['embeddings_save_path']}/val_X.npy")
    y_val_raw = np.load(f"{CONFIG['embeddings_save_path']}/val_y_est.npy")

    X_test_emb = np.load(f"{CONFIG['embeddings_save_path']}/test_X.npy")
    y_test_raw = np.load(f"{CONFIG['embeddings_save_path']}/test_y_est.npy")

    train_df = pd.read_csv(CONFIG['train_path'], keep_default_na=False)
    val_df = pd.read_csv(CONFIG['val_path'], keep_default_na=False)
    test_df = pd.read_csv(CONFIG['test_path'], keep_default_na=False)

    num_cols = [
        'has_description', 'has_code_block',
        'is_dev_task', 'is_test_task', 'is_analysis_task',
        'text_len', 'word_count'
    ]
    cat_cols = ['region', 'subsystem', 'commitments']
    emb_cols = [f'emb_{i}' for i in range(768)]

    df_train_part = pd.DataFrame(X_train_emb, columns=emb_cols)
    for col in num_cols:
        df_train_part[col] = train_df[col].astype(np.float32)
    for col in cat_cols:
        df_train_part[col] = train_df[col].astype(str)

    df_val_part = pd.DataFrame(X_val_emb, columns=emb_cols)
    for col in num_cols:
        df_val_part[col] = val_df[col].astype(np.float32)
    for col in cat_cols:
        df_val_part[col] = val_df[col].astype(str)

    df_test = pd.DataFrame(X_test_emb, columns=emb_cols)
    for col in num_cols:
        df_test[col] = test_df[col].astype(np.float32)
    for col in cat_cols:
        df_test[col] = test_df[col].astype(str)

    X_cv_all = pd.concat([df_train_part, df_val_part], ignore_index=True)
    y_cv_all = np.concatenate([y_train_raw, y_val_raw])

    print(f"Loaded features matrix. CV Pool: {X_cv_all.shape}, Test: {df_test.shape}")
    return X_cv_all, y_cv_all, df_test, y_test_raw


def train_cv(X_cv, y_cv, base_path):
    print("CatBoost estimate regression")

    cat_features_indices = ['region', 'subsystem', 'commitments']
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

        train_pool = Pool(
            data=X_train_fold,
            label=y_train_fold,
            cat_features=cat_features_indices
        )

        val_pool = Pool(
            data=X_val_fold,
            label=y_val_fold,
            cat_features=cat_features_indices
        )

        model = CatBoostRegressor(
            iterations=15000,
            learning_rate=0.03,
            depth=8,
            loss_function='RMSE',
            l2_leaf_reg=6.0,
            eval_metric='RMSE',
            random_seed=42 + fold,
            task_type="CPU",
            cat_features=cat_features_indices
        )

        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=150,
            logging_level='Verbose'
        )

        val_pred_log = model.predict(val_pool)
        val_pred = np.expm1(val_pred_log)
        val_pred = np.clip(val_pred, a_min=0.0, a_max=None)

        fold_r2 = r2_score(y_val_fold_raw, val_pred)
        print(f"Fold {fold + 1} Validation R² Score: {fold_r2:.4f}")
        cv_scores.append(fold_r2)

        fold_path = os.path.join(base_dir, f"estimate_fold_{fold}.cbm")
        model.save_model(fold_path)
        saved_model_paths.append(fold_path)

        del model
        del train_pool
        del val_pool

    print(f"\nMean CV R² Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    return saved_model_paths


def evaluate_ensemble(model_paths, X_test, y_test_raw):
    print(f"Evaluating ensemble on test data (Target: Actual Logged Days)")

    cat_features_indices = ['region', 'subsystem', 'commitments']

    test_pool = Pool(
        data=X_test,
        cat_features=cat_features_indices
    )

    preds_log_list = []
    for path in model_paths:
        model = CatBoostRegressor().load_model(path)
        preds_log_list.append(model.predict(test_pool))
        del model

    mean_pred_log = np.mean(preds_log_list, axis=0)

    y_pred = np.expm1(mean_pred_log)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    y_true = y_test_raw

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    errors_95_pct = np.percentile(abs_errors, 95)

    print("CatBoost regression evaluation results (Logged Days in FTE):")
    print(f"  MAE:{mae:.3f} FTE")
    print(f"  RMSE: {rmse:.3f} FTE")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Median errors: {np.median(abs_errors):.3f} FTE")
    print(f"  95% of errors are below: {errors_95_pct:.3f} FTE")
    print(f"  Max errors: {np.max(abs_errors):.3f} FTE")

    print("\nAccuracy (Logged Days vs Predicted):")
    print(f"  ≤ 0.25 FTE (2h): {(np.mean(abs_errors <= 0.25) * 100):.2f}% задач")
    print(f"  ≤ 0.50 FTE (4h): {(np.mean(abs_errors <= 0.50) * 100):.2f}% задач")
    print(f"  ≤ 1.00 FTE: {(np.mean(abs_errors <= 1.00) * 100):.2f}% задач")
    print(f"  > 2.00 FTE: {(np.mean(abs_errors > 2.00) * 100):.2f}% задач")


def main():
    X_cv, y_cv, X_test, y_test_raw = load_data()
    model_paths = train_cv(X_cv, y_cv, CONFIG['catboost_estimate_model_save_path'])
    evaluate_ensemble(model_paths, X_test, y_test_raw)


if __name__ == '__main__':
    main()
