import os
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import CONFIG


def load_data():
    print(f"Loading embeddings from {CONFIG['embeddings_save_path']}")
    X_train_raw = np.load(f"{CONFIG['embeddings_save_path']}/train_X.npy")
    y_train_raw = np.load(f"{CONFIG['embeddings_save_path']}/train_y_est.npy")

    X_val = np.load(f"{CONFIG['embeddings_save_path']}/val_X.npy")
    y_val = np.load(f"{CONFIG['embeddings_save_path']}/val_y_est.npy")

    X_test = np.load(f"{CONFIG['embeddings_save_path']}/test_X.npy")
    y_test_raw = np.load(f"{CONFIG['embeddings_save_path']}/test_y_est.npy")

    lower_bound, upper_bound = np.percentile(y_train_raw, 1), np.percentile(y_train_raw, 99)
    print(f"Filtering train: keeping estimates between {lower_bound:.3f} and {upper_bound:.3f} FTE")

    train_mask = (y_train_raw >= lower_bound) & (y_train_raw <= upper_bound)

    X_train = X_train_raw[train_mask]
    y_train = y_train_raw[train_mask]
    print(f"Dropped {len(y_train_raw) - len(y_train)} outliers from train")

    print(f"Loaded features matrix. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test_raw


def train(X_train, y_train, X_val, y_val):
    print("CatBoost training")

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=9,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=42,
        task_type="CPU"
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=70,
        logging_level='Verbose'
    )
    return model


def evaluate(model, X_test, y_test_raw):
    print(f"Evaluating model on test data")

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, a_min=0.0, a_max=None)

    y_true = y_test_raw

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    errors_95_pct = np.percentile(abs_errors, 95)

    print("CatBoost regression evalution results:")
    print(f"  MAE:{mae:.3f} FTE")
    print(f"  RMSE: {rmse:.3f} FTE")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Median errors: {np.median(abs_errors):.3f} FTE")
    print(f"  95% of errors are below: {errors_95_pct:.3f} FTE")
    print(f"  Max errors: {np.max(abs_errors):.3f} FTE")

    print("\nAccuracy:")
    print(f"  ≤ 0.25 FTE (2h): {(np.mean(abs_errors <= 0.25) * 100):.2f}% задач")
    print(f"  ≤ 0.50 FTE (4h): {(np.mean(abs_errors <= 0.50) * 100):.2f}% задач")
    print(f"  ≤ 1.00 FTE: {(np.mean(abs_errors <= 1.00) * 100):.2f}% задач")
    print(f"  > 2.00 FTE: {(np.mean(abs_errors > 2.00) * 100):.2f}% задач")

def save_model(model, path="models/catboost_estimate_model.cbm"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)
    print(f"CatBoost model saved to: {path}")


def main():
    X_train, y_train, X_val, y_val, X_test, y_test_raw = load_data()
    model = train(X_train, y_train, X_val, y_val)
    evaluate(model, X_test, y_test_raw)
    save_model(model, CONFIG['catboost_estimate_model_save_path'])


if __name__ == '__main__':
    main()
