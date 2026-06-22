import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from config import CONFIG

def load_risk_data():
    print(f"Loading embeddings from {CONFIG['embeddings_save_path']}")
    X_train_emb = np.load(f"{CONFIG['embeddings_save_path']}/train_X.npy")
    y_train = np.load(f"{CONFIG['embeddings_save_path']}/train_y_risk.npy")

    X_val_emb = np.load(f"{CONFIG['embeddings_save_path']}/val_X.npy")
    y_val = np.load(f"{CONFIG['embeddings_save_path']}/val_y_risk.npy")

    X_test_emb = np.load(f"{CONFIG['embeddings_save_path']}/test_X.npy")
    y_test = np.load(f"{CONFIG['embeddings_save_path']}/test_y_risk.npy")

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
    y_cv_all = np.concatenate([y_train, y_val])

    print(f"Loaded features matrix. CV Pool: {X_cv_all.shape}, Test: {df_test.shape}")
    return X_cv_all, y_cv_all, df_test, y_test


def train_cv(X_cv, y_cv, base_path):
    print("CatBoost risk classification")

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
        y_train_fold = y_cv[train_idx]
        y_val_fold = y_cv[val_idx]

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

        model = CatBoostClassifier(
            iterations=4000,
            learning_rate=0.03,
            depth=6,
            loss_function='MultiClass',
            eval_metric='MultiClass',
            auto_class_weights='Balanced',
            random_seed=42 + fold,
            task_type="CPU",
            cat_features=cat_features_indices
        )

        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=100,
            logging_level='Verbose'
        )

        val_pred = model.predict(val_pool).flatten()
        fold_acc = accuracy_score(y_val_fold, val_pred)
        print(f"Fold {fold + 1} Validation Accuracy: {(fold_acc * 100):.2f}%")
        cv_scores.append(fold_acc)

        fold_path = os.path.join(base_dir, f"risk_fold_{fold}.cbm")
        model.save_model(fold_path)
        saved_model_paths.append(fold_path)

        del model
        del train_pool
        del val_pool

    print(f"\nMean CV Accuracy: {(np.mean(cv_scores) * 100):.2f}% (+/- {(np.std(cv_scores) * 100):.2f}%)")
    return saved_model_paths


def evaluate_ensemble(model_paths, X_test, y_test):
    print(f"Evaluating ensemble on test data")

    cat_features_indices = ['region', 'subsystem', 'commitments']

    test_pool = Pool(
        data=X_test,
        cat_features=cat_features_indices
    )

    probas_list = []
    for path in model_paths:
        model = CatBoostClassifier().load_model(path)
        probas_list.append(model.predict_proba(test_pool))
        del model

    mean_probas = np.mean(probas_list, axis=0)
    y_pred = np.argmax(mean_probas, axis=1)

    accuracy = accuracy_score(y_test, y_pred) * 100

    print("CatBoost multiclass evalution results:")
    print(f"Accuracy: {accuracy:.2f}%")

    print("Detailed risk level stats:")
    print(classification_report(y_test, y_pred,
                                target_names=['Low Risk (0)', 'Medium Risk (1)', 'High Risk (2)']))


def main():
    X_cv, y_cv, X_test, y_test = load_risk_data()
    model_paths = train_cv(X_cv, y_cv, CONFIG['catboost_risk_model_save_path'])
    evaluate_ensemble(model_paths, X_test, y_test)


if __name__ == '__main__':
    main()
