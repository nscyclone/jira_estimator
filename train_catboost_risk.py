import os
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from config import CONFIG
from load_catboost_data import load_catboost_data

CAT_FEATURE_COLS = ['region', 'subsystem', 'commitments']


def train_cv(X_cv, y_cv, base_path):
    print("CatBoost risk classification")

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

        train_pool = Pool(data=X_train_fold, label=y_train_fold, cat_features=CAT_FEATURE_COLS)
        val_pool = Pool(data=X_val_fold, label=y_val_fold, cat_features=CAT_FEATURE_COLS)

        model = CatBoostClassifier(
            iterations=4000,
            learning_rate=0.03,
            depth=6,
            loss_function='MultiClass',
            eval_metric='MultiClass',
            auto_class_weights='Balanced',
            random_seed=42 + fold,
            task_type="CPU",
            cat_features=CAT_FEATURE_COLS,
        )

        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, logging_level='Verbose')

        val_pred = model.predict(val_pool).flatten()
        fold_acc = accuracy_score(y_val_fold, val_pred)
        print(f"Fold {fold + 1} Validation Accuracy: {(fold_acc * 100):.2f}%")
        cv_scores.append(fold_acc)

        fold_path = os.path.join(base_dir, f"risk_fold_{fold}.cbm")
        model.save_model(fold_path)
        saved_model_paths.append(fold_path)

        del model, train_pool, val_pool

    print(f"\nMean CV Accuracy: {(np.mean(cv_scores) * 100):.2f}% (+/- {(np.std(cv_scores) * 100):.2f}%)")
    return saved_model_paths


def evaluate_ensemble(model_paths, X_test, y_test):
    print("Evaluating ensemble on test data")

    test_pool = Pool(data=X_test, cat_features=CAT_FEATURE_COLS)

    probas_list = []
    for path in model_paths:
        model = CatBoostClassifier().load_model(path)
        probas_list.append(model.predict_proba(test_pool))
        del model

    y_pred = np.argmax(np.mean(probas_list, axis=0), axis=1)
    accuracy = accuracy_score(y_test, y_pred) * 100

    print("CatBoost multiclass evaluation results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print("Detailed risk level stats:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk (0)', 'Medium Risk (1)', 'High Risk (2)']))


def main():
    X_cv, y_cv, X_test, y_test = load_catboost_data('risk')
    model_paths = train_cv(X_cv, y_cv, CONFIG['catboost_risk_model_save_path'])
    evaluate_ensemble(model_paths, X_test, y_test)


if __name__ == '__main__':
    main()
