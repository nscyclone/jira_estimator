import os
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from config import CONFIG

def load_risk_data():
    print(f"Loading embeddings from {CONFIG['embeddings_save_path']}")
    X_train = np.load(f"{CONFIG['embeddings_save_path']}/train_X.npy")
    y_train = np.load(f"{CONFIG['embeddings_save_path']}/train_y_risk.npy").astype(int)

    X_val = np.load(f"{CONFIG['embeddings_save_path']}/val_X.npy")
    y_val = np.load(f"{CONFIG['embeddings_save_path']}/val_y_risk.npy").astype(int)

    X_test = np.load(f"{CONFIG['embeddings_save_path']}/test_X.npy")
    y_test = np.load(f"{CONFIG['embeddings_save_path']}/test_y_risk.npy").astype(int)

    print(f"Loaded features matrix. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def train(X_train, y_train, X_val, y_val):
    print("CatBoost training")

    iterations = 1000
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        eval_metric='TotalF1:average=Macro',
        auto_class_weights='Balanced',
        random_seed=42,
        task_type="CPU"
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        logging_level='Verbose'
    )
    return model


def evaluate(model, X_test, y_test):
    print(f"Evaluating model on test data")

    y_pred = model.predict(X_test).flatten()

    accuracy = accuracy_score(y_test, y_pred) * 100

    print("CatBoost multiclass evalution results:")
    print(f"Accuracy: {accuracy:.2f}%")

    print("Detailed risk level stats:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=['Low (0)', 'Average (1)', 'High (2)', 'Critical (3)']
    ))


def save_model(model, path="models/catboost_risk_model.cbm"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)
    print(f"CatBoost model saved to: {path}")


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_risk_data()
    model = train(X_train, y_train, X_val, y_val)
    evaluate(model, X_test, y_test)
    save_model(model, CONFIG['catboost_risk_model_save_path'])


if __name__ == '__main__':
    main()
