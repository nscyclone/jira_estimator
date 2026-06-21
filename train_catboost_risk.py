import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
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

    cat_cols = ['region', 'subsystem', 'commitments']
    cat_train = train_df[cat_cols].astype(str).to_numpy()
    cat_val = val_df[cat_cols].astype(str).to_numpy()
    cat_test = test_df[cat_cols].astype(str).to_numpy()

    X_train = np.hstack([X_train_emb, cat_train]).astype(object)
    X_val = np.hstack([X_val_emb, cat_val]).astype(object)
    X_test = np.hstack([X_test_emb, cat_test]).astype(object)

    print(f"Loaded features matrix. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def train(X_train, y_train, X_val, y_val):
    print("CatBoost training")

    cat_features_indices = [768, 769, 770]

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        auto_class_weights='Balanced',
        random_seed=42,
        task_type="CPU",
        cat_features=cat_features_indices
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
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
    print(classification_report(y_test, y_pred,
                                target_names=['Low Risk (0)', 'Medium Risk (1)', 'High Risk (2)']))


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
