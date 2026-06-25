import pandas as pd
import numpy as np
from config import CONFIG

NUM_COLS = [
    'has_description', 'has_code_block',
    'is_dev_task', 'is_test_task', 'is_analysis_task',
    'text_len', 'word_count',
]
CAT_COLS = ['region', 'subsystem', 'commitments']


def _build_split_df(emb_array: np.ndarray, meta_df: pd.DataFrame) -> pd.DataFrame:
    emb_cols = [f'emb_{i}' for i in range(emb_array.shape[1])]
    df = pd.DataFrame(emb_array, columns=emb_cols)
    for col in NUM_COLS:
        df[col] = meta_df[col].astype(np.float32)
    for col in CAT_COLS:
        df[col] = meta_df[col].astype(str)
    return df


def load_catboost_data(target: str):
    """
    Load embeddings + CSV metadata for CatBoost training.

    target: 'est' (regression) or 'risk' (classification)
    Returns: (X_cv, y_cv, X_test, y_test)
    """
    base = CONFIG['embeddings_save_path']
    print(f"Loading embeddings from {base}")

    X_train_emb = np.load(f"{base}/train_X.npy")
    X_val_emb = np.load(f"{base}/val_X.npy")
    X_test_emb = np.load(f"{base}/test_X.npy")

    y_train = np.load(f"{base}/train_y_{target}.npy")
    y_val = np.load(f"{base}/val_y_{target}.npy")
    y_test = np.load(f"{base}/test_y_{target}.npy")

    train_df = pd.read_csv(CONFIG['train_path'], keep_default_na=False)
    val_df = pd.read_csv(CONFIG['val_path'], keep_default_na=False)
    test_df = pd.read_csv(CONFIG['test_path'], keep_default_na=False)

    df_train = _build_split_df(X_train_emb, train_df)
    df_val = _build_split_df(X_val_emb, val_df)
    df_test = _build_split_df(X_test_emb, test_df)

    X_cv = pd.concat([df_train, df_val], ignore_index=True)
    y_cv = np.concatenate([y_train, y_val])

    print(f"Feature matrix ready. CV: {X_cv.shape}, Test: {df_test.shape}")
    return X_cv, y_cv, df_test, y_test
