import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from config import CONFIG


def main():
    print("Loading splitted CSV files...")
    train_df = pd.read_csv(CONFIG['train_path'], keep_default_na=False)
    val_df = pd.read_csv(CONFIG['val_path'], keep_default_na=False)
    test_df = pd.read_csv(CONFIG['test_path'], keep_default_na=False)

    print("Fitting industrial fast TF-IDF matrix (BM25 weighting logic)...")
    # sublinear_tf=True applies logarithmic scaling (1 + log(tf)), mirroring Okapi BM25's saturation curve.
    # min_df=2 drops single unique typos to shrink vocabulary and eliminate noise.
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        max_features=25000,
        stop_words=None
    )

    X_train_sparse = vectorizer.fit_transform(train_df['text'])
    X_val_sparse = vectorizer.transform(val_df['text'])
    X_test_sparse = vectorizer.transform(test_df['text'])

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} terms")

    print("Reducing dimensionality via TruncatedSVD (LSA) to 32 components...")
    svd = TruncatedSVD(n_components=32, random_state=42)
    X_train_lsa = svd.fit_transform(X_train_sparse).astype(np.float32)
    X_val_lsa = svd.transform(X_val_sparse).astype(np.float32)
    X_test_lsa = svd.transform(X_test_sparse).astype(np.float32)

    print(f"Explained variance ratio total: {np.sum(svd.explained_variance_ratio_):.4f}")

    os.makedirs(CONFIG['embeddings_save_path'], exist_ok=True)

    np.save(f"{CONFIG['embeddings_save_path']}/train_X.npy", X_train_lsa)
    np.save(f"{CONFIG['embeddings_save_path']}/val_X.npy", X_val_lsa)
    np.save(f"{CONFIG['embeddings_save_path']}/test_X.npy", X_test_lsa)

    artifacts = {
        "vectorizer": vectorizer,
        "svd_transformer": svd
    }
    with open(f"{CONFIG['embeddings_save_path']}/bm25_lsa_pipeline.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Processing complete. Files successfully saved to {CONFIG['embeddings_save_path']}")


if __name__ == '__main__':
    main()
