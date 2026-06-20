import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from config import CONFIG


def extract_embeddings(csv_path, output_prefix):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModel.from_pretrained(CONFIG['model_name']).to(device)
    model.eval()

    print(f"Reading a dataset from {CONFIG['train_path']}")
    df = pd.read_csv(csv_path)
    embeddings_list = []

    batch_size = CONFIG['batch_size']
    for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {os.path.basename(csv_path)}"):
        batch_texts = df['text'].iloc[i:i + batch_size].astype(str).tolist()

        encodings = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=CONFIG['max_length'],
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings)

            # Get embedding of [CLS]-token
            hidden_states = outputs if isinstance(outputs, tuple) else outputs.last_hidden_state
            cls_embeddings = hidden_states[:, 0, :].cpu().numpy()

        embeddings_list.append(cls_embeddings)

    # Glue into a single matrix
    X = np.vstack(embeddings_list)

    # Targets
    y_estimate = df['logged_days'].values
    y_risk = df['risk_level'].values

    np.save(f"{CONFIG['embeddings_save_path']}/{output_prefix}_X.npy", X)
    np.save(f"{CONFIG['embeddings_save_path']}/{output_prefix}_y_est.npy", y_estimate)
    np.save(f"{CONFIG['embeddings_save_path']}/{output_prefix}_y_risk.npy", y_risk)
    print(f"Saved embeddings matrix shape: {X.shape}")


def main():
    os.makedirs(CONFIG['embeddings_save_path'], exist_ok=True)
    extract_embeddings(CONFIG['train_path'], 'train')
    extract_embeddings(CONFIG['val_path'], 'val')
    extract_embeddings(CONFIG['test_path'], 'test')
    print("Done.")


if __name__ == '__main__':
    main()