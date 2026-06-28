import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class JiraDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256, target: str = 'estimate'):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target = target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        if self.target == 'estimate':
            sample['estimate'] = torch.tensor(np.log1p(row['estimate']), dtype=torch.float32)
        else:
            sample['risk_level'] = torch.tensor(row['risk_level'], dtype=torch.long)
        return sample
