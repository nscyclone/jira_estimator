import pandas as pd
import torch
from torch.utils.data import Dataset

class JiraDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        estimate = torch.tensor(row['estimate'], dtype=torch.float32)
        risk_level = torch.tensor(row['risk_level'], dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'estimate': estimate,
            'risk_level': risk_level
        }