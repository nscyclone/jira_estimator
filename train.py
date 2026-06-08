import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoTokenizer

from load_data import JiraDataset
from model import MultiTaskRuBERT
from config import CONFIG

def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f'Loading tokenizer for: {CONFIG['model_name']}')
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    print(f'Reading the training and validation datasets from {CONFIG['train_path']} and {CONFIG['val_path']}')
    train_df, val_df = pd.read_csv(CONFIG['train_path']), pd.read_csv(CONFIG['val_path'])

    train_dataset, val_dataset = JiraDataset(df=train_df, tokenizer=tokenizer, max_length=CONFIG['max_length']), JiraDataset(df=val_df, tokenizer=tokenizer, max_length=CONFIG['max_length'])

    train_loader, val_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True), DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    model = MultiTaskRuBERT(
        model_name=CONFIG['model_name'],
        num_risk_classes=CONFIG['num_risk_classes']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()

    print(f'Starting training. Total epochs: {CONFIG['epochs']}')

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_estimate = batch['estimate'].to(device)
            target_risk = batch['risk_level'].to(device)

            optimizer.zero_grad()

            pred_estimate, pred_risk = model(input_ids, attention_mask)

            loss = criterion_reg(pred_estimate, target_estimate) + criterion_cls(pred_risk, target_risk)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0
        total_abs_error = 0.0
        correct_risk_preds = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                target_estimate = batch['estimate'].to(device)
                target_risk = batch['risk_level'].to(device)

                pred_estimate, pred_risk = model(input_ids, attention_mask)
                total_val_loss += (criterion_reg(pred_estimate, target_estimate) + criterion_cls(pred_risk, target_risk)).item()

                abs_error = torch.abs(pred_estimate - target_estimate)
                total_abs_error += abs_error.sum().item()

                pred_classes = torch.argmax(pred_risk, dim=1)
                correct_risk_preds += (pred_classes == target_risk).sum().item()

                total_samples += len(pred_classes)

            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            final_mae = total_abs_error / total_samples
            final_accuracy = (correct_risk_preds / total_samples) * 100

            print(f'Epoch {epoch+1}/{CONFIG['epochs']}')
            print(f'Train loss: {avg_train_loss:.4f} | Validation loss: {avg_val_loss:.4f}')
            print(f'Risk class accuracy: {final_accuracy:.2f}%')
            print(f'FTE MAE: {final_mae:.3f} FTE')

        os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
        torch.save(model.state_dict(), CONFIG['model_save_path'])

    print(f'Training has finished, model has been saved to {CONFIG["model_save_path"]}')

if __name__ == '__main__':
    train()