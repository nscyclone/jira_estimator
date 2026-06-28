import os
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

from dataset import JiraDataset
from estimate_model import EstimateRuBERT
from config import CONFIG


def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f"Loading tokenizer for: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    print(f"Reading the training dataset from {CONFIG['train_path']}")
    train_df = pd.read_csv(CONFIG['train_path'])
    train_dataset = JiraDataset(df=train_df, tokenizer=tokenizer, max_length=CONFIG['max_length'], target='estimate')
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    val_df = pd.read_csv(CONFIG['val_path'])
    val_dataset = JiraDataset(df=val_df, tokenizer=tokenizer, max_length=CONFIG['max_length'], target='estimate')
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    model = EstimateRuBERT(model_name=CONFIG['model_name']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion_reg = nn.MSELoss()

    print(f"Starting training. Total epochs: {CONFIG['epochs']}")
    os.makedirs(os.path.dirname(CONFIG['model_save_path']), exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_estimate = batch['estimate'].to(device)

            optimizer.zero_grad()
            pred_estimate = model(input_ids, attention_mask)
            loss = criterion_reg(pred_estimate, target_estimate)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            criterion_reg=criterion_reg,
            avg_train_loss=avg_train_loss,
            epoch=epoch,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CONFIG['model_save_path'])

    print(f"Training has finished, best model saved to {CONFIG['model_save_path']}")


def validate(model: nn.Module, val_loader: DataLoader, device, criterion_reg, avg_train_loss, epoch) -> float:
    model.eval()
    total_val_loss = 0
    total_abs_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_estimate = batch['estimate'].to(device)

            pred_estimate = model(input_ids, attention_mask)
            total_val_loss += criterion_reg(pred_estimate, target_estimate).item()

            pred_days = np.expm1(pred_estimate.cpu().numpy())
            target_days = np.expm1(target_estimate.cpu().numpy())
            total_abs_error += np.abs(pred_days - target_days).sum().item()
            total_samples += len(pred_estimate)

    avg_val_loss = total_val_loss / len(val_loader)
    final_mae = total_abs_error / total_samples

    print(f"Epoch {epoch + 1}/{CONFIG['epochs']}")
    print(f"Train loss: {avg_train_loss:.4f} | Validation loss: {avg_val_loss:.4f}")
    print(f"FTE MAE: {final_mae:.3f} FTE")

    return avg_val_loss


def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    print(f"Reading the test dataset from {CONFIG['test_path']}")
    test_df = pd.read_csv(CONFIG["test_path"])
    test_dataset = JiraDataset(df=test_df, tokenizer=tokenizer, max_length=CONFIG["max_length"], target='estimate')
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    model = EstimateRuBERT(model_name=CONFIG["model_name"]).to(device)

    if not os.path.exists(CONFIG["model_save_path"]):
        print(f"Weights file not found at {CONFIG['model_save_path']}. Train first.")
        return

    model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=device, weights_only=True))
    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["estimate"].cpu().numpy()
            preds = model(input_ids, attention_mask).cpu().numpy()

            all_targets.extend(np.expm1(targets))
            all_predictions.extend(np.clip(np.expm1(preds), a_min=0.0, a_max=None))

    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    std_error = np.std(errors)

    print(f"Final metrics for {CONFIG['model_name']}:")
    print(f"Tasks tested:   {len(y_true)}")
    print(f"MAE:    {mae:.3f} FTE")
    print(f"RMSE:  {rmse:.3f} FTE")
    print(f"STD: {std_error:.3f} FTE")
    print(f"R² Score: {r2:.4f}")

    print("\nStats for absolute errors:")
    print(f"  Median error:           {np.median(abs_errors):.3f} FTE")
    print(f"  Max absolute error: {np.max(abs_errors):.3f} FTE")

    print("\nAccuracy:")
    print(f"  ≤ 0.25 FTE (2h):   {(np.mean(abs_errors <= 0.25) * 100):.2f}%")
    print(f"  ≤ 0.50 FTE (4h):   {(np.mean(abs_errors <= 0.50) * 100):.2f}%")
    print(f"  ≤ 1.00 FTE:   {(np.mean(abs_errors <= 1.00) * 100):.2f}%")
    print(f"  > 2.00 FTE: {(np.mean(abs_errors > 2.00) * 100):.2f}%")


if __name__ == '__main__':
    train()
    evaluate()
    print('Done')
