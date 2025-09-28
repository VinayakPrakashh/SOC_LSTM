import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

FEATURE_COLS = ['Voltage [V]', 'Current [A]', 'Temperature [degC]']
LABEL_COL = 'SOC [-]'
DEFAULT_TEMPS = ['25degC', '0degC', 'n10degC', 'n20degC', '10degC', '40degC']


class BatteryDatasetLSTM(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, sequence_length: int, filenames=None, times=None):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.filenames = filenames
        self.times = times

    def __len__(self):
        return max(0, len(self.features) - self.sequence_length)

    def __getitem__(self, idx):
        j = idx + self.sequence_length
        x = self.features[idx:j]
        y = self.labels[j - 1]
        filename = self.filenames[j - 1] if self.filenames is not None else ""
        time = self.times[j - 1] if self.times is not None else 0.0
        return x, y, filename, time


def load_data(directory: str, temps) -> pd.DataFrame:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Data directory not found: {directory}")

    frames = []
    for t in os.listdir(directory):
        if t in temps:
            tp = os.path.join(directory, t)
            if not os.path.isdir(tp):
                continue
            for f in os.listdir(tp):
                if 'Charge' in f or 'Dis' in f:
                    continue
                if f.endswith('.csv'):
                    df = pd.read_csv(os.path.join(tp, f))
                    df['SourceFile'] = f
                    # Derived columns present in data but not used in 3-feature set
                    df['Power [W]'] = df['Voltage [V]'] * df['Current [A]']
                    df['CC_Capacity [Ah]'] = (df['Current [A]'] * df['Time [s]'].diff().fillna(0) / 3600).cumsum()
                    frames.append(df)
    if not frames:
        raise RuntimeError(f"No CSV files found for {temps} in {directory}")
    return pd.concat(frames, ignore_index=True)


class SoCLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def build_splits(df: pd.DataFrame):
    unique_files = np.array(list(set(df['SourceFile'])))
    train_files, temp_files = train_test_split(unique_files, test_size=0.2, random_state=24)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=24)

    def filt(names):
        return df[df['SourceFile'].isin(names)]

    return filt(train_files), filt(val_files), filt(test_files)


def to_dataset(dframe: pd.DataFrame, scaler: StandardScaler, seq_len: int, device: torch.device) -> BatteryDatasetLSTM:
    feats_np = scaler.transform(dframe[FEATURE_COLS].values)
    feats = torch.tensor(feats_np, dtype=torch.float32, device=device)
    labs = torch.tensor(dframe[LABEL_COL].values, dtype=torch.float32, device=device)
    fn = dframe['SourceFile'].values
    ts = dframe['Time [s]'].values
    return BatteryDatasetLSTM(feats, labs, seq_len, fn, ts)


def train_one(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running = 0.0
    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))


def eval_one(model, loader, criterion, device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    running = 0.0
    preds, labels = [], []
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            out = model(x)
            loss = criterion(out, y)
            running += loss.item()
            preds.extend(out.cpu().view(-1).tolist())
            labels.extend(y.cpu().view(-1).tolist())
    return running / max(1, len(loader)), np.array(preds), np.array(labels)


def main():
    ap = argparse.ArgumentParser(description='Train/Test SoC LSTM with 3 features')
    ap.add_argument('--data_dir', default=os.path.join('..', 'dataset', 'LG_HG2_processed'))
    ap.add_argument('--temps', nargs='*', default=DEFAULT_TEMPS)
    ap.add_argument('--seq_len', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--hidden_size', type=int, default=94)
    ap.add_argument('--num_layers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3.5e-5)
    ap.add_argument('--model_out', default='soc_lstm_model_3feat.pth')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--no_plots', action='store_true')
    args = ap.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    print(f'Using device: {device}')

    # Load and prepare data
    df = load_data(args.data_dir, args.temps)
    # Fit scaler only on train split features to avoid leakage
    train_df, val_df, test_df = build_splits(df)

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values)

    train_ds = to_dataset(train_df, scaler, args.seq_len, device)
    val_ds = to_dataset(val_df, scaler, args.seq_len, device)
    test_ds = to_dataset(test_df, scaler, args.seq_len, device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = SoCLSTM(input_size=len(FEATURE_COLS), hidden_size=args.hidden_size, num_layers=args.num_layers).to(device).type(torch.float32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    best_val = float('inf')
    best_state: Dict[str, Any] = {}
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one(model, train_loader, criterion, optimizer, device)
        va_loss, _, _ = eval_one(model, val_loader, criterion, device)
        print(f'Epoch {epoch:03d}: train_loss={tr_loss:.6f} val_loss={va_loss:.6f}')
        if va_loss < best_val:
            best_val = va_loss
            best_state = {
                'model_state_dict': model.state_dict(),
                'feature_cols': FEATURE_COLS,
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'seq_len': args.seq_len,
            }

    # Load best state into model
    if best_state:
        model.load_state_dict(best_state['model_state_dict'])

    # Test
    te_loss, preds, labels = eval_one(model, test_loader, criterion, device)
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    print(f'Test MSE: {mse:.6f}\nTest MAE: {mae:.6f}')

    if not args.no_plots:
        plt.figure(figsize=(8, 8))
        plt.scatter(labels, preds, alpha=0.5)
        plt.xlabel('True SOC')
        plt.ylabel('Predicted SOC')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.plot([0, 1], [0, 1], color='red')
        plt.title('Predicted SOC vs True SOC (Test Set) - 3 features')
        plt.tight_layout()
        plt.show()

    # Save checkpoint
    out_path = args.model_out
    torch.save(best_state if best_state else {'model_state_dict': model.state_dict(), 'feature_cols': FEATURE_COLS}, out_path)
    print(f'Saved model checkpoint to: {out_path}')


if __name__ == '__main__':
    main()
