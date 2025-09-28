import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import Optional, Tuple

DEFAULT_FEATURE_COLS = ['Voltage [V]', 'Current [A]', 'Temperature [degC]']
LABEL_COL = 'SOC [-]'
DEFAULT_TEMPS = ['25degC', '0degC', 'n10degC', 'n20degC', '10degC', '40degC']


class BatteryDatasetLSTM(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, seq_len: int, filenames=None, times=None):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len
        self.filenames = filenames
        self.times = times

    def __len__(self):
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx):
        j = idx + self.seq_len
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
                    # These columns may be in your dataset; compute if needed (not used in 3 features)
                    if 'Power [W]' not in df.columns:
                        df['Power [W]'] = df['Voltage [V]'] * df['Current [A]']
                    if 'CC_Capacity [Ah]' not in df.columns:
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


def make_scaler_from_ckpt(ckpt) -> Optional[StandardScaler]:
    mean = ckpt.get('scaler_mean')
    scale = ckpt.get('scaler_scale')
    if mean is None or scale is None:
        return None
    sc = StandardScaler()
    # Minimal init: set attributes so transform works
    sc.mean_ = np.array(mean)
    sc.scale_ = np.array(scale)
    sc.var_ = sc.scale_ ** 2  # not exact if with ddof, but sufficient
    sc.n_features_in_ = len(mean)
    return sc


def to_dataset(dframe: pd.DataFrame, scaler: StandardScaler, feature_cols, seq_len: int, device: torch.device) -> BatteryDatasetLSTM:
    feats_np = scaler.transform(dframe[feature_cols].values)
    feats = torch.tensor(feats_np, dtype=torch.float32, device=device)
    labs = torch.tensor(dframe[LABEL_COL].values, dtype=torch.float32, device=device)
    fn = dframe['SourceFile'].values
    ts = dframe['Time [s]'].values
    return BatteryDatasetLSTM(feats, labs, seq_len, fn, ts)


def evaluate(model, loader, device) -> Tuple[np.ndarray, np.ndarray, float, float]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.cpu().view(-1).tolist())
            labels.extend(y.cpu().view(-1).tolist())
    preds = np.array(preds)
    labels = np.array(labels)
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    return preds, labels, mse, mae


def per_file_timeseries(model, loader, device):
    model.eval()
    results = {}
    with torch.no_grad():
        for x, y, filenames, times in loader:
            x = x.to(device)
            out = model(x).cpu().view(-1).numpy()
            y = y.cpu().view(-1).numpy()
            for fn, t, p, l in zip(filenames, times, out, y):
                if fn not in results:
                    results[fn] = {'times': [], 'pred': [], 'label': []}
                results[fn]['times'].append(t)
                results[fn]['pred'].append(p)
                results[fn]['label'].append(l)
    return results


def main():
    ap = argparse.ArgumentParser(description='Test SoC LSTM (3 features) similar to notebook')
    ap.add_argument('--model_path', default='soc_lstm_model_3feat.pth')
    # If not provided, will be resolved relative to this script's directory
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--temps', nargs='*', default=DEFAULT_TEMPS)
    ap.add_argument('--seq_len', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--hidden_size', type=int, default=94)
    ap.add_argument('--num_layers', type=int, default=4)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--no_plots', action='store_true')
    args = ap.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    print(f'Using device: {device}')

    # Load checkpoint
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f'Model file not found: {args.model_path}')
    ckpt = torch.load(args.model_path, map_location=device)

    feature_cols = ckpt.get('feature_cols', DEFAULT_FEATURE_COLS)
    input_size = len(feature_cols)
    ckpt_hidden = ckpt.get('hidden_size', args.hidden_size)
    ckpt_layers = ckpt.get('num_layers', args.num_layers)
    ckpt_seq = ckpt.get('seq_len', args.seq_len)

    # Build model
    model = SoCLSTM(input_size=input_size, hidden_size=ckpt_hidden, num_layers=ckpt_layers).to(device).type(torch.float32)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()

    # Resolve dataset directory (default: ../dataset/LG_HG2_processed relative to this file)
    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.normpath(os.path.join(script_dir, '..', 'dataset', 'LG_HG2_processed'))
    else:
        data_dir = args.data_dir

    # Load and split data like notebook
    df = load_data(data_dir, args.temps)
    train_df, val_df, test_df = build_splits(df)

    # Prepare scaler
    scaler = make_scaler_from_ckpt(ckpt)
    if scaler is None:
        # Fallback: fit on train split to avoid leakage
        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols].values)
        print('Warning: checkpoint missing scaler stats; using train-split-fitted scaler.')

    # Datasets/loaders
    test_ds = to_dataset(test_df, scaler, feature_cols, ckpt_seq, device)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Evaluate
    preds, labels, mse, mae = evaluate(model, test_loader, device)
    print(f'Mean Squared Error on Test Set: {mse:.6f}')
    print(f'Mean Absolute Error on Test Set: {mae:.6f}')

    if not args.no_plots:
        # Scatter
        plt.figure(figsize=(8, 8))
        plt.scatter(labels, preds, alpha=0.5)
        plt.xlabel('True SOC')
        plt.ylabel('Predicted SOC')
        plt.axis('equal'); plt.axis('square')
        plt.xlim([0, 1]); plt.ylim([0, 1])
        plt.plot([0, 1], [0, 1], color='red')
        plt.title('Predicted SOC vs True SOC (Test Set) - 3 features')
        plt.tight_layout(); plt.show()

        # Per-file time series like in notebook
        results = per_file_timeseries(model, test_loader, device)
        for fn, data in results.items():
            plt.figure(figsize=(12, 6))
            plt.plot(data['times'], data['label'], label='True SOC', color='blue')
            plt.plot(data['times'], data['pred'], label='Predicted SOC', color='red')
            plt.title(f'Test File: {fn}')
            plt.xlabel('Time [s]'); plt.ylabel('SOC')
            plt.legend(); plt.tight_layout(); plt.show()


if __name__ == '__main__':
    main()
