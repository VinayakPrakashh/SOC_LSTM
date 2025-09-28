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


FEATURE_COLS = ['Voltage [V]', 'Current [A]', 'Temperature [degC]', 'Power [W]', 'CC_Capacity [Ah]']
LABEL_COL = 'SOC [-]'
DEFAULT_TEMPS = ['25degC', '0degC', 'n10degC', 'n20degC', '10degC', '40degC']


def load_data(directory: str, temperatures):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Data directory not found: {directory}")

    frames = []
    for temp_folder in os.listdir(directory):
        if temp_folder in temperatures:
            temp_path = os.path.join(directory, temp_folder)
            if not os.path.isdir(temp_path):
                continue
            for file in os.listdir(temp_path):
                if 'Charge' in file or 'Dis' in file:
                    continue  # Skip constant charge and discharge files
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(temp_path, file))
                    df['SourceFile'] = file

                    # Calculate power
                    df['Power [W]'] = df['Voltage [V]'] * df['Current [A]']

                    # Cumulative capacity (Ah) by integrating current over time
                    df['CC_Capacity [Ah]'] = (
                        df['Current [A]'] * df['Time [s]'].diff().fillna(0) / 3600
                    ).cumsum()

                    frames.append(df)
    if not frames:
        raise RuntimeError(
            f"No CSV files found for temperatures {temperatures} in {directory}."
        )
    return pd.concat(frames, ignore_index=True)


class BatteryDatasetLSTM(Dataset):
    def __init__(self, data_tensor, labels_tensor, sequence_length=20, filenames=None, times=None):
        self.sequence_length = sequence_length
        self.features = data_tensor
        self.labels = labels_tensor
        self.filenames = filenames
        self.times = times

    def __len__(self):
        return max(0, len(self.features) - self.sequence_length)

    def __getitem__(self, idx):
        idx_end = idx + self.sequence_length
        sequence = self.features[idx:idx_end]
        label = self.labels[idx_end - 1]
        filename = self.filenames[idx_end - 1] if self.filenames is not None else ""
        time = self.times[idx_end - 1] if self.times is not None else 0.0
        return sequence, label, filename, time

    def get_unique_filenames(self):
        return set(self.filenames) if self.filenames is not None else set()


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


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    preds, labels = [], []
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


def build_loaders(df: pd.DataFrame, sequence_length: int, batch_size: int, device):
    # Scale features (replicates notebook behavior: fit on full data)
    scaler = StandardScaler()
    df = df.copy()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    # Train/Val/Test split by filenames to avoid leakage
    unique_files = np.array(list(set(df['SourceFile'])))
    train_files, temp_files = train_test_split(unique_files, test_size=0.2, random_state=24)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=24)

    def filter_by_files(d, names):
        return d[d['SourceFile'].isin(names)]

    train_df = filter_by_files(df, train_files)
    val_df = filter_by_files(df, val_files)
    test_df = filter_by_files(df, test_files)

    def to_dataset(dframe):
        feats = torch.tensor(dframe[FEATURE_COLS].values, dtype=torch.float32, device=device)
        labs = torch.tensor(dframe[LABEL_COL].values, dtype=torch.float32, device=device)
        fn = dframe['SourceFile'].values
        ts = dframe['Time [s]'].values
        return BatteryDatasetLSTM(feats, labs, sequence_length, fn, ts)

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)
    test_ds = to_dataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Test SoC LSTM model (extracted from notebook)")
    parser.add_argument("--model_path", default="soc_lstm_model.pth", help="Path to saved model .pth file")
    parser.add_argument("--data_dir", default=os.path.join("..", "dataset", "LG_HG2_processed"), help="Processed dataset directory")
    parser.add_argument("--temps", nargs="*", default=DEFAULT_TEMPS, help="Temperature folders to include")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden_size", type=int, default=94, help="Hidden size used at training")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of LSTM layers used at training")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--no_plots", action="store_true", help="Disable plots")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from: {args.data_dir} (temps: {args.temps})")
    data = load_data(args.data_dir, args.temps)

    # Ensure required columns exist
    missing_cols = [c for c in FEATURE_COLS + [LABEL_COL, 'Time [s]', 'SourceFile'] if c not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in data: {missing_cols}")

    # Build loaders
    _, _, test_loader = build_loaders(data, args.seq_len, args.batch_size, device)

    # Build and load model
    input_size = len(FEATURE_COLS)
    model = SoCLSTM(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device).type(torch.float32)

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # Evaluate
    preds, labels, mse, mae = evaluate_model(model, test_loader, device)
    print(f"Mean Squared Error on Test Set: {mse:.6f}")
    print(f"Mean Absolute Error on Test Set: {mae:.6f}")

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
        plt.title('Predicted SOC vs True SOC (Test Set)')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
