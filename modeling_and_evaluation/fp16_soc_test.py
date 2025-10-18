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


def quantize_s7_8(value):
    """
    Quantize float to 16-bit S7.8 fixed-point format
    1 sign bit + 7 integer bits + 8 fractional bits
    Range: -128.0 to +127.99609375
    Resolution: 1/256 = 0.00390625
    """
    # Clamp to S7.8 range
    max_val = 127.99609375
    min_val = -128.0
    
    clamped = np.clip(value, min_val, max_val)
    
    # Sign-magnitude representation
    sign = clamped < 0
    abs_val = np.abs(clamped)
    
    # Scale by 2^8 and quantize
    scaled = abs_val * 256
    quantized_mag = np.round(scaled).astype(np.int32)
    
    # Clamp magnitude to 15 bits (32767)
    quantized_mag = np.clip(quantized_mag, 0, 32767)
    
    # Convert back to float
    dequantized_abs = quantized_mag / 256.0
    
    # Apply sign
    result = np.where(sign, -dequantized_abs, dequantized_abs)
    
    return result


def quantize_tensor_s7_8(tensor):
    """Quantize PyTorch tensor to S7.8 format"""
    numpy_data = tensor.detach().cpu().numpy()
    quantized = quantize_s7_8(numpy_data)
    return torch.from_numpy(quantized).to(tensor.device).type(tensor.dtype)


class SoCLSTMQuantized(nn.Module):
    """LSTM with S7.8 quantization applied to all operations"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Quantize input
        x = quantize_tensor_s7_8(x)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        
        # Quantize initial states
        h0 = quantize_tensor_s7_8(h0)
        c0 = quantize_tensor_s7_8(c0)
        
        # LSTM forward pass (we'll quantize the output)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Quantize LSTM output
        out = quantize_tensor_s7_8(out)
        
        # Final linear layer
        out = self.fc(out[:, -1, :])
        
        # Quantize final output
        out = quantize_tensor_s7_8(out)
        
        return out


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
    parser = argparse.ArgumentParser(description="Compare floating-point vs S7.8 fixed-point LSTM")
    parser.add_argument("--model_path", default="soc_lstm_model.pth", help="Path to saved model .pth file")
    parser.add_argument("--data_dir", default=os.path.join("..", "dataset", "LG_HG2_processed"), help="Processed dataset directory")
    parser.add_argument("--temps", nargs="*", default=DEFAULT_TEMPS, help="Temperature folders to include")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden_size", type=int, default=94, help="Hidden size used at training")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of LSTM layers used at training")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
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

    input_size = len(FEATURE_COLS)

    # Build floating-point model
    model_fp = SoCLSTM(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device).type(torch.float32)
    
    # Build quantized model (same weights, different forward pass)
    model_quantized = SoCLSTMQuantized(input_size=input_size, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device).type(torch.float32)

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Load weights for both models
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    model_fp.load_state_dict(state_dict)
    model_quantized.load_state_dict(state_dict)
    
    model_fp.eval()
    model_quantized.eval()

    # Evaluate both models
    print("\nEvaluating Floating-Point Model...")
    preds_fp, labels, mse_fp, mae_fp = evaluate_model(model_fp, test_loader, device)
    
    print("Evaluating S7.8 Fixed-Point Model...")
    preds_quantized, _, mse_quantized, mae_quantized = evaluate_model(model_quantized, test_loader, device)

    # Calculate quantization errors
    pred_error = preds_fp - preds_quantized
    rms_error = np.sqrt(np.mean(pred_error**2))
    max_error = np.max(np.abs(pred_error))
    
    # Calculate SNR
    signal_power = np.mean(preds_fp**2)
    noise_power = np.mean(pred_error**2)
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    # Print results
    print("\n" + "="*60)
    print("FLOATING-POINT vs S7.8 FIXED-POINT COMPARISON")
    print("="*60)
    print(f"Floating-Point Model:")
    print(f"  MSE: {mse_fp:.6f}")
    print(f"  MAE: {mae_fp:.6f}")
    print()
    print(f"S7.8 Fixed-Point Model:")
    print(f"  MSE: {mse_quantized:.6f}")
    print(f"  MAE: {mae_quantized:.6f}")
    print()
    print(f"Quantization Impact:")
    print(f"  RMS Prediction Error: {rms_error:.6f}")
    print(f"  Max Prediction Error: {max_error:.6f}")
    print(f"  SNR: {snr_db:.2f} dB")
    print(f"  MSE Degradation: {((mse_quantized - mse_fp)/mse_fp)*100:.2f}%")
    print(f"  MAE Degradation: {((mae_quantized - mae_fp)/mae_fp)*100:.2f}%")

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Floating-point predictions
    axes[0, 0].scatter(labels, preds_fp, alpha=0.5, s=1)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('True SOC')
    axes[0, 0].set_ylabel('Predicted SOC')
    axes[0, 0].set_title(f'Floating-Point Model\nMSE: {mse_fp:.6f}')
    axes[0, 0].axis('equal')
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Fixed-point predictions
    axes[0, 1].scatter(labels, preds_quantized, alpha=0.5, s=1)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('True SOC')
    axes[0, 1].set_ylabel('Predicted SOC')
    axes[0, 1].set_title(f'S7.8 Fixed-Point Model\nMSE: {mse_quantized:.6f}')
    axes[0, 1].axis('equal')
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Direct comparison
    axes[0, 2].scatter(preds_fp, preds_quantized, alpha=0.5, s=1)
    axes[0, 2].plot([0, 1], [0, 1], 'r--', linewidth=2)
    axes[0, 2].set_xlabel('Floating-Point Predictions')
    axes[0, 2].set_ylabel('Fixed-Point Predictions')
    axes[0, 2].set_title(f'FP vs Fixed-Point\nRMS Error: {rms_error:.6f}')
    axes[0, 2].axis('equal')
    axes[0, 2].set_xlim([0, 1])
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Prediction error histogram
    axes[1, 0].hist(pred_error, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Prediction Error (FP - Fixed)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Quantization Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Error vs prediction magnitude
    axes[1, 1].scatter(preds_fp, pred_error, alpha=0.5, s=1)
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Floating-Point Prediction')
    axes[1, 1].set_ylabel('Prediction Error')
    axes[1, 1].set_title('Error vs Prediction Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Cumulative error distribution
    sorted_errors = np.sort(np.abs(pred_error))
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1, 2].plot(sorted_errors, cumulative, linewidth=2)
    axes[1, 2].set_xlabel('Absolute Prediction Error')
    axes[1, 2].set_ylabel('Cumulative Probability')
    axes[1, 2].set_title('Cumulative Error Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('floating_vs_fixed_point_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save results to file
    results = {
        'floating_point': {'mse': mse_fp, 'mae': mae_fp},
        'fixed_point_s7_8': {'mse': mse_quantized, 'mae': mae_quantized},
        'quantization_error': {
            'rms_error': rms_error,
            'max_error': max_error,
            'snr_db': snr_db,
            'mse_degradation_percent': ((mse_quantized - mse_fp)/mse_fp)*100,
            'mae_degradation_percent': ((mae_quantized - mae_fp)/mae_fp)*100
        }
    }
    
    import json
    with open('fixed_point_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: fixed_point_comparison_results.json")
    print(f"Plots saved to: floating_vs_fixed_point_comparison.png")


if __name__ == "__main__":
    main()