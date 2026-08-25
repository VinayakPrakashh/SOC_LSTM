import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
                    continue
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(temp_path, file))
                    df['SourceFile'] = file
                    df['Power [W]'] = df['Voltage [V]'] * df['Current [A]']
                    df['CC_Capacity [Ah]'] = (
                        df['Current [A]'] * df['Time [s]'].diff().fillna(0) / 3600
                    ).cumsum()
                    frames.append(df)
    if not frames:
        raise RuntimeError(f"No CSV files found for temperatures {temperatures} in {directory}.")
    return pd.concat(frames, ignore_index=True)


def find_max_input_values(df: pd.DataFrame):
    """Find maximum values in the dataset (before normalization)"""
    print("\n" + "="*70)
    print("MAXIMUM VALUES IN DATASET (RAW DATA)")
    print("="*70)
    
    for col in FEATURE_COLS:
        max_val = df[col].max()
        min_val = df[col].min()
        abs_max = max(abs(max_val), abs(min_val))
        print(f"{col:25s}: Max={max_val:10.6f}, Min={min_val:10.6f}, Abs_Max={abs_max:10.6f}")
    
    print("="*70)
    
    # After standardization
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[FEATURE_COLS])
    scaled_df = pd.DataFrame(scaled_data, columns=FEATURE_COLS)
    
    print("\nMAXIMUM VALUES AFTER STANDARDIZATION (NORMALIZED)")
    print("="*70)
    for col in FEATURE_COLS:
        max_val = scaled_df[col].max()
        min_val = scaled_df[col].min()
        abs_max = max(abs(max_val), abs(min_val))
        print(f"{col:25s}: Max={max_val:10.6f}, Min={min_val:10.6f}, Abs_Max={abs_max:10.6f}")
    
    print("="*70)
    
    # Overall maximum across all features (normalized)
    overall_max = scaled_df.abs().max().max()
    print(f"\nOVERALL MAXIMUM ABSOLUTE VALUE (normalized): {overall_max:.6f}")
    print("="*70 + "\n")
    
    return overall_max


def main():
    print("="*70)
    print("FINDING MAXIMUM INPUT VALUES IN DATASET")
    print("="*70)
    
    # Configuration
    data_dir = "LG_HG2_processed"
    
    # Load data
    print(f"\nLoading data from: {data_dir}")
    data = load_data(data_dir, DEFAULT_TEMPS)
    print(f"Total samples: {len(data)}")
    
    # Find maximum values
    max_val = find_max_input_values(data)


if __name__ == "__main__":
    main()