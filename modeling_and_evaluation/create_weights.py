import torch
import numpy as np
from pathlib import Path

def extract_all_lstm_weights_separately(pth_file='lstm_soc_model.pth', output_dir='lstm_weights_separated'):
    """
    Extract ALL LSTM weights and save them separately for each gate
    
    Creates separate files for:
    - Input gate: W_ih, W_hh, bias
    - Forget gate: W_ih, W_hh, bias
    - Cell gate: W_ih, W_hh, bias
    - Output gate: W_ih, W_hh, bias
    """
    
    print("="*80)
    print(" EXTRACTING ALL LSTM WEIGHTS SEPARATELY")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_path.absolute()}")
    
    # Load model
    print(f"\nLoading model from: {pth_file}")
    model_data = torch.load(pth_file, map_location='cpu')
    
    # Get state dict
    if isinstance(model_data, dict):
        if 'model_state_dict' in model_data:
            state_dict = model_data['model_state_dict']
        elif 'state_dict' in model_data:
            state_dict = model_data['state_dict']
        else:
            state_dict = model_data
    else:
        state_dict = model_data.state_dict() if hasattr(model_data, 'state_dict') else model_data
    
    print("\nFound layers:")
    for name, param in state_dict.items():
        print(f"  {name}: {tuple(param.shape)}")
    
    # Extract full weight matrices
    print("\n" + "="*80)
    print(" EXTRACTING FULL WEIGHT MATRICES")
    print("="*80)
    
    W_ih_full = state_dict['lstm.weight_ih_l0'].cpu().numpy()  # (376, 5)
    W_hh_full = state_dict['lstm.weight_hh_l0'].cpu().numpy()  # (376, 94)
    b_ih_full = state_dict['lstm.bias_ih_l0'].cpu().numpy()    # (376,)
    b_hh_full = state_dict['lstm.bias_hh_l0'].cpu().numpy()    # (376,)
    
    print(f"\nFull matrices extracted:")
    print(f"  W_ih: {W_ih_full.shape} (input-to-hidden)")
    print(f"  W_hh: {W_hh_full.shape} (hidden-to-hidden)")
    print(f"  b_ih: {b_ih_full.shape} (input bias)")
    print(f"  b_hh: {b_hh_full.shape} (hidden bias)")
    
    # Combine biases
    b_combined = b_ih_full + b_hh_full
    print(f"  b_combined: {b_combined.shape} (b_ih + b_hh)")
    
    # Gate configuration
    hidden_size = 94
    gate_names = ['input', 'forget', 'cell', 'output']
    
    print("\n" + "="*80)
    print(" SPLITTING INTO INDIVIDUAL GATES")
    print("="*80)
    
    # Dictionary to store all weights
    all_weights = {}
    
    for gate_idx, gate_name in enumerate(gate_names):
        start_row = gate_idx * hidden_size
        end_row = (gate_idx + 1) * hidden_size
        
        print(f"\n{gate_name.upper()} GATE (rows {start_row}:{end_row})")
        print("-" * 60)
        
        # Extract gate-specific weights
        W_ih_gate = W_ih_full[start_row:end_row, :]  # (94, 5)
        W_hh_gate = W_hh_full[start_row:end_row, :]  # (94, 94)
        b_ih_gate = b_ih_full[start_row:end_row]     # (94,)
        b_hh_gate = b_hh_full[start_row:end_row]     # (94,)
        b_combined_gate = b_combined[start_row:end_row]  # (94,)
        
        print(f"  W_ih_{gate_name}: {W_ih_gate.shape}")
        print(f"  W_hh_{gate_name}: {W_hh_gate.shape}")
        print(f"  b_ih_{gate_name}: {b_ih_gate.shape}")
        print(f"  b_hh_{gate_name}: {b_hh_gate.shape}")
        print(f"  b_combined_{gate_name}: {b_combined_gate.shape}")
        
        # Store in dictionary
        all_weights[gate_name] = {
            'W_ih': W_ih_gate,
            'W_hh': W_hh_gate,
            'b_ih': b_ih_gate,
            'b_hh': b_hh_gate,
            'b_combined': b_combined_gate
        }
        
        # Save individual files for this gate
        gate_dir = output_path / gate_name
        gate_dir.mkdir(exist_ok=True)
        
        # Save as numpy binary
        np.save(gate_dir / f'W_ih_{gate_name}_94x5.npy', W_ih_gate)
        np.save(gate_dir / f'W_hh_{gate_name}_94x94.npy', W_hh_gate)
        np.save(gate_dir / f'b_ih_{gate_name}_94.npy', b_ih_gate)
        np.save(gate_dir / f'b_hh_{gate_name}_94.npy', b_hh_gate)
        np.save(gate_dir / f'b_combined_{gate_name}_94.npy', b_combined_gate)
        
        # Save as text (comma-separated)
        np.savetxt(gate_dir / f'W_ih_{gate_name}_94x5.txt', W_ih_gate, fmt='%.8f', delimiter=',')
        np.savetxt(gate_dir / f'W_hh_{gate_name}_94x94.txt', W_hh_gate, fmt='%.8f', delimiter=',')
        np.savetxt(gate_dir / f'b_ih_{gate_name}_94.txt', b_ih_gate, fmt='%.8f')
        np.savetxt(gate_dir / f'b_hh_{gate_name}_94.txt', b_hh_gate, fmt='%.8f')
        np.savetxt(gate_dir / f'b_combined_{gate_name}_94.txt', b_combined_gate, fmt='%.8f')
        
        # Save concatenated [W_ih | W_hh | bias] for this gate (94 x 100)
        gate_matrix_100 = np.concatenate([W_ih_gate, W_hh_gate, b_combined_gate.reshape(-1, 1)], axis=1)
        np.save(gate_dir / f'{gate_name}_gate_94x100.npy', gate_matrix_100)
        np.savetxt(gate_dir / f'{gate_name}_gate_94x100.txt', gate_matrix_100, fmt='%.8f', delimiter=',')
        
        print(f"  ✓ Saved all files to: {gate_dir}")
    
    # Save full concatenated matrices
    print("\n" + "="*80)
    print(" SAVING FULL MATRICES")
    print("="*80)
    
    full_dir = output_path / 'full'
    full_dir.mkdir(exist_ok=True)
    
    # Save complete matrices
    np.save(full_dir / 'W_ih_full_376x5.npy', W_ih_full)
    np.save(full_dir / 'W_hh_full_376x94.npy', W_hh_full)
    np.save(full_dir / 'b_ih_full_376.npy', b_ih_full)
    np.save(full_dir / 'b_hh_full_376.npy', b_hh_full)
    np.save(full_dir / 'b_combined_full_376.npy', b_combined)
    
    np.savetxt(full_dir / 'W_ih_full_376x5.txt', W_ih_full, fmt='%.8f', delimiter=',')
    np.savetxt(full_dir / 'W_hh_full_376x94.txt', W_hh_full, fmt='%.8f', delimiter=',')
    np.savetxt(full_dir / 'b_ih_full_376.txt', b_ih_full, fmt='%.8f')
    np.savetxt(full_dir / 'b_hh_full_376.txt', b_hh_full, fmt='%.8f')
    np.savetxt(full_dir / 'b_combined_full_376.txt', b_combined, fmt='%.8f')
    
    # Create full 376x100 matrix
    matrix_376x100 = np.concatenate([W_ih_full, W_hh_full, b_combined.reshape(-1, 1)], axis=1)
    np.save(full_dir / 'W_all_376x100.npy', matrix_376x100)
    np.savetxt(full_dir / 'W_all_376x100.txt', matrix_376x100, fmt='%.8f', delimiter=',')
    
    print(f"✓ Saved full matrices to: {full_dir}")
    
    # Create comprehensive documentation
    create_documentation(output_path, all_weights, W_ih_full, W_hh_full, b_combined)
    
    # Create summary
    create_summary(output_path, all_weights)
    
    return all_weights

def create_documentation(output_path, all_weights, W_ih_full, W_hh_full, b_combined):
    """Create detailed documentation file"""
    
    doc_file = output_path / 'README.txt'
    
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" LSTM WEIGHTS - SEPARATED BY GATE\n")
        f.write("="*80 + "\n\n")
        
        f.write("DIRECTORY STRUCTURE:\n")
        f.write("-" * 80 + "\n\n")
        f.write("lstm_weights_separated/\n")
        f.write("├── input/                 # Input gate weights\n")
        f.write("│   ├── W_ih_input_94x5.npy       (input-to-hidden)\n")
        f.write("│   ├── W_hh_input_94x94.npy      (hidden-to-hidden)\n")
        f.write("│   ├── b_ih_input_94.npy         (input bias)\n")
        f.write("│   ├── b_hh_input_94.npy         (hidden bias)\n")
        f.write("│   ├── b_combined_input_94.npy   (combined bias)\n")
        f.write("│   ├── input_gate_94x100.npy     (all concatenated)\n")
        f.write("│   └── *.txt versions\n")
        f.write("│\n")
        f.write("├── forget/                # Forget gate weights\n")
        f.write("│   └── (same structure as input/)\n")
        f.write("│\n")
        f.write("├── cell/                  # Cell gate weights\n")
        f.write("│   └── (same structure as input/)\n")
        f.write("│\n")
        f.write("├── output/                # Output gate weights\n")
        f.write("│   └── (same structure as input/)\n")
        f.write("│\n")
        f.write("└── full/                  # Complete matrices (all 4 gates)\n")
        f.write("    ├── W_ih_full_376x5.npy\n")
        f.write("    ├── W_hh_full_376x94.npy\n")
        f.write("    ├── b_combined_full_376.npy\n")
        f.write("    ├── W_all_376x100.npy        (complete 376x100 matrix)\n")
        f.write("    └── *.txt versions\n\n")
        
        f.write("="*80 + "\n")
        f.write(" MATRIX DIMENSIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Per Gate (94 units per gate):\n")
        f.write("  W_ih:       94 x 5   (input-to-hidden weights)\n")
        f.write("  W_hh:       94 x 94  (hidden-to-hidden weights)\n")
        f.write("  b_ih:       94 x 1   (input bias)\n")
        f.write("  b_hh:       94 x 1   (hidden bias)\n")
        f.write("  b_combined: 94 x 1   (b_ih + b_hh)\n")
        f.write("  Concatenated: 94 x 100 ([W_ih | W_hh | bias])\n\n")
        
        f.write("Full (All 4 gates):\n")
        f.write("  W_ih:       376 x 5   (4 × 94 rows)\n")
        f.write("  W_hh:       376 x 94  (4 × 94 rows)\n")
        f.write("  b_combined: 376 x 1\n")
        f.write("  Complete:   376 x 100\n\n")
        
        f.write("="*80 + "\n")
        f.write(" GATE ORGANIZATION\n")
        f.write("="*80 + "\n\n")
        
        f.write("Rows in full matrices:\n")
        f.write("  [0:94]      - Input Gate\n")
        f.write("  [94:188]    - Forget Gate\n")
        f.write("  [188:282]   - Cell Gate\n")
        f.write("  [282:376]   - Output Gate\n\n")
        
        f.write("="*80 + "\n")
        f.write(" USAGE EXAMPLES\n")
        f.write("="*80 + "\n\n")
        
        f.write("Python - Load individual gate:\n")
        f.write("-" * 80 + "\n")
        f.write("import numpy as np\n\n")
        f.write("# Load input gate weights\n")
        f.write("W_ih_input = np.load('input/W_ih_input_94x5.npy')\n")
        f.write("W_hh_input = np.load('input/W_hh_input_94x94.npy')\n")
        f.write("bias_input = np.load('input/b_combined_input_94.npy')\n\n")
        f.write("# Or load the concatenated matrix\n")
        f.write("input_gate_all = np.load('input/input_gate_94x100.npy')\n\n\n")
        
        f.write("Python - Load full matrix:\n")
        f.write("-" * 80 + "\n")
        f.write("# Load complete 376x100 matrix\n")
        f.write("W_all = np.load('full/W_all_376x100.npy')\n\n")
        f.write("# Split by gates\n")
        f.write("input_gate = W_all[0:94, :]\n")
        f.write("forget_gate = W_all[94:188, :]\n")
        f.write("cell_gate = W_all[188:282, :]\n")
        f.write("output_gate = W_all[282:376, :]\n\n\n")
        
        f.write("="*80 + "\n")
        f.write(" STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        for gate_name, weights in all_weights.items():
            f.write(f"{gate_name.upper()} GATE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  W_ih range: [{weights['W_ih'].min():.6f}, {weights['W_ih'].max():.6f}]\n")
            f.write(f"  W_hh range: [{weights['W_hh'].min():.6f}, {weights['W_hh'].max():.6f}]\n")
            f.write(f"  bias range: [{weights['b_combined'].min():.6f}, {weights['b_combined'].max():.6f}]\n")
            f.write(f"  W_ih mean:  {weights['W_ih'].mean():.6f}\n")
            f.write(f"  W_hh mean:  {weights['W_hh'].mean():.6f}\n")
            f.write(f"  bias mean:  {weights['b_combined'].mean():.6f}\n\n")
        
        f.write("FULL MATRICES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  W_ih range: [{W_ih_full.min():.6f}, {W_ih_full.max():.6f}]\n")
        f.write(f"  W_hh range: [{W_hh_full.min():.6f}, {W_hh_full.max():.6f}]\n")
        f.write(f"  bias range: [{b_combined.min():.6f}, {b_combined.max():.6f}]\n")
    
    print(f"✓ Documentation created: {doc_file}")

def create_summary(output_path, all_weights):
    """Create visual summary"""
    
    summary_file = output_path / 'SUMMARY.txt'
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" FILE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        total_files = 0
        
        for gate_name in ['input', 'forget', 'cell', 'output']:
            f.write(f"{gate_name.upper()} GATE FILES:\n")
            f.write("-" * 80 + "\n")
            
            files = [
                f'W_ih_{gate_name}_94x5.npy',
                f'W_ih_{gate_name}_94x5.txt',
                f'W_hh_{gate_name}_94x94.npy',
                f'W_hh_{gate_name}_94x94.txt',
                f'b_ih_{gate_name}_94.npy',
                f'b_ih_{gate_name}_94.txt',
                f'b_hh_{gate_name}_94.npy',
                f'b_hh_{gate_name}_94.txt',
                f'b_combined_{gate_name}_94.npy',
                f'b_combined_{gate_name}_94.txt',
                f'{gate_name}_gate_94x100.npy',
                f'{gate_name}_gate_94x100.txt',
            ]
            
            for fname in files:
                f.write(f"  ✓ {gate_name}/{fname}\n")
                total_files += 1
            f.write("\n")
        
        f.write("FULL MATRICES:\n")
        f.write("-" * 80 + "\n")
        
        full_files = [
            'W_ih_full_376x5.npy',
            'W_ih_full_376x5.txt',
            'W_hh_full_376x94.npy',
            'W_hh_full_376x94.txt',
            'b_ih_full_376.npy',
            'b_ih_full_376.txt',
            'b_hh_full_376.npy',
            'b_hh_full_376.txt',
            'b_combined_full_376.npy',
            'b_combined_full_376.txt',
            'W_all_376x100.npy',
            'W_all_376x100.txt',
        ]
        
        for fname in full_files:
            f.write(f"  ✓ full/{fname}\n")
            total_files += 1
        
        f.write("\n" + "="*80 + "\n")
        f.write(f" TOTAL FILES CREATED: {total_files}\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Summary created: {summary_file}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    pth_file = "soc_lstm_model_1layer.pth"
    output_dir = "lstm_weights_separated"
    
    print("\n" + "="*80)
    print(" LSTM WEIGHT EXTRACTOR - SEPARATE FILES FOR EACH GATE")
    print("="*80)
    print(f"\nInput:  {pth_file}")
    print(f"Output: {output_dir}/")
    print("\n" + "="*80 + "\n")
    
    # Extract all weights
    all_weights = extract_all_lstm_weights_separately(pth_file, output_dir)
    
    print("\n" + "="*80)
    print(" ✅ EXTRACTION COMPLETE!")
    print("="*80)
    
    print(f"\nAll files saved to: {Path(output_dir).absolute()}")
    
    print("\n" + "="*80)
    print(" DIRECTORY STRUCTURE")
    print("="*80)
    print("""
lstm_weights_separated/
├── input/          → Input gate (94 units)
│   ├── W_ih_input_94x5.npy/.txt
│   ├── W_hh_input_94x94.npy/.txt
│   ├── b_combined_input_94.npy/.txt
│   └── input_gate_94x100.npy/.txt
│
├── forget/         → Forget gate (94 units)
│   └── (same structure)
│
├── cell/           → Cell gate (94 units)
│   └── (same structure)
│
├── output/         → Output gate (94 units)
│   └── (same structure)
│
├── full/           → All gates combined (376 units)
│   ├── W_ih_full_376x5.npy/.txt
│   ├── W_hh_full_376x94.npy/.txt
│   ├── b_combined_full_376.npy/.txt
│   └── W_all_376x100.npy/.txt
│
├── README.txt      → Complete documentation
└── SUMMARY.txt     → File list
    """)
    
    print("="*80)
    print("\n✅ ALL WEIGHTS EXTRACTED AND ORGANIZED BY GATE!\n")