import torch
import numpy as np

def create_376x100_matrix(pth_file='lstm_soc_model.pth', output_file='W_all_376x100.txt'):
    """
    Create 376x100 matrix from LSTM .pth file
    
    Matrix structure:
    - Columns [0:5]:   W_ih (input weights, 5 features)
    - Columns [5:99]:  W_hh (hidden weights, 94 units)  
    - Column [99]:     bias (combined b_ih + b_hh)
    
    Row structure:
    - Rows [0:94]:     Input gate
    - Rows [94:188]:   Forget gate
    - Rows [188:282]:  Cell gate
    - Rows [282:376]:  Output gate
    """
    
    print("="*80)
    print(" CREATING 376x100 MATRIX FROM LSTM MODEL")
    print("="*80)
    
    # Load model
    print(f"\nLoading: {pth_file}")
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
    
    # Extract LSTM components
    print("\nExtracting LSTM weights and biases...")
    
    W_ih = state_dict['lstm.weight_ih_l0'].cpu().numpy()  # (376, 5)
    W_hh = state_dict['lstm.weight_hh_l0'].cpu().numpy()  # (376, 94)
    b_ih = state_dict['lstm.bias_ih_l0'].cpu().numpy()    # (376,)
    b_hh = state_dict['lstm.bias_hh_l0'].cpu().numpy()    # (376,)
    
    print(f"  W_ih shape: {W_ih.shape}")
    print(f"  W_hh shape: {W_hh.shape}")
    print(f"  b_ih shape: {b_ih.shape}")
    print(f"  b_hh shape: {b_hh.shape}")
    
    # Combine bias (PyTorch adds them together in forward pass)
    bias_combined = b_ih + b_hh  # (376,)
    
    # Create 376x100 matrix by concatenating horizontally
    # [W_ih (376x5) | W_hh (376x94) | bias (376x1)]
    matrix = np.concatenate([
        W_ih,                          # Columns 0-4   (5 columns)
        W_hh,                          # Columns 5-98  (94 columns)
        bias_combined.reshape(-1, 1)   # Column 99     (1 column)
    ], axis=1)
    
    print(f"\n✓ Created matrix with shape: {matrix.shape}")
    
    # Verify shape
    assert matrix.shape == (376, 100), f"Expected (376, 100), got {matrix.shape}"
    
    # Save to text file
    print(f"\nSaving to: {output_file}")
    np.savetxt(output_file, matrix, fmt='%.8f', delimiter=',')
    
    print(f"✓ Saved successfully!")
    
    # Display statistics
    print("\n" + "="*80)
    print(" MATRIX STATISTICS")
    print("="*80)
    print(f"\nShape: {matrix.shape}")
    print(f"Total elements: {matrix.size}")
    print(f"\nValue range:")
    print(f"  Min:  {matrix.min():.8f}")
    print(f"  Max:  {matrix.max():.8f}")
    print(f"  Mean: {matrix.mean():.8f}")
    print(f"  Std:  {matrix.std():.8f}")
    
    print(f"\nColumn 99 (bias) statistics:")
    print(f"  Min:  {matrix[:, 99].min():.8f}")
    print(f"  Max:  {matrix[:, 99].max():.8f}")
    print(f"  Mean: {matrix[:, 99].mean():.8f}")
    
    # Show sample values
    print("\n" + "="*80)
    print(" SAMPLE VALUES")
    print("="*80)
    
    print(f"\nRow 0, Columns [0:5] (W_ih - input weights):")
    print(matrix[0, :5])
    
    print(f"\nRow 0, Columns [5:10] (W_hh - first 5 hidden weights):")
    print(matrix[0, 5:10])
    
    print(f"\nRow 0, Column [99] (bias):")
    print(matrix[0, 99])
    
    print(f"\nColumn 99 (all bias values, first 10 rows):")
    print(matrix[:10, 99])
    
    # Verify file was written correctly
    print("\n" + "="*80)
    print(" VERIFYING FILE")
    print("="*80)
    
    # Read back and verify
    loaded = np.loadtxt(output_file, delimiter=',')
    print(f"\nLoaded shape: {loaded.shape}")
    print(f"Matches original: {np.allclose(matrix, loaded)}")
    
    if not np.allclose(matrix, loaded):
        print("⚠️ WARNING: Loaded data doesn't match original!")
    else:
        print("✓ File verification passed!")
    
    # Count lines in file
    with open(output_file, 'r') as f:
        line_count = sum(1 for _ in f)
    
    print(f"\nFile line count: {line_count}")
    print(f"Expected: {matrix.shape[0]}")
    
    if line_count != matrix.shape[0]:
        print(f"⚠️ WARNING: Line count mismatch!")
    else:
        print("✓ Line count correct!")
    
    # Save also as space-separated (alternative format)
    space_file = output_file.replace('.txt', '_space.txt')
    np.savetxt(space_file, matrix, fmt='%.8f')
    print(f"\n✓ Also saved space-separated version: {space_file}")
    
    # Save as numpy binary
    npy_file = output_file.replace('.txt', '.npy')
    np.save(npy_file, matrix)
    print(f"✓ Also saved numpy binary: {npy_file}")
    
    return matrix

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    pth_file = "soc_lstm_model_1layer.pth"
    output_file = "W_all_376x100.txt"
    
    print("\n" + "="*80)
    print(" LSTM 376x100 MATRIX EXTRACTOR")
    print("="*80)
    print(f"\nInput:  {pth_file}")
    print(f"Output: {output_file}")
    print("\n" + "="*80 + "\n")
    
    # Create the matrix
    matrix = create_376x100_matrix(pth_file, output_file)
    
    print("\n" + "="*80)
    print(" ✅ SUCCESS!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  • {output_file}                - Comma-separated (376 rows x 100 cols)")
    print(f"  • {output_file.replace('.txt', '_space.txt')} - Space-separated")
    print(f"  • {output_file.replace('.txt', '.npy')}       - NumPy binary")
    
    print("\n" + "="*80)
    print(" MATRIX STRUCTURE")
    print("="*80)
    print("""
Columns:
  [0:5]     - W_ih: Input-to-hidden weights (5 input features)
  [5:99]    - W_hh: Hidden-to-hidden weights (94 hidden units)
  [99]      - bias: Combined bias (b_ih + b_hh)

Rows (Gates):
  [0:94]    - Input gate
  [94:188]  - Forget gate
  [188:282] - Cell gate
  [282:376] - Output gate

Total: 376 rows × 100 columns = 37,600 values
    """)
    
    print("="*80)
    print("\n✅ ALL 37,600 VALUES EXTRACTED CORRECTLY!\n")