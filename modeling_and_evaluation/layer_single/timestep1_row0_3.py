import torch
import numpy as np

def create_376x100_matrix(pth_file='soc_lstm_model_1layer.pth'):
    """Create 376x100 matrix from LSTM .pth file"""
    
    print("="*80)
    print(" CREATING 376x100 MATRIX FROM LSTM MODEL")
    print("="*80)
    
    print(f"\nLoading: {pth_file}")
    model_data = torch.load(pth_file, map_location='cpu')
    
    if isinstance(model_data, dict):
        if 'model_state_dict' in model_data:
            state_dict = model_data['model_state_dict']
        elif 'state_dict' in model_data:
            state_dict = model_data['state_dict']
        else:
            state_dict = model_data
    else:
        state_dict = model_data.state_dict() if hasattr(model_data, 'state_dict') else model_data
    
    W_ih = state_dict['lstm.weight_ih_l0'].cpu().numpy()
    W_hh = state_dict['lstm.weight_hh_l0'].cpu().numpy()
    b_ih = state_dict['lstm.bias_ih_l0'].cpu().numpy()
    b_hh = state_dict['lstm.bias_hh_l0'].cpu().numpy()
    
    print(f"\n  W_ih shape: {W_ih.shape}")
    print(f"  W_hh shape: {W_hh.shape}")
    print(f"  b_ih shape: {b_ih.shape}")
    print(f"  b_hh shape: {b_hh.shape}")
    
    bias_combined = b_ih + b_hh
    matrix = np.concatenate([W_ih, W_hh, bias_combined.reshape(-1, 1)], axis=1)
    
    print(f"\n✓ Created matrix with shape: {matrix.shape}")
    return matrix


def compute_timestep0(W_all, x0):
    """Compute timestep 0 to get h_prev and c_prev for timestep 1"""
    
    print("\n" + "="*80)
    print(" COMPUTING TIMESTEP 0 (for h_prev and c_prev)")
    print("="*80)
    
    h_prev = np.zeros(94)
    c_prev = np.zeros(94)
    
    input_vector = np.concatenate([x0, h_prev, [1.0]])
    
    # Compute accumulated output
    accumulated_output = np.zeros(376)
    
    total_cols = 100
    cols_per_tile = 4
    num_tiles = total_cols // cols_per_tile
    
    for tile_idx in range(num_tiles):
        start_col = tile_idx * cols_per_tile
        end_col = start_col + cols_per_tile
        
        tile_weights = W_all[:, start_col:end_col]
        tile_input = input_vector[start_col:end_col]
        partial_output = tile_weights @ tile_input
        accumulated_output += partial_output
    
    # Split into gates
    i_gate_raw = accumulated_output[0:94]
    f_gate_raw = accumulated_output[94:188]
    g_gate_raw = accumulated_output[188:282]
    o_gate_raw = accumulated_output[282:376]
    
    # Apply activation functions
    i_t = 1.0 / (1.0 + np.exp(-i_gate_raw))
    f_t = 1.0 / (1.0 + np.exp(-f_gate_raw))
    g_t = np.tanh(g_gate_raw)
    o_t = 1.0 / (1.0 + np.exp(-o_gate_raw))
    
    # Compute states
    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * np.tanh(c_t)
    
    print(f"\n✓ Timestep 0 computed")
    print(f"  h_t (first 10): {h_t[:10].tolist()}")
    print(f"  c_t (first 10): {c_t[:10].tolist()}")
    
    print("\n" + "="*80)
    print(" TIMESTEP 0 OUTPUT h_t (ALL 94 VALUES)")
    print("="*80)
    print("This h_t becomes the h_prev input for Timestep 1\n")
    for i in range(94):
        print(f"  h_t[{i:2d}] = {h_t[i]:13.10f}")
    
    return h_t, c_t


def compute_all_25_tiles_timestep1_rows0to3(W_all, x1, h_prev, c_prev):
    """
    Compute all 25 tiles for TIMESTEP 1, rows 0-3 with detailed breakdown
    
    Args:
        W_all: (376, 100) weight matrix
        x1: (5,) input features for timestep 1
        h_prev: (94,) hidden state from timestep 0
        c_prev: (94,) cell state from timestep 0
    
    Returns:
        Dictionary with all tiles computation and final results
    """
    
    print("\n" + "="*80)
    print(" TIMESTEP 1: ALL 25 TILES COMPUTATION FOR ROWS [0:3]")
    print("="*80)
    
    # Create input vector (100,) = [x1 (5,), h_prev (94,), 1.0]
    input_vector = np.concatenate([x1, h_prev, [1.0]])
    
    print(f"\nTimestep 1 Input Vector (100×1):")
    print(f"  x1 (features):       {x1.tolist()}")
    print(f"  h_prev (first 10):   {h_prev[:10].tolist()}")
    print(f"  bias:                1.0")
    
    print("\n" + "="*80)
    print(" h_prev INPUT FOR TIMESTEP 1 (ALL 94 VALUES)")
    print("="*80)
    print("This is the h_t output from Timestep 0\n")
    for i in range(94):
        print(f"  h_prev[{i:2d}] = {h_prev[i]:13.10f}")
    
    # Tiled computation parameters
    total_cols = 100
    cols_per_tile = 4
    num_tiles = total_cols // cols_per_tile  # 25 tiles
    
    print(f"\nTiling Configuration:")
    print(f"  Total columns: {total_cols}")
    print(f"  Column tiles: {num_tiles} tiles ({cols_per_tile} columns per tile)")
    print(f"  Focus rows: [0:3] (4 rows)")
    
    # Accumulator for rows 0-3
    accumulated_output_rows0to3 = np.zeros(4)
    
    # Store all tile results
    all_tile_results = []
    
    print("\n" + "="*80)
    print("TILE-BY-TILE COMPUTATION FOR ROWS [0:3] - TIMESTEP 1")
    print("="*80)
    
    # Process each column tile
    for tile_idx in range(num_tiles):
        start_col = tile_idx * cols_per_tile
        end_col = start_col + cols_per_tile
        
        # Extract column tile for rows [0:3]: (4, 4)
        tile_weights = W_all[0:4, start_col:end_col]
        
        # Extract corresponding input values: (4,)
        tile_input = input_vector[start_col:end_col]
        
        # Compute partial result: (4, 4) @ (4, 1) = (4, 1)
        partial_output = tile_weights @ tile_input
        
        # Accumulate
        accumulated_output_rows0to3 += partial_output
        
        # Store tile result
        all_tile_results.append({
            'tile_idx': tile_idx,
            'start_col': start_col,
            'end_col': end_col,
            'tile_weights': tile_weights,
            'tile_input': tile_input,
            'partial_output': partial_output.copy(),
            'accumulated_output': accumulated_output_rows0to3.copy()
        })
        
        # Print detailed tile information
        print(f"\n{'='*80}")
        print(f"TILE {tile_idx}: Columns [{start_col}:{end_col-1}]")
        print("="*80)
        print(f"Tile Input Vector:  {tile_input.tolist()}")
        
        print(f"\nWeight Matrix [Rows 0-3, Cols {start_col}-{end_col-1}]:")
        for i in range(4):
            row_idx = i
            print(f"  Row {row_idx}: [{tile_weights[i, 0]:13.10f}, {tile_weights[i, 1]:13.10f}, {tile_weights[i, 2]:13.10f}, {tile_weights[i, 3]:13.10f}]")
        
        print(f"\nPartial Output from Tile {tile_idx}:")
        for i in range(4):
            row_idx = i
            print(f"  Row {row_idx}: {partial_output[i]:13.10f}")
        
        print(f"\nAccumulated Output After Tile {tile_idx}:")
        for i in range(4):
            row_idx = i
            print(f"  Row {row_idx}: {accumulated_output_rows0to3[i]:13.10f}")
    
    print("\n" + "="*80)
    print(" FINAL ACCUMULATED OUTPUT FOR ROWS [0:3] - TIMESTEP 1")
    print("="*80)
    
    for i in range(4):
        row_idx = i
        print(f"Row {row_idx} (Input Gate): {accumulated_output_rows0to3[i]:13.10f}")
    
    # Apply sigmoid to get activated values
    i_t_rows0to3 = 1.0 / (1.0 + np.exp(-accumulated_output_rows0to3))
    
    print("\n" + "="*80)
    print(" ACTIVATED VALUES (After Sigmoid) - ROWS [0:3]")
    print("="*80)
    for i in range(4):
        print(f"Row {i}: {i_t_rows0to3[i]:13.10f}")
    
    return {
        'all_tile_results': all_tile_results,
        'final_accumulated': accumulated_output_rows0to3,
        'activated_values': i_t_rows0to3,
        'input_vector': input_vector,
        'h_prev': h_prev,
        'num_tiles': num_tiles
    }


def save_timestep1_results(results, prefix="timestep1_rows0to3"):
    """Save detailed all 25 tiles computation results for timestep 1, rows 0-3"""
    
    print("\n" + "="*80)
    print(" SAVING TIMESTEP 1 RESULTS TO FILE")
    print("="*80)
    
    with open(f"{prefix}_all_25_tiles_detailed.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("TIMESTEP 1: ALL 25 TILES COMPUTATION FOR ROWS [0:3]\n")
        f.write("25 Column Tiles × 4 Columns per Tile = 100 Total Columns\n")
        f.write("Focus: Rows [0:3] (Input Gate)\n")
        f.write("="*80 + "\n\n")
        
        # Write h_prev values
        f.write("="*80 + "\n")
        f.write(" h_prev INPUT FOR TIMESTEP 1 (ALL 94 VALUES)\n")
        f.write("="*80 + "\n")
        f.write("This is the h_t output from Timestep 0\n\n")
        h_prev = results['h_prev']
        for i in range(94):
            f.write(f"  h_prev[{i:2d}] = {h_prev[i]:13.10f}\n")
        f.write("\n")
        
        num_tiles = results['num_tiles']
        
        for tile_result in results['all_tile_results']:
            tile_idx = tile_result['tile_idx']
            start_col = tile_result['start_col']
            end_col = tile_result['end_col']
            tile_weights = tile_result['tile_weights']
            tile_input = tile_result['tile_input']
            partial_output = tile_result['partial_output']
            accumulated_output = tile_result['accumulated_output']
            
            f.write("="*80 + "\n")
            f.write(f"TILE {tile_idx}: Columns [{start_col}:{end_col-1}]\n")
            f.write("="*80 + "\n")
            f.write(f"Tile Input Vector (4 values): {tile_input.tolist()}\n\n")
            
            f.write(f"Weight Matrix [Rows 0-3, Cols {start_col}-{end_col-1}]:\n")
            for i in range(4):
                row_idx = i
                f.write(f"  Row {row_idx}: [{tile_weights[i, 0]:13.10f}, {tile_weights[i, 1]:13.10f}, {tile_weights[i, 2]:13.10f}, {tile_weights[i, 3]:13.10f}]\n")
            
            f.write(f"\nPartial Output from Tile {tile_idx}:\n")
            for i in range(4):
                row_idx = i
                f.write(f"  Row {row_idx}: {partial_output[i]:13.10f}\n")
            
            f.write(f"\nAccumulated Output After Tile {tile_idx}:\n")
            for i in range(4):
                row_idx = i
                f.write(f"  Row {row_idx}: {accumulated_output[i]:13.10f}\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write(" FINAL ACCUMULATED OUTPUT FOR ROWS [0:3] - TIMESTEP 1\n")
        f.write("="*80 + "\n")
        final_accumulated = results['final_accumulated']
        for i in range(4):
            row_idx = i
            f.write(f"Row {row_idx} (Input Gate): {final_accumulated[i]:13.10f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(" ACTIVATED VALUES (After Sigmoid) - ROWS [0:3]\n")
        f.write("="*80 + "\n")
        activated_values = results['activated_values']
        for i in range(4):
            f.write(f"Row {i}: {activated_values[i]:13.10f}\n")
    
    print(f"✓ Saved all 25 tiles details to: {prefix}_all_25_tiles_detailed.txt")


if __name__ == "__main__":
    
    print("\n" + "="*80)
    print(" LSTM TIMESTEP 1 - ROWS [0:3], ALL 25 TILES")
    print("="*80)
    
    # Step 1: Create 376x100 weight matrix
    pth_file = "soc_lstm_model_1layer.pth"
    W_all = create_376x100_matrix(pth_file)
    
    # Step 2: Define inputs
    x0 = np.array([-0.116864, 0.396518, 1.655884, 0.388196, -0.634126])  # Timestep 0
    x1 = np.array([-0.116864, 0.396518, 1.655884, 0.388196, -0.634162])  # Timestep 1
    
    print("\n" + "="*80)
    print(" INPUT DATA")
    print("="*80)
    print(f"\nTimestep 0 features (x0): {x0.tolist()}")
    print(f"Timestep 1 features (x1): {x1.tolist()}")
    
    # Step 3: Compute timestep 0 to get h_prev and c_prev
    h_prev, c_prev = compute_timestep0(W_all, x0)
    
    # Step 4: Compute all 25 tiles for timestep 1, rows 0-3
    print("\n" + "="*80)
    print(" COMPUTING TIMESTEP 1 - ALL 25 TILES FOR ROWS [0:3]")
    print("="*80)
    
    results = compute_all_25_tiles_timestep1_rows0to3(W_all, x1, h_prev, c_prev)
    
    # Step 5: Save results
    save_timestep1_results(results, prefix="timestep1_rows0to3")
    
    print("\n" + "="*80)
    print(" ✅ TIMESTEP 1 COMPUTATION COMPLETE!")
    print("="*80)
    print("\nGenerated file:")
    print("  • timestep1_rows0to3_all_25_tiles_detailed.txt")
    
    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80)
    print(f"\nProcessed {results['num_tiles']} tiles for Timestep 1")
    print(f"Each tile: (4 rows × 4 cols) @ (4×1 input) = (4×1 partial output)")
    print(f"\nFinal Accumulated Output for Rows [0:3] (Input Gate) - Timestep 1:")
    for i in range(4):
        print(f"  Row {i}: {results['final_accumulated'][i]:13.10f}")
    
    print(f"\nActivated Values (After Sigmoid) - Rows [0:3]:")
    for i in range(4):
        print(f"  Row {i}: {results['activated_values'][i]:13.10f}")
    
    print("\n" + "="*80)
    print("\n✅ VERIFICATION COMPLETE!\n")