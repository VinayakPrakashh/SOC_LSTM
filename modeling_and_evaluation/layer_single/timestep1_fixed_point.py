import torch
import numpy as np

# Fixed-point configuration: Q7.8 with sign-magnitude (1 sign + 7 int + 8 frac)
FRAC_BITS = 8
SCALE = 2 ** FRAC_BITS  # 256
SIGN_BIT = 0x8000  # Bit 15 for 16-bit representation
MAGNITUDE_MASK = 0x7FFF  # Bits 0-14

def float_to_fixed_sign_mag(value):
    """Convert float to Q7.8 sign-magnitude fixed-point"""
    magnitude = int(round(abs(value) * SCALE))
    
    # Clamp to 15-bit range
    if magnitude > 0x7FFF:
        magnitude = 0x7FFF
    
    # Set sign bit if negative
    if value < 0:
        return SIGN_BIT | magnitude
    else:
        return magnitude

def fixed_to_float_sign_mag(fixed_value):
    """Convert Q7.8 sign-magnitude fixed-point to float"""
    # Extract sign and magnitude
    is_negative = (fixed_value & SIGN_BIT) != 0
    magnitude = fixed_value & MAGNITUDE_MASK
    
    # Convert to float
    float_val = magnitude / SCALE
    
    return -float_val if is_negative else float_val

def fixed_multiply_sign_mag(a, b):
    """Multiply two Q7.8 sign-magnitude numbers"""
    # Extract signs and magnitudes
    sign_a = (a & SIGN_BIT) != 0
    sign_b = (b & SIGN_BIT) != 0
    mag_a = a & MAGNITUDE_MASK
    mag_b = b & MAGNITUDE_MASK
    
    # Multiply magnitudes
    result_mag = (mag_a * mag_b) >> FRAC_BITS
    
    # Clamp magnitude
    if result_mag > 0x7FFF:
        result_mag = 0x7FFF
    
    # Determine result sign (XOR of input signs)
    result_sign = sign_a ^ sign_b
    
    # Combine sign and magnitude
    if result_sign:
        return SIGN_BIT | result_mag
    else:
        return result_mag

def fixed_add_sign_mag(a, b):
    """Add two Q7.8 sign-magnitude numbers"""
    # Convert to float, add, convert back
    # (For simplicity - can be optimized for pure fixed-point)
    float_a = fixed_to_float_sign_mag(a)
    float_b = fixed_to_float_sign_mag(b)
    result = float_a + float_b
    return float_to_fixed_sign_mag(result)

def fixed_sigmoid(x):
    """Sigmoid approximation using sign-magnitude fixed-point"""
    float_val = fixed_to_float_sign_mag(x)
    sigmoid_val = 1.0 / (1.0 + np.exp(-float_val))
    return float_to_fixed_sign_mag(sigmoid_val)

def create_376x100_matrix_fixed(pth_file='soc_lstm_model_1layer.pth'):
    """Create 376x100 matrix in Q7.8 sign-magnitude fixed-point"""
    
    print("="*80)
    print(" CREATING 376x100 MATRIX (Q7.8 SIGN-MAGNITUDE)")
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
    matrix_float = np.concatenate([W_ih, W_hh, bias_combined.reshape(-1, 1)], axis=1)
    
    # Convert to sign-magnitude fixed-point
    matrix_fixed = np.zeros_like(matrix_float, dtype=np.int32)
    for i in range(matrix_float.shape[0]):
        for j in range(matrix_float.shape[1]):
            matrix_fixed[i, j] = float_to_fixed_sign_mag(matrix_float[i, j])
    
    print(f"\n✓ Created matrix with shape: {matrix_fixed.shape}")
    print(f"  Format: Q7.8 Sign-Magnitude (Bit 15=sign, Bits 14:0=magnitude)")
    print(f"  Scale factor: {SCALE}")
    
    return matrix_fixed, matrix_float


def compute_timestep0_fixed(W_all_fixed, x0_float):
    """Compute timestep 0 using Q7.8 sign-magnitude fixed-point"""
    
    print("\n" + "="*80)
    print(" COMPUTING TIMESTEP 0 (Q7.8 SIGN-MAGNITUDE)")
    print("="*80)
    
    # Convert input to sign-magnitude fixed-point
    x0_fixed = np.array([float_to_fixed_sign_mag(x) for x in x0_float], dtype=np.int32)
    h_prev_fixed = np.zeros(94, dtype=np.int32)
    c_prev_fixed = np.zeros(94, dtype=np.int32)
    
    # Create input vector [x0, h_prev, 1.0]
    bias_fixed = float_to_fixed_sign_mag(1.0)
    input_vector_fixed = np.concatenate([x0_fixed, h_prev_fixed, [bias_fixed]])
    
    print(f"\nInput x0 (Q7.8 sign-magnitude):")
    for i, (f, fx) in enumerate(zip(x0_float, x0_fixed)):
        sign_bit = "1" if (fx & SIGN_BIT) else "0"
        magnitude = fx & MAGNITUDE_MASK
        print(f"  x0[{i}]: {f:10.6f} -> 0x{fx:04X} (sign={sign_bit}, mag={magnitude})")
    
    # Compute accumulated output
    accumulated_output_fixed = np.zeros(376, dtype=np.int32)
    
    total_cols = 100
    cols_per_tile = 4
    num_tiles = total_cols // cols_per_tile
    
    for tile_idx in range(num_tiles):
        start_col = tile_idx * cols_per_tile
        end_col = start_col + cols_per_tile
        
        tile_weights_fixed = W_all_fixed[:, start_col:end_col]
        tile_input_fixed = input_vector_fixed[start_col:end_col]
        
        # Sign-magnitude matrix-vector multiply
        for row in range(376):
            partial_sum = accumulated_output_fixed[row]
            for col in range(4):
                product = fixed_multiply_sign_mag(tile_weights_fixed[row, col], tile_input_fixed[col])
                partial_sum = fixed_add_sign_mag(partial_sum, product)
            accumulated_output_fixed[row] = partial_sum
    
    # Split into gates
    i_gate_fixed = accumulated_output_fixed[0:94]
    f_gate_fixed = accumulated_output_fixed[94:188]
    g_gate_fixed = accumulated_output_fixed[188:282]
    o_gate_fixed = accumulated_output_fixed[282:376]
    
    # Apply activation functions
    i_t_fixed = np.array([fixed_sigmoid(x) for x in i_gate_fixed], dtype=np.int32)
    f_t_fixed = np.array([fixed_sigmoid(x) for x in f_gate_fixed], dtype=np.int32)
    
    # Tanh approximation
    g_t_fixed = np.zeros(94, dtype=np.int32)
    for i in range(94):
        float_val = fixed_to_float_sign_mag(g_gate_fixed[i])
        tanh_val = np.tanh(float_val)
        g_t_fixed[i] = float_to_fixed_sign_mag(tanh_val)
    
    o_t_fixed = np.array([fixed_sigmoid(x) for x in o_gate_fixed], dtype=np.int32)
    
    # Compute cell state: c_t = f_t * c_prev + i_t * g_t
    c_t_fixed = np.zeros(94, dtype=np.int32)
    for i in range(94):
        term1 = fixed_multiply_sign_mag(f_t_fixed[i], c_prev_fixed[i])
        term2 = fixed_multiply_sign_mag(i_t_fixed[i], g_t_fixed[i])
        c_t_fixed[i] = fixed_add_sign_mag(term1, term2)
    
    # Compute hidden state: h_t = o_t * tanh(c_t)
    h_t_fixed = np.zeros(94, dtype=np.int32)
    for i in range(94):
        float_val = fixed_to_float_sign_mag(c_t_fixed[i])
        tanh_val = np.tanh(float_val)
        tanh_fixed = float_to_fixed_sign_mag(tanh_val)
        h_t_fixed[i] = fixed_multiply_sign_mag(o_t_fixed[i], tanh_fixed)
    
    print(f"\n✓ Timestep 0 computed (sign-magnitude)")
    
    # Convert h_t to float for display
    h_t_float = np.array([fixed_to_float_sign_mag(x) for x in h_t_fixed])
    c_t_float = np.array([fixed_to_float_sign_mag(x) for x in c_t_fixed])
    
    print(f"  h_t (first 10, float): {h_t_float[:10].tolist()}")
    print(f"  c_t (first 10, float): {c_t_float[:10].tolist()}")
    
    print("\n" + "="*80)
    print(" TIMESTEP 0 OUTPUT h_t (ALL 94 VALUES - Q7.8 SIGN-MAGNITUDE)")
    print("="*80)
    print("Format: Hex (Sign|Mag) | Float Value\n")
    for i in range(94):
        sign_bit = "1" if (h_t_fixed[i] & SIGN_BIT) else "0"
        magnitude = h_t_fixed[i] & MAGNITUDE_MASK
        print(f"  h_t[{i:2d}] = 0x{h_t_fixed[i]:04X} (S={sign_bit}|M={magnitude:5d}) | {h_t_float[i]:13.10f}")
    
    return h_t_fixed, c_t_fixed


def compute_all_25_tiles_timestep1_rows0to3_fixed(W_all_fixed, x1_float, h_prev_fixed, c_prev_fixed):
    """Compute timestep 1 using Q7.8 sign-magnitude for rows 0-3"""
    
    print("\n" + "="*80)
    print(" TIMESTEP 1: Q7.8 SIGN-MAGNITUDE COMPUTATION FOR ROWS [0:3]")
    print("="*80)
    
    # Convert inputs to sign-magnitude fixed-point
    x1_fixed = np.array([float_to_fixed_sign_mag(x) for x in x1_float], dtype=np.int32)
    
    print(f"\nInput x1 (Q7.8 sign-magnitude):")
    for i, (f, fx) in enumerate(zip(x1_float, x1_fixed)):
        sign_bit = "1" if (fx & SIGN_BIT) else "0"
        magnitude = fx & MAGNITUDE_MASK
        print(f"  x1[{i}]: {f:10.6f} -> 0x{fx:04X} (S={sign_bit}|M={magnitude:5d})")
    
    # Create input vector
    bias_fixed = float_to_fixed_sign_mag(1.0)
    input_vector_fixed = np.concatenate([x1_fixed, h_prev_fixed, [bias_fixed]])
    
    print("\n" + "="*80)
    print(" h_prev INPUT FOR TIMESTEP 1 (Q7.8 SIGN-MAGNITUDE)")
    print("="*80)
    print("Format: Hex (Sign|Mag) | Float Value\n")
    for i in range(94):
        h_float = fixed_to_float_sign_mag(h_prev_fixed[i])
        sign_bit = "1" if (h_prev_fixed[i] & SIGN_BIT) else "0"
        magnitude = h_prev_fixed[i] & MAGNITUDE_MASK
        print(f"  h_prev[{i:2d}] = 0x{h_prev_fixed[i]:04X} (S={sign_bit}|M={magnitude:5d}) | {h_float:13.10f}")
    
    # Tiling parameters
    total_cols = 100
    cols_per_tile = 4
    num_tiles = total_cols // cols_per_tile
    
    print(f"\nTiling Configuration:")
    print(f"  Total columns: {total_cols}")
    print(f"  Column tiles: {num_tiles} tiles")
    print(f"  Focus rows: [0:3]")
    
    # Accumulator for rows 0-3
    accumulated_output_fixed = np.zeros(4, dtype=np.int32)
    
    all_tile_results = []
    
    print("\n" + "="*80)
    print("TILE-BY-TILE COMPUTATION (Q7.8 SIGN-MAGNITUDE)")
    print("="*80)
    
    for tile_idx in range(num_tiles):
        start_col = tile_idx * cols_per_tile
        end_col = start_col + cols_per_tile
        
        tile_weights_fixed = W_all_fixed[0:4, start_col:end_col]
        tile_input_fixed = input_vector_fixed[start_col:end_col]
        
        # Compute partial output
        partial_output_fixed = np.zeros(4, dtype=np.int32)
        for row in range(4):
            partial_sum = 0
            for col in range(4):
                product = fixed_multiply_sign_mag(tile_weights_fixed[row, col], tile_input_fixed[col])
                partial_sum = fixed_add_sign_mag(partial_sum, product)
            partial_output_fixed[row] = partial_sum
        
        # Accumulate
        for row in range(4):
            accumulated_output_fixed[row] = fixed_add_sign_mag(accumulated_output_fixed[row], partial_output_fixed[row])
        
        # Store results
        all_tile_results.append({
            'tile_idx': tile_idx,
            'start_col': start_col,
            'end_col': end_col,
            'tile_weights_fixed': tile_weights_fixed.copy(),
            'tile_input_fixed': tile_input_fixed.copy(),
            'partial_output_fixed': partial_output_fixed.copy(),
            'accumulated_output_fixed': accumulated_output_fixed.copy()
        })
        
        # Print tile details
        print(f"\n{'='*80}")
        print(f"TILE {tile_idx}: Columns [{start_col}:{end_col-1}]")
        print("="*80)
        
        print(f"\nTile Input (Q7.8 sign-mag):")
        for i in range(4):
            float_val = fixed_to_float_sign_mag(tile_input_fixed[i])
            sign_bit = "1" if (tile_input_fixed[i] & SIGN_BIT) else "0"
            magnitude = tile_input_fixed[i] & MAGNITUDE_MASK
            print(f"  Col {start_col+i}: 0x{tile_input_fixed[i]:04X} (S={sign_bit}|M={magnitude:5d}) = {float_val:10.6f}")
        
        print(f"\nPartial Output (Q7.8):")
        for row in range(4):
            float_val = fixed_to_float_sign_mag(partial_output_fixed[row])
            sign_bit = "1" if (partial_output_fixed[row] & SIGN_BIT) else "0"
            magnitude = partial_output_fixed[row] & MAGNITUDE_MASK
            print(f"  Row {row}: 0x{partial_output_fixed[row]:04X} (S={sign_bit}|M={magnitude:5d}) = {float_val:13.10f}")
        
        print(f"\nAccumulated Output (Q7.8):")
        for row in range(4):
            float_val = fixed_to_float_sign_mag(accumulated_output_fixed[row])
            sign_bit = "1" if (accumulated_output_fixed[row] & SIGN_BIT) else "0"
            magnitude = accumulated_output_fixed[row] & MAGNITUDE_MASK
            print(f"  Row {row}: 0x{accumulated_output_fixed[row]:04X} (S={sign_bit}|M={magnitude:5d}) = {float_val:13.10f}")
    
    print("\n" + "="*80)
    print(" FINAL ACCUMULATED OUTPUT (Q7.8 SIGN-MAGNITUDE)")
    print("="*80)
    for row in range(4):
        float_val = fixed_to_float_sign_mag(accumulated_output_fixed[row])
        sign_bit = "1" if (accumulated_output_fixed[row] & SIGN_BIT) else "0"
        magnitude = accumulated_output_fixed[row] & MAGNITUDE_MASK
        print(f"Row {row}: 0x{accumulated_output_fixed[row]:04X} (S={sign_bit}|M={magnitude:5d}) = {float_val:13.10f}")
    
    # Apply sigmoid
    activated_values_fixed = np.array([fixed_sigmoid(x) for x in accumulated_output_fixed], dtype=np.int32)
    activated_values_float = np.array([fixed_to_float_sign_mag(x) for x in activated_values_fixed])
    
    print("\n" + "="*80)
    print(" ACTIVATED VALUES (After Sigmoid - Q7.8)")
    print("="*80)
    for row in range(4):
        sign_bit = "1" if (activated_values_fixed[row] & SIGN_BIT) else "0"
        magnitude = activated_values_fixed[row] & MAGNITUDE_MASK
        print(f"Row {row}: 0x{activated_values_fixed[row]:04X} (S={sign_bit}|M={magnitude:5d}) = {activated_values_float[row]:13.10f}")
    
    return {
        'all_tile_results': all_tile_results,
        'final_accumulated_fixed': accumulated_output_fixed,
        'activated_values_fixed': activated_values_fixed,
        'activated_values_float': activated_values_float,
        'h_prev_fixed': h_prev_fixed,
        'num_tiles': num_tiles
    }


if __name__ == "__main__":
    
    print("\n" + "="*80)
    print(" LSTM SIGN-MAGNITUDE FIXED-POINT COMPUTATION (Q7.8)")
    print("="*80)
    print(f"Format: Bit 15 = sign, Bits 14:0 = magnitude")
    print(f"Scale: {SCALE}")
    
    # Step 1: Create weight matrix
    pth_file = "soc_lstm_model_1layer.pth"
    W_all_fixed, W_all_float = create_376x100_matrix_fixed(pth_file)
    
    # Step 2: Define inputs
    x0_float = np.array([-0.116864, 0.396518, 1.655884, 0.388196, -0.634126])
    x1_float = np.array([-0.116864, 0.396518, 1.655884, 0.388196, -0.634162])
    
    print("\n" + "="*80)
    print(" INPUT DATA")
    print("="*80)
    print(f"\nTimestep 0: {x0_float.tolist()}")
    print(f"Timestep 1: {x1_float.tolist()}")
    
    # Step 3: Compute timestep 0
    h_prev_fixed, c_prev_fixed = compute_timestep0_fixed(W_all_fixed, x0_float)
    
    # Step 4: Compute timestep 1
    results = compute_all_25_tiles_timestep1_rows0to3_fixed(W_all_fixed, x1_float, h_prev_fixed, c_prev_fixed)
    
    print("\n" + "="*80)
    print(" ✅ SIGN-MAGNITUDE FIXED-POINT COMPUTATION COMPLETE!")
    print("="*80)
    print(f"\nFinal Accumulated Output (Rows 0-3):")
    for i in range(4):
        fixed_val = results['final_accumulated_fixed'][i]
        float_val = fixed_to_float_sign_mag(fixed_val)
        sign_bit = "1" if (fixed_val & SIGN_BIT) else "0"
        magnitude = fixed_val & MAGNITUDE_MASK
        print(f"  Row {i}: 0x{fixed_val:04X} (S={sign_bit}|M={magnitude:5d}) = {float_val:13.10f}")
    
    print(f"\nActivated Values (After Sigmoid):")
    for i in range(4):
        print(f"  Row {i}: {results['activated_values_float'][i]:13.10f}")
    
    print("\n✅ DONE!\n")