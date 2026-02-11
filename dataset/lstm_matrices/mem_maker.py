import numpy as np

def float_to_fixed_point(value, int_bits=7, frac_bits=8):
    """
    Convert floating point to fixed point representation
    
    Format: 1 sign bit + 7 integer bits + 8 fractional bits = 16 bits total
    
    Args:
        value: floating point number
        int_bits: number of integer bits (default 7)
        frac_bits: number of fractional bits (default 8)
    
    Returns:
        16-bit fixed point value as hex string
    """
    total_bits = 1 + int_bits + frac_bits  # 16 bits total
    
    # Scale the value by 2^frac_bits
    scaled_value = value * (2 ** frac_bits)
    
    # Round to nearest integer
    fixed_value = int(round(scaled_value))
    
    # Handle negative numbers using two's complement
    if fixed_value < 0:
        fixed_value = (1 << total_bits) + fixed_value
    
    # Mask to ensure we only use 16 bits
    fixed_value = fixed_value & ((1 << total_bits) - 1)
    
    return fixed_value

def convert_matrix_to_fixed_point(input_file, output_file, int_bits=7, frac_bits=8):
    """
    Convert entire matrix to fixed point and save as memory file
    """
    
    print("="*80)
    print(" CONVERTING TO FIXED POINT FORMAT")
    print("="*80)
    print(f"\nFormat: 1 sign + {int_bits} integer + {frac_bits} fractional = {1+int_bits+frac_bits} bits")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Read the data
    print("\nReading data...")
    data = np.loadtxt(input_file, delimiter=',')
    print(f"Shape: {data.shape}")
    print(f"Min value: {data.min():.6f}")
    print(f"Max value: {data.max():.6f}")
    
    # Check range
    max_representable = (2 ** int_bits) - (1.0 / (2 ** frac_bits))
    min_representable = -(2 ** int_bits)
    
    print(f"\nRepresentable range:")
    print(f"  Min: {min_representable}")
    print(f"  Max: {max_representable}")
    
    if data.min() < min_representable or data.max() > max_representable:
        print("\n⚠️ WARNING: Some values are outside representable range!")
        print(f"  Values < {min_representable}: {np.sum(data < min_representable)}")
        print(f"  Values > {max_representable}: {np.sum(data > max_representable)}")
    
    # Convert to fixed point
    print("\nConverting to fixed point...")
    rows, cols = data.shape
    
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"// LSTM Weight Matrix - Fixed Point Format\n")
        f.write(f"// Format: 1 sign + {int_bits} integer + {frac_bits} fractional = {1+int_bits+frac_bits} bits\n")
        f.write(f"// Matrix size: {rows} x {cols}\n")
        f.write(f"// Total values: {rows * cols}\n")
        f.write(f"//\n")
        f.write(f"// Each line contains one 16-bit hex value (4 hex digits)\n")
        f.write(f"// Address format: row * {cols} + col\n")
        f.write(f"//\n\n")
        
        # Write data
        address = 0
        for i in range(rows):
            for j in range(cols):
                float_val = data[i, j]
                fixed_val = float_to_fixed_point(float_val, int_bits, frac_bits)
                
                # Write as 4-digit hex (16 bits)
                f.write(f"{fixed_val:04X}\n")
                address += 1
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{rows} rows...")
    
    print(f"\n✓ Conversion complete!")
    print(f"  Total values written: {rows * cols}")
    print(f"  Output file: {output_file}")
    
    # Create verification file with both formats
    verify_file = output_file.replace('.mem', '_verify.txt')
    with open(verify_file, 'w') as f:
        f.write(f"{'Address':<10} {'Float':<15} {'Fixed(Hex)':<12} {'Fixed(Dec)':<12} {'Reconstructed':<15} {'Error':<12}\n")
        f.write("="*90 + "\n")
        
        for i in range(min(100, rows)):  # First 100 values for verification
            for j in range(min(10, cols)):  # First 10 columns
                addr = i * cols + j
                float_val = data[i, j]
                fixed_val = float_to_fixed_point(float_val, int_bits, frac_bits)
                
                # Reconstruct the value
                if fixed_val >= (1 << (1 + int_bits + frac_bits - 1)):  # Check sign bit
                    # Negative number (two's complement)
                    reconstructed = (fixed_val - (1 << (1 + int_bits + frac_bits))) / (2 ** frac_bits)
                else:
                    # Positive number
                    reconstructed = fixed_val / (2 ** frac_bits)
                
                error = abs(float_val - reconstructed)
                
                f.write(f"{addr:<10} {float_val:<15.8f} {fixed_val:04X}         {fixed_val:<12} {reconstructed:<15.8f} {error:<12.8e}\n")
    
    print(f"✓ Verification file created: {verify_file}")
    
    # Statistics
    print("\n" + "="*80)
    print(" CONVERSION STATISTICS")
    print("="*80)
    
    # Reconstruct all values to check error
    errors = []
    for i in range(rows):
        for j in range(cols):
            float_val = data[i, j]
            fixed_val = float_to_fixed_point(float_val, int_bits, frac_bits)
            
            if fixed_val >= (1 << (1 + int_bits + frac_bits - 1)):
                reconstructed = (fixed_val - (1 << (1 + int_bits + frac_bits))) / (2 ** frac_bits)
            else:
                reconstructed = fixed_val / (2 ** frac_bits)
            
            errors.append(abs(float_val - reconstructed))
    
    errors = np.array(errors)
    print(f"\nQuantization Error:")
    print(f"  Mean error:   {errors.mean():.8e}")
    print(f"  Max error:    {errors.max():.8e}")
    print(f"  Min error:    {errors.min():.8e}")
    print(f"  Std error:    {errors.std():.8e}")
    
    # Resolution
    resolution = 1.0 / (2 ** frac_bits)
    print(f"\nFixed Point Resolution: {resolution} ({1/resolution:.0f} steps per unit)")

def create_verilog_mem_init(input_file, output_file, int_bits=7, frac_bits=8):
    """
    Create Verilog-compatible memory initialization file
    """
    
    print("\n" + "="*80)
    print(" CREATING VERILOG MEMORY FILE")
    print("="*80)
    
    # Read data
    data = np.loadtxt(input_file, delimiter=',')
    rows, cols = data.shape
    
    verilog_file = output_file.replace('.mem', '_verilog.mem')
    
    with open(verilog_file, 'w') as f:
        # Verilog memory initialization format
        f.write(f"// Verilog Memory Initialization File\n")
        f.write(f"// Format: @address data\n")
        f.write(f"// Size: {rows * cols} words x 16 bits\n\n")
        
        address = 0
        for i in range(rows):
            for j in range(cols):
                float_val = data[i, j]
                fixed_val = float_to_fixed_point(float_val, int_bits, frac_bits)
                
                # Write in Verilog format: @address data
                f.write(f"@{address:08X} {fixed_val:04X}\n")
                address += 1
    
    print(f"✓ Verilog memory file created: {verilog_file}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    input_file = "W_all_376x100.txt"
    output_file = "W_all_376x100_fixed.mem"
    
    # Convert to fixed point (1 sign + 7 int + 8 frac = 16 bits)
    convert_matrix_to_fixed_point(input_file, output_file, int_bits=7, frac_bits=8)
    
    # Create Verilog-compatible version
    create_verilog_mem_init(input_file, output_file, int_bits=7, frac_bits=8)
    
    print("\n" + "="*80)
    print(" ✅ ALL FILES CREATED!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. W_all_376x100_fixed.mem        - Standard memory file (hex values)")
    print("  2. W_all_376x100_fixed_verify.txt - Verification with errors")
    print("  3. W_all_376x100_fixed_verilog.mem - Verilog $readmemh format")
    print("\n" + "="*80)
    print(" USAGE IN VERILOG")
    print("="*80)
    print("""
// In your Verilog module:
reg [15:0] weight_mem [0:37599];  // 376 * 100 = 37600 words

initial begin
    $readmemh("W_all_376x100_fixed_verilog.mem", weight_mem);
end

// Access example:
wire [15:0] weight_value;
assign weight_value = weight_mem[row_index * 100 + col_index];
    """)