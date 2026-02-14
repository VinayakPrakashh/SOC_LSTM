import numpy as np

def float_to_sign_magnitude(value):
    """
    Convert float to S7.8 sign-magnitude (16-bit)
    Bit 15: Sign (0=+, 1=-)
    Bits 14-8: Integer (7 bits)
    Bits 7-0: Fraction (8 bits)
    """
    sign_bit = 1 if value < 0 else 0
    abs_value = abs(value)
    
    # Scale by 2^8 = 256
    scaled = int(round(abs_value * 256))
    
    # Limit to 15 bits (max value for magnitude)
    if scaled > 32767:
        scaled = 32767
    
    # Combine: sign bit (bit 15) | magnitude (bits 14-0)
    fixed_point = (sign_bit << 15) | scaled
    
    return fixed_point

def create_mem_file(txt_file, mem_file):
    """
    Convert 376x100 txt file to Verilog memory file (.mem)
    """
    
    print("="*80)
    print(" CREATING VERILOG MEMORY FILE (S7.8 SIGN-MAGNITUDE)")
    print("="*80)
    
    # Read the 376x100 matrix
    print(f"\nReading: {txt_file}")
    data = np.loadtxt(txt_file, delimiter=',')
    rows, cols = data.shape
    
    print(f"Matrix shape: {rows} x {cols}")
    print(f"Total values: {rows * cols}")
    
    # Statistics
    print(f"\nOriginal data:")
    print(f"  Min:  {data.min():.6f}")
    print(f"  Max:  {data.max():.6f}")
    print(f"  Mean: {data.mean():.6f}")
    
    # Create memory file
    print(f"\nConverting to fixed-point and saving to: {mem_file}")
    
    with open(mem_file, 'w') as f:
        # Header
        f.write("// LSTM Weights - Verilog Memory File\n")
        f.write("// Format: S7.8 (Sign-Magnitude)\n")
        f.write("// Bit 15:    Sign (0=positive, 1=negative)\n")
        f.write("// Bits 14-8: Integer part (7 bits)\n")
        f.write("// Bits 7-0:  Fractional part (8 bits)\n")
        f.write(f"// Size: {rows} x {cols} = {rows*cols} values\n")
        f.write("// Each line: one 16-bit hex value\n")
        f.write("//\n")
        f.write("// Usage in Verilog:\n")
        f.write("//   reg [15:0] weight_mem [0:37599];\n")
        f.write("//   initial $readmemh(\"weights.mem\", weight_mem);\n")
        f.write("//\n\n")
        
        # Convert and write all values
        for i in range(rows):
            for j in range(cols):
                float_val = data[i, j]
                fixed_val = float_to_sign_magnitude(float_val)
                
                # Write as 4-digit hex
                f.write(f"{fixed_val:04X}\n")
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{rows} rows...")
    
    print(f"\n✓ Memory file created: {mem_file}")
    print(f"  Total lines written: {rows * cols}")
    
    # Create FULL verification file (ALL 37,600 values)
    verify_file = mem_file.replace('.mem', '_verify_full.txt')
    
    print(f"\nCreating FULL verification file (all {rows*cols} values)...")
    
    with open(verify_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write(" FULL VERIFICATION FILE - ALL 37,600 VALUES\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Addr':<8} {'Row':<5} {'Col':<5} {'Float':<15} {'Hex':<8} {'Binary':<20} {'Reconstructed':<15} {'Error':<12}\n")
        f.write("="*100 + "\n")
        
        for i in range(rows):
            for j in range(cols):
                addr = i * cols + j
                float_val = data[i, j]
                fixed_val = float_to_sign_magnitude(float_val)
                
                # Reconstruct value
                sign = (fixed_val >> 15) & 1
                magnitude = fixed_val & 0x7FFF
                reconstructed = magnitude / 256.0
                if sign:
                    reconstructed = -reconstructed
                
                error = abs(float_val - reconstructed)
                
                # Binary format: S_IIIIIII_FFFFFFFF
                binary = f"{fixed_val:016b}"
                binary_fmt = f"{binary[0]}_{binary[1:8]}_{binary[8:]}"
                
                f.write(f"{addr:<8} {i:<5} {j:<5} {float_val:<15.8f} {fixed_val:04X}     {binary_fmt:<20} {reconstructed:<15.8f} {error:<12.2e}\n")
            
            # Add separator every 94 rows (each gate)
            if (i + 1) % 94 == 0 and i < rows - 1:
                gate_num = (i + 1) // 94
                gate_names = ['INPUT', 'FORGET', 'CELL', 'OUTPUT']
                f.write("\n" + "-"*100 + "\n")
                f.write(f" END OF {gate_names[gate_num-1]} GATE | START OF {gate_names[gate_num]} GATE\n")
                f.write("-"*100 + "\n\n")
            
            # Progress
            if (i + 1) % 50 == 0:
                print(f"  Written {i+1}/{rows} rows...")
    
    print(f"\n✓ FULL verification file created: {verify_file}")
    print(f"  Total lines: {rows * cols}")
    
    # Create compact verification (summary only)
    summary_file = mem_file.replace('.mem', '_verify_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" VERIFICATION SUMMARY (First & Last 50 values from each gate)\n")
        f.write("="*80 + "\n\n")
        
        gate_names = ['INPUT', 'FORGET', 'CELL', 'OUTPUT']
        
        for gate_idx, gate_name in enumerate(gate_names):
            f.write(f"\n{'='*80}\n")
            f.write(f" {gate_name} GATE (Rows {gate_idx*94}:{(gate_idx+1)*94})\n")
            f.write(f"{'='*80}\n\n")
            
            # First 50 values
            f.write(f"First 50 values:\n")
            f.write(f"{'Addr':<8} {'Row':<5} {'Col':<5} {'Float':<15} {'Hex':<8}\n")
            f.write("-"*60 + "\n")
            
            start_row = gate_idx * 94
            count = 0
            for i in range(start_row, min(start_row + 1, rows)):
                for j in range(min(50, cols)):
                    addr = i * cols + j
                    float_val = data[i, j]
                    fixed_val = float_to_sign_magnitude(float_val)
                    f.write(f"{addr:<8} {i:<5} {j:<5} {float_val:<15.8f} {fixed_val:04X}\n")
            
            # Last 50 values
            f.write(f"\nLast 50 values:\n")
            f.write(f"{'Addr':<8} {'Row':<5} {'Col':<5} {'Float':<15} {'Hex':<8}\n")
            f.write("-"*60 + "\n")
            
            end_row = (gate_idx + 1) * 94 - 1
            for i in range(end_row, end_row + 1):
                for j in range(max(0, cols - 50), cols):
                    addr = i * cols + j
                    float_val = data[i, j]
                    fixed_val = float_to_sign_magnitude(float_val)
                    f.write(f"{addr:<8} {i:<5} {j:<5} {float_val:<15.8f} {fixed_val:04X}\n")
    
    print(f"✓ Summary verification: {summary_file}")
    
    # Calculate statistics
    print("\n" + "="*80)
    print(" STATISTICS")
    print("="*80)
    
    errors = []
    for i in range(rows):
        for j in range(cols):
            float_val = data[i, j]
            fixed_val = float_to_sign_magnitude(float_val)
            
            # Reconstruct
            sign = (fixed_val >> 15) & 1
            magnitude = fixed_val & 0x7FFF
            reconstructed = magnitude / 256.0
            if sign:
                reconstructed = -reconstructed
            
            errors.append(abs(float_val - reconstructed))
    
    errors = np.array(errors)
    
    print(f"\nQuantization Error (all {rows*cols} values):")
    print(f"  Mean:  {errors.mean():.8f}")
    print(f"  Max:   {errors.max():.8f}")
    print(f"  Min:   {errors.min():.8f}")
    print(f"  Std:   {errors.std():.8f}")
    
    print(f"\nResolution: {1/256:.8f} (S7.8 format)")

def create_testbench_example(output_file='verilog_usage_example.v'):
    """Create example Verilog code"""
    
    with open(output_file, 'w') as f:
        f.write("""// Example: How to use the weights.mem file in Verilog

module lstm_weights_memory (
    input wire clk,
    input wire [15:0] addr,        // Address: 0 to 37599
    output reg [15:0] weight_data  // Weight value (S7.8)
);

    // Memory array: 37600 words x 16 bits
    reg [15:0] weight_mem [0:37599];
    
    // Load weights from file
    initial begin
        $readmemh("weights_376x100.mem", weight_mem);
        $display("Loaded %0d weights", 37600);
    end
    
    // Read weight
    always @(posedge clk) begin
        weight_data <= weight_mem[addr];
    end
    
endmodule

// Example: Access specific weight
module lstm_weight_access_example;
    
    reg [15:0] weight_mem [0:37599];
    
    initial begin
        $readmemh("weights_376x100.mem", weight_mem);
        
        // Access weight at row=0, col=0
        $display("Weight[0][0] = %h", weight_mem[0]);
        
        // Access weight at row=5, col=10
        // Address = row * 100 + col = 5 * 100 + 10 = 510
        $display("Weight[5][10] = %h", weight_mem[510]);
        
        // Extract sign and magnitude
        automatic logic sign;
        automatic logic [14:0] magnitude;
        
        sign = weight_mem[0][15];
        magnitude = weight_mem[0][14:0];
        
        $display("Sign bit: %b, Magnitude: %h", sign, magnitude);
    end
    
endmodule
""")
    
    print(f"✓ Verilog example: {output_file}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    
    txt_file = "lstm_weights_separated/full/W_all_376x100.txt"
    mem_file = "weights_376x100.mem"
    
    print("\n" + "="*80)
    print(" CONVERT 376x100 TXT TO VERILOG MEMORY FILE")
    print("="*80)
    print(f"\nInput:  {txt_file}")
    print(f"Output: {mem_file}")
    print("\nFormat: S7.8 Sign-Magnitude (16-bit)")
    print("  Bit 15:    Sign (0=+, 1=-)")
    print("  Bits 14-8: Integer (7 bits)")
    print("  Bits 7-0:  Fraction (8 bits)")
    print("\n" + "="*80 + "\n")
    
    # Create memory file
    create_mem_file(txt_file, mem_file)
    
    # Create Verilog example
    create_testbench_example('verilog_usage_example.v')
    
    print("\n" + "="*80)
    print(" ✅ DONE!")
    print("="*80)
    print("\nFiles created:")
    print(f"  1. {mem_file}                      - Memory file (37,600 hex values)")
    print(f"  2. weights_376x100_verify_full.txt      - FULL verification (ALL 37,600 values)")
    print(f"  3. weights_376x100_verify_summary.txt   - Summary (first/last 50 per gate)")
    print(f"  4. verilog_usage_example.v              - Verilog example code")
    
    print("\n" + "="*80)
    print(" FILE SIZES")
    print("="*80)
    
    import os
    files = [
        mem_file,
        "weights_376x100_verify_full.txt",
        "weights_376x100_verify_summary.txt",
        "verilog_usage_example.v"
    ]
    
    for fname in files:
        if os.path.exists(fname):
            size = os.path.getsize(fname) / 1024  # KB
            print(f"  {fname:<45} {size:>10.2f} KB")
    
    print("\n" + "="*80)
    print(" USAGE IN VERILOG")
    print("="*80)
    print("""
// Declare memory
reg [15:0] weight_mem [0:37599];

// Load from file
initial begin
    $readmemh("weights_376x100.mem", weight_mem);
end

// Access weight at row i, column j
wire [15:0] weight = weight_mem[i * 100 + j];

// Extract components
wire sign = weight[15];              // 0=positive, 1=negative
wire [6:0] int_part = weight[14:8];  // Integer part
wire [7:0] frac_part = weight[7:0];  // Fractional part
    """)
    
    print("="*80)
    print("\n✅ FULL VERIFICATION FILE WITH ALL 37,600 VALUES CREATED!\n")