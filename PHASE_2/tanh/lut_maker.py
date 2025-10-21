import math

def float_to_s7_8(value):
    """Convert float to S7.8 format (16-bit signed fixed-point)"""
    # Clamp value to S7.8 range: [-128, 127.99609375]
    value = max(-128.0, min(127.99609375, value))
    
    # Scale by 256 and round
    scaled = round(value * 256)
    
    # Convert to 16-bit signed representation
    if scaled < 0:
        scaled = (1 << 16) + scaled  # Two's complement
    
    return scaled & 0xFFFF

def generate_tanh_verilog_file():
    """Generate complete Verilog file with tanh LUT"""
    
    input_min = 0.25
    input_max = 3.0
    step_size = 0.01
    
    # Calculate number of entries
    num_entries = int((input_max - input_min) / step_size) + 1
    addr_width = math.ceil(math.log2(num_entries))
    
    # Create output file with UTF-8 encoding
    output_file = "tanh_lut_generated.v"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("// Tanh LUT for S7.8 Fixed-Point Format\n")
        f.write("// Generated automatically - DO NOT EDIT MANUALLY\n")
        f.write(f"// Input range: [{input_min}, {input_max}]\n")
        f.write(f"// Step size: {step_size}\n")
        f.write(f"// Number of entries: {num_entries}\n")
        f.write(f"// Address width: {addr_width} bits\n")
        f.write("// Data format: S7.8 (16-bit signed, 8 fractional bits)\n\n")
        
        # Write module declaration
        f.write("module tanh_lut_s7_8 #(\n")
        f.write("    parameter WIDTH = 16,\n")
        f.write(f"    parameter ADDR_WIDTH = {addr_width},\n")
        f.write(f"    parameter LUT_SIZE = {num_entries}\n")
        f.write(") (\n")
        f.write("    input  [ADDR_WIDTH-1:0] addr,\n")
        f.write("    output [WIDTH-1:0] tanh_out\n")
        f.write(");\n\n")
        
        # Write LUT array declaration
        f.write("    // LUT data array\n")
        f.write("    reg [WIDTH-1:0] tanh_lut [0:LUT_SIZE-1];\n\n")
        
        # Write initialization block
        f.write("    // Initialize LUT with tanh values\n")
        f.write("    initial begin\n")
        
        # Generate LUT data
        for i in range(num_entries):
            input_val = input_min + i * step_size
            tanh_val = math.tanh(input_val)
            s7_8_val = float_to_s7_8(tanh_val)
            
            if i % 4 == 0:
                f.write(f"        tanh_lut[{i:4d}] = 16'h{s7_8_val:04X}; ")
            else:
                f.write(f"tanh_lut[{i:4d}] = 16'h{s7_8_val:04X}; ")
            
            if (i + 1) % 4 == 0:
                start_range = input_min + (i - 3) * step_size
                end_range = input_val
                f.write(f"// {start_range:.2f} to {end_range:.2f}\n")
            elif i == num_entries - 1:
                # Handle last line if not divisible by 4
                remaining = (i % 4) + 1
                start_idx = i - (remaining - 1)
                start_range = input_min + start_idx * step_size
                f.write(f"// {start_range:.2f} to {input_val:.2f}\n")
        
        f.write("    end\n\n")
        
        # Write output assignment
        f.write("    // Output assignment with bounds checking\n")
        f.write("    assign tanh_out = (addr < LUT_SIZE) ? tanh_lut[addr] : tanh_lut[LUT_SIZE-1];\n\n")
        
        f.write("endmodule\n\n")
        
        # Write address calculator module
        f.write("// Address calculator for tanh LUT\n")
        f.write("module tanh_addr_calculator #(\n")
        f.write("    parameter INPUT_WIDTH = 16,\n")
        f.write(f"    parameter ADDR_WIDTH = {addr_width},\n")
        f.write("    parameter FRAC_BITS = 8\n")
        f.write(") (\n")
        f.write("    input  [INPUT_WIDTH-1:0] input_value,    // S7.8 input value\n")
        f.write("    output [ADDR_WIDTH-1:0]  lut_addr,       // Address for LUT\n")
        f.write("    output                   addr_valid,      // Address is within valid range\n")
        f.write("    output                   use_symmetry,    // Use tanh symmetry for negative inputs\n")
        f.write("    output                   saturate_low,    // Input below minimum range\n")
        f.write("    output                   saturate_high    // Input above maximum range\n")
        f.write(");\n\n")
        
        # Write address calculator logic
        input_min_hex = float_to_s7_8(input_min)
        input_max_hex = float_to_s7_8(input_max)
        step_size_hex = float_to_s7_8(step_size)
        
        f.write("    // LUT parameters\n")
        f.write(f"    localparam [INPUT_WIDTH-1:0] INPUT_MIN = 16'h{input_min_hex:04X};  // {input_min} in S7.8\n")
        f.write(f"    localparam [INPUT_WIDTH-1:0] INPUT_MAX = 16'h{input_max_hex:04X};  // {input_max} in S7.8\n")
        f.write(f"    localparam [INPUT_WIDTH-1:0] STEP_SIZE = 16'h{step_size_hex:04X};  // {step_size} in S7.8\n")
        f.write(f"    localparam MAX_ADDR = {num_entries-1};\n\n")
        
        f.write("    wire signed [INPUT_WIDTH-1:0] signed_input;\n")
        f.write("    wire [INPUT_WIDTH-1:0] abs_input;\n")
        f.write("    wire input_negative;\n")
        f.write("    wire [INPUT_WIDTH-1:0] offset_input;\n")
        f.write("    wire [INPUT_WIDTH+7:0] scaled_addr_wide;\n")
        f.write("    wire [ADDR_WIDTH-1:0] scaled_addr;\n\n")
        
        f.write("    // Input processing\n")
        f.write("    assign signed_input = input_value;\n")
        f.write("    assign input_negative = signed_input[INPUT_WIDTH-1];\n")
        f.write("    assign abs_input = input_negative ? (~input_value + 1'b1) : input_value;\n\n")
        
        f.write("    // Check saturation conditions\n")
        f.write("    assign saturate_low = (abs_input < INPUT_MIN);\n")
        f.write("    assign saturate_high = (abs_input > INPUT_MAX);\n\n")
        
        f.write("    // Calculate address: (abs_input - INPUT_MIN) / STEP_SIZE\n")
        f.write("    assign offset_input = abs_input - INPUT_MIN;\n")
        f.write("    \n")
        f.write("    // Division by step size (approximation for 0.01)\n")
        f.write("    // Multiply by 100 to approximate division by 0.01\n")  # Fixed the Unicode issue
        f.write("    assign scaled_addr_wide = offset_input * 100;\n")
        f.write("    assign scaled_addr = scaled_addr_wide[15:8];\n\n")
        
        f.write("    // Generate final address with bounds checking\n")
        f.write("    assign lut_addr = saturate_low ? 0 :\n")
        f.write("                      saturate_high ? MAX_ADDR :\n")
        f.write("                      (scaled_addr > MAX_ADDR) ? MAX_ADDR :\n")
        f.write("                      scaled_addr;\n\n")
        
        f.write("    // Control signals\n")
        f.write("    assign addr_valid = ~saturate_low && ~saturate_high;\n")
        f.write("    assign use_symmetry = input_negative;\n\n")
        
        f.write("endmodule\n")
    
    print(f"Tanh LUT Verilog file generated: {output_file}")
    print(f"Total entries: {num_entries}")
    print(f"Address width: {addr_width} bits")
    print(f"Memory usage: {num_entries * 2} bytes")

def generate_verification_file():
    """Generate verification data for testbench"""
    
    verification_file = "tanh_verification_data.txt"
    
    with open(verification_file, 'w', encoding='utf-8') as f:  # Added UTF-8 encoding
        f.write("# Tanh Verification Data\n")
        f.write("# Format: Input_Decimal Input_Hex Tanh_Decimal Tanh_Hex\n")
        
        test_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
        
        for val in test_values:
            tanh_val = math.tanh(val)
            input_hex = float_to_s7_8(val)
            tanh_hex = float_to_s7_8(tanh_val)
            f.write(f"{val:.2f} 0x{input_hex:04X} {tanh_val:.6f} 0x{tanh_hex:04X}\n")
    
    print(f"Verification data generated: {verification_file}")

def print_sample_values():
    """Print sample tanh values for verification"""
    print("\n=== Sample Tanh Values (S7.8 Format) ===")
    print("Input\tTanh\tS7.8\tHex")
    print("-" * 40)
    
    test_values = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for val in test_values:
        tanh_val = math.tanh(val)
        s7_8_val = float_to_s7_8(tanh_val)
        print(f"{val:.2f}\t{tanh_val:.4f}\t{s7_8_val}\t0x{s7_8_val:04X}")

if __name__ == "__main__":
    generate_tanh_verilog_file()
    generate_verification_file()
    print_sample_values()