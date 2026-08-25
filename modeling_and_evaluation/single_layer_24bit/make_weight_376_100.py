import numpy as np
import os

def float_to_fixed_point_s3_20(value):
    """
    Convert floating point to 24-bit fixed point S3.20 format
    Format: 1 sign bit + 3 integer bits + 20 fractional bits = 24 bits total
    Sign-Magnitude representation (NOT two's complement)
    """
    max_val = 8.0 - 2**(-20)
    min_val = -8.0
    
    if value > max_val:
        print(f"Warning: Clamping {value} to {max_val}")
        value = max_val
    elif value < min_val:
        print(f"Warning: Clamping {value} to {min_val}")
        value = min_val
    
    # Extract sign
    sign_bit = 1 if value < 0 else 0
    
    # Work with magnitude
    magnitude = abs(value)
    
    # Scale magnitude by 2^20
    scaled = int(round(magnitude * (2**20)))
    
    # Combine: sign bit is MSB (bit 23), magnitude in bits [22:0]
    result = (sign_bit << 23) | (scaled & 0x7FFFFF)
    
    return result & 0xFFFFFF

def fixed_point_to_float_s3_20(fixed_val):
    """
    Convert 24-bit sign-magnitude S3.20 back to floating point
    """
    sign_bit = (fixed_val >> 23) & 1
    magnitude = fixed_val & 0x7FFFFF  # Lower 23 bits
    
    value = magnitude / (2**20)
    
    if sign_bit:
        value = -value
    
    return value

def process_weight_file(input_file, output_mem_file, output_verify_file):
    """
    Process weight file and generate memory file and verification file
    """
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"ERROR: File '{input_file}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print(f"Looking for: {os.path.abspath(input_file)}")
        return False
    
    print(f"Reading weights from: {input_file}")
    print(f"File size: {os.path.getsize(input_file)} bytes")
    
    # Read the weight file with multiple parsing attempts
    weights = []
    line_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line_count += 1
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip comment lines
            if line.startswith('#') or line.startswith('//'):
                continue
            
            # Try to parse as float
            try:
                # Handle scientific notation and various formats
                weight = float(line)
                weights.append(weight)
            except ValueError:
                # Try splitting by common delimiters
                parts = line.replace(',', ' ').replace('\t', ' ').split()
                for part in parts:
                    try:
                        weight = float(part)
                        weights.append(weight)
                    except ValueError:
                        if len(part) > 0:
                            print(f"Warning: Could not parse '{part}' on line {line_num}")
    
    print(f"Total lines read: {line_count}")
    print(f"Total weights parsed: {len(weights)}")
    
    # Check if we got any weights
    if len(weights) == 0:
        print("\nERROR: No weights were successfully parsed!")
        print("Please check the file format. Expected format:")
        print("  - One floating-point number per line")
        print("  - OR space/comma/tab separated values")
        print("\nFirst 10 lines of the file:")
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(f"  Line {i+1}: {repr(line.strip())}")
        return False
    
    # Statistics
    print(f"\nWeight Statistics:")
    print(f"  Min:    {min(weights):.6f}")
    print(f"  Max:    {max(weights):.6f}")
    print(f"  Mean:   {np.mean(weights):.6f}")
    print(f"  Median: {np.median(weights):.6f}")
    print(f"  Std:    {np.std(weights):.6f}")
    
    # Check if any values are out of range
    out_of_range = [w for w in weights if w < -8.0 or w >= 8.0]
    if out_of_range:
        print(f"\nWARNING: {len(out_of_range)} values out of S3.20 range [-8.0, 8.0):")
        for val in out_of_range[:10]:  # Show first 10
            print(f"  {val}")
    
    # Convert to fixed point
    print("\nConverting to S3.20 fixed-point format...")
    fixed_weights = []
    for i, weight in enumerate(weights):
        fixed_val = float_to_fixed_point_s3_20(weight)
        fixed_weights.append(fixed_val)
        
        # Show progress for large files
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(weights)} weights...")
    
    # Write memory file (hex format, one value per line)
    print(f"\nWriting memory file to: {output_mem_file}")
    with open(output_mem_file, 'w') as f:
        for fixed_val in fixed_weights:
            # Write as 6 hex digits (24 bits)
            f.write(f"{fixed_val:06X}\n")
    
    print(f"Memory file written: {len(fixed_weights)} entries")
    
    # Write verification file
    print(f"Writing verification file to: {output_verify_file}")
    with open(output_verify_file, 'w') as f:
        f.write("="*140 + "\n")
        f.write("S3.20 Fixed-Point Conversion Verification\n")
        f.write("Format: 1 sign bit + 3 integer bits + 20 fractional bits = 24 bits\n")
        f.write("Range: -8.0 to +7.999999...\n")
        f.write("="*140 + "\n\n")
        
        f.write(f"{'Index':<8}{'Original':<15}{'Hex':<10}{'Binary (S|III|FRAC...)':<30}{'Recovered':<15}{'Error':<15}\n")
        f.write("-"*140 + "\n")
        
        max_error = 0
        total_error = 0
        
        for i, (orig, fixed_val) in enumerate(zip(weights, fixed_weights)):
            recovered = fixed_point_to_float_s3_20(fixed_val)
            error = abs(orig - recovered)
            max_error = max(max_error, error)
            total_error += error
            
            # Convert to binary string for visualization
            binary_str = format(fixed_val, '024b')
            sign_bit = binary_str[0]
            integer_bits = binary_str[1:4]
            frac_bits = binary_str[4:14] + "..."  # Show first 10 frac bits
            binary_formatted = f"{sign_bit}|{integer_bits}|{frac_bits}"
            
            # Write to file
            f.write(f"{i:<8}{orig:<15.10f}0x{fixed_val:06X}   {binary_formatted:<30}{recovered:<15.10f}{error:<15.2e}\n")
            
            # Show samples in console (first 10, last 10, and some middle ones)
            if i < 10 or i >= len(weights) - 10 or (i % 5000 == 0):
                print(f"[{i:5d}] {orig:12.8f} -> 0x{fixed_val:06X} ({sign_bit}|{integer_bits}|...) -> {recovered:12.8f} (err: {error:.2e})")
        
        avg_error = total_error / len(weights) if len(weights) > 0 else 0
        f.write("\n" + "="*140 + "\n")
        f.write(f"Conversion Statistics:\n")
        f.write(f"  Total Weights:        {len(weights)}\n")
        f.write(f"  Maximum Error:        {max_error:.15e}\n")
        f.write(f"  Average Error:        {avg_error:.15e}\n")
        f.write(f"  Quantization Step:    {2**(-20):.15e} (2^-20)\n")
        f.write(f"  Bits Used:            24 bits (1 sign + 3 int + 20 frac)\n")
        f.write("="*140 + "\n")
        
        print(f"\n" + "="*80)
        print(f"Error Analysis:")
        print(f"  Maximum Error:     {max_error:.10e}")
        print(f"  Average Error:     {avg_error:.10e}")
        print(f"  Quantization Step: {2**(-20):.10e}")
        print("="*80)
    
    return True

def main():
    input_file = "timestep0_W_all_376x100.txt"
    output_mem_file = "lstm_weight_ih_l0_s3_20.mem"
    output_verify_file = "lstm_weight_ih_l0_s3_20_verify.txt"
    
    print("="*80)
    print("Fixed Point Converter: Float to S3.20 Format (24-bit)")
    print("Format: 1 sign bit + 3 integer bits + 20 fractional bits")
    print("Range: -8.0 to +7.999999...")
    print("Resolution: 2^-20 ≈ 9.54e-07")
    print("="*80 + "\n")
    
    success = process_weight_file(input_file, output_mem_file, output_verify_file)
    
    if success:
        print("\n" + "="*80)
        print("✓ Conversion Complete!")
        print(f"  Memory file:       {output_mem_file}")
        print(f"  Verification file: {output_verify_file}")
        print("\nYou can use this in Verilog with:")
        print(f'  $readmemh("{output_mem_file}", memory_array);')
        print("="*80)
    else:
        print("\n" + "="*80)
        print("✗ Conversion Failed!")
        print("Please check the error messages above.")
        print("="*80)

if __name__ == "__main__":
    main()