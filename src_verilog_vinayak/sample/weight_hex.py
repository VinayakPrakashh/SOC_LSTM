import numpy as np

def create_readmemh_file_corrected():
    # Load the padded matrix
    W_padded = np.loadtxt("W_all_matrix_padded.csv", delimiter=",")
    print(f"Matrix shape: {W_padded.shape}")
    
    def float_to_fixed_point_sign_magnitude(val, frac_bits=6, total_bits=12):
        """Convert float to S1.5.6 sign-magnitude format (1 sign + 11 magnitude bits)"""
        # Get the magnitude
        magnitude = abs(val)
        
        # Scale by 2^frac_bits for fractional part
        scaled_magnitude = int(round(magnitude * (2**frac_bits)))
        
        # Limit to 11 bits (2^11 - 1 = 2047)
        scaled_magnitude = min(scaled_magnitude, 2047)
        
        # Set sign bit if negative
        if val < 0:
            result = 0x800 | scaled_magnitude  # Set MSB (bit 11) to 1
        else:
            result = scaled_magnitude
        
        return result & 0xFFF  # Mask to 12 bits
    
    # Create the corrected memory file
    with open("weight_matrix_corrected.mem", "w") as f:
        f.write("// Weight matrix memory initialization file\n")
        f.write("// Format: S1.5.6 Sign-Magnitude (1 sign + 5 integer + 6 fractional bits)\n")
        f.write("// MSB=1 for negative, MSB=0 for positive\n")
        f.write("// Matrix size: 16x16 (256 entries)\n")
        f.write("// Address mapping: addr = row * 16 + col\n\n")
        
        for row in range(16):
            for col in range(16):
                val = W_padded[row, col]
                fixed_val = float_to_fixed_point_sign_magnitude(val)
                addr = row * 16 + col
                
                # Write hex value with comment showing sign bit
                sign_bit = (fixed_val >> 11) & 1
                magnitude = fixed_val & 0x7FF
                f.write(f"{fixed_val:03X}  // [{row:2d}][{col:2d}] = {val:7.3f} (S={sign_bit}, Mag={magnitude:03X})\n")
    
    print("✓ Created weight_matrix_corrected.mem")
    
    # Create clean version
    with open("weight_matrix_corrected_clean.mem", "w") as f:
        for row in range(16):
            for col in range(16):
                val = W_padded[row, col]
                fixed_val = float_to_fixed_point_sign_magnitude(val)
                f.write(f"{fixed_val:03X}\n")
    
    print("✓ Created weight_matrix_corrected_clean.mem")
    
    # Show examples of the corrected conversion
    print("\nCorrected conversions (Sign-Magnitude):")
    print("Float    → Fixed → Hex   → Sign|Magnitude")
    print("-" * 45)
    examples = [0.1, -0.1, 0.2, -0.05, 0.5, -0.08]
    for val in examples:
        fixed = float_to_fixed_point_sign_magnitude(val)
        sign_bit = (fixed >> 11) & 1
        magnitude = fixed & 0x7FF
        print(f"{val:7.3f} → {fixed:5d} → {fixed:03X} → {sign_bit}|{magnitude:03X}")
    
    # Create verification showing the difference
    print(f"\nKey corrections:")
    print(f"-0.1: Old=FFA (two's complement) → New=806 (sign-magnitude)")
    print(f"-0.05: Old=FFD → New=803") 
    print(f"-0.08: Old=FFB → New=805")

# Run the corrected conversion
create_readmemh_file_corrected()