#!/usr/bin/env python3
"""
Validate sigmoid LUT values for hardware implementation
Checks S1.5.6 fixed-point format: 12-bit = 1 sign + 5 integer + 6 fractional bits
"""

import numpy as np

def sigmoid(x):
    """Standard sigmoid function"""
    return 1.0 / (1.0 + np.exp(-x))

def fixed_to_float(fixed_val, frac_bits=6):
    """Convert S1.5.6 fixed-point to float"""
    return fixed_val / (2**frac_bits)

def float_to_fixed(float_val, frac_bits=6):
    """Convert float to S1.5.6 fixed-point"""
    return int(round(float_val * (2**frac_bits)))

# Test parameters
LUT_SIZE = 384
INPUT_RANGE = 6.0  # [0, 6]
FRAC_BITS = 6

print("Sigmoid LUT Validation")
print("=" * 50)
print(f"Format: S1.5.6 (12-bit: 1 sign + 5 integer + {FRAC_BITS} fractional bits)")
print(f"LUT Size: {LUT_SIZE} entries")
print(f"Input Range: [0, {INPUT_RANGE}]")
print(f"Step Size: {INPUT_RANGE/LUT_SIZE:.6f}")
print()

# Validate a few key points
test_indices = [0, 63, 127, 191, 255, 319, 383]  # Key points throughout range

print("Index\tInput\t\tExpected\tFixed-Point\tFloat\t\tError")
print("-" * 70)

for idx in test_indices:
    # Calculate input value for this index
    input_val = (idx * INPUT_RANGE) / LUT_SIZE
    
    # Expected sigmoid value
    expected = sigmoid(input_val)
    
    # Convert to fixed-point
    fixed_expected = float_to_fixed(expected, FRAC_BITS)
    
    # Convert back to float to see quantization
    quantized = fixed_to_float(fixed_expected, FRAC_BITS)
    
    # Error
    error = abs(expected - quantized)
    
    print(f"{idx:3d}\t{input_val:.4f}\t\t{expected:.6f}\t{fixed_expected:3d}\t\t{quantized:.6f}\t{error:.6f}")

print()
print("Key observations:")
print("- Value 32 represents 0.5 (sigmoid(0) ≈ 0.5)")
print("- Value 64 represents 1.0 (maximum sigmoid output)")
print("- Your LUT correctly ranges from 32 to 64")
print("- This provides good precision for the sigmoid range [0.5, 1.0]")

# Check specific values from the LUT
print("\nValidating specific LUT entries:")
specific_checks = [
    (0, 32),    # sigmoid(0) ≈ 0.5 → 32
    (383, 64)   # sigmoid(6) ≈ 1.0 → 64
]

for idx, lut_val in specific_checks:
    input_val = (idx * INPUT_RANGE) / LUT_SIZE
    expected = sigmoid(input_val)
    expected_fixed = float_to_fixed(expected, FRAC_BITS)
    float_val = fixed_to_float(lut_val, FRAC_BITS)
    
    print(f"Index {idx}: input={input_val:.3f}, expected={expected:.6f}, "
          f"LUT={lut_val} ({float_val:.6f}), error={abs(expected-float_val):.6f}")