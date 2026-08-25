import torch
import numpy as np

def float_to_fixed_point_16bit(value, sign_bits=1, int_bits=7, frac_bits=8):
    """
    Convert float to 16-bit fixed point (sign-magnitude representation)
    Format: 1 sign bit + 7 integer bits + 8 fractional bits
    NOT 2's complement - MSB is just sign bit
    
    Returns: 16-bit binary string
    """
    # Clamp value to representable range
    max_val = (2**int_bits - 1) + (2**frac_bits - 1) / (2**frac_bits)
    min_val = -max_val
    
    if value > max_val:
        value = max_val
    elif value < min_val:
        value = min_val
    
    # Get sign bit
    sign_bit = '1' if value < 0 else '0'
    abs_value = abs(value)
    
    # Split into integer and fractional parts
    integer_part = int(abs_value)
    fractional_part = abs_value - integer_part
    
    # Convert integer part to binary (7 bits)
    int_binary = format(integer_part, f'0{int_bits}b')
    
    # Convert fractional part to binary (8 bits)
    frac_binary = ''
    for _ in range(frac_bits):
        fractional_part *= 2
        bit = int(fractional_part)
        frac_binary += str(bit)
        fractional_part -= bit
    
    # Combine: sign + integer + fractional
    fixed_point = sign_bit + int_binary + frac_binary
    
    return fixed_point

def generate_fc_mem_files(model_path="soc_lstm_model_1layer.pth"):
    """Generate .mem files for FC layer weights and biases"""
    
    print("="*70)
    print("GENERATING FC LAYER MEMORY FILES (16-bit Fixed Point)")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    
    # Define model structure to match checkpoint
    class LinearScratch(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
            self.bias = torch.nn.Parameter(torch.empty(out_features)) if bias else None
    
    class SoCLSTM(torch.nn.Module):
        def __init__(self, input_size=5, hidden_size=94, num_layers=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = LinearScratch(hidden_size, 1)
    
    model = SoCLSTM()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("✓ Model loaded")
    
    # Extract FC layer weights and bias
    fc_weight = model.fc.weight.data.numpy()  # Shape: (1, 94)
    fc_bias = model.fc.bias.data.numpy() if model.fc.bias is not None else np.array([0.0])  # Shape: (1,)
    
    print(f"\nFC Layer Information:")
    print(f"  Weight shape: {fc_weight.shape}")
    print(f"  Bias shape: {fc_bias.shape}")
    print(f"  Weight range: [{fc_weight.min():.6f}, {fc_weight.max():.6f}]")
    print(f"  Bias value: {fc_bias[0]:.6f}")
    
    # Flatten weights (1x94 -> 94 values)
    fc_weight_flat = fc_weight.flatten()
    
    print(f"\n{'='*70}")
    print("FIXED POINT FORMAT")
    print(f"{'='*70}")
    print("Format: 16-bit (1 sign + 7 integer + 8 fractional)")
    print("Sign: 0=positive, 1=negative")
    print("NOT 2's complement - MSB is sign bit only")
    print(f"{'='*70}\n")
    
    # Convert to fixed point and save weights
    print("Converting weights to fixed point...")
    with open('fc_weights.mem', 'w') as f:
        f.write("// FC Layer Weights (94 values)\n")
        f.write("// Format: 16-bit fixed point (1 sign + 7 int + 8 frac)\n")
        f.write("// Address | Binary Value | Decimal Value\n")
        
        for i, weight in enumerate(fc_weight_flat):
            fixed_point = float_to_fixed_point_16bit(weight)
            hex_val = format(int(fixed_point, 2), '04x')
            f.write(f"{fixed_point}  // [{i:2d}] 0x{hex_val} = {weight:12.8f}\n")
    
    print(f"✓ Saved fc_weights.mem (94 entries)")
    
    # Convert to fixed point and save bias
    print("Converting bias to fixed point...")
    with open('fc_bias.mem', 'w') as f:
        f.write("// FC Layer Bias (1 value)\n")
        f.write("// Format: 16-bit fixed point (1 sign + 7 int + 8 frac)\n")
        f.write("// Binary Value | Decimal Value\n")
        
        bias_val = fc_bias[0]
        fixed_point = float_to_fixed_point_16bit(bias_val)
        hex_val = format(int(fixed_point, 2), '04x')
        f.write(f"{fixed_point}  // 0x{hex_val} = {bias_val:12.8f}\n")
    
    print(f"✓ Saved fc_bias.mem (1 entry)")
    
    # Print sample conversions
    print(f"\n{'='*70}")
    print("SAMPLE CONVERSIONS (First 10 weights)")
    print(f"{'='*70}")
    print(f"{'Index':>5} | {'Float':>12} | {'Binary (16-bit)':>18} | {'Hex':>6}")
    print("-"*70)
    
    for i in range(min(10, len(fc_weight_flat))):
        weight = fc_weight_flat[i]
        fixed_point = float_to_fixed_point_16bit(weight)
        hex_val = format(int(fixed_point, 2), '04x')
        print(f"{i:5d} | {weight:12.8f} | {fixed_point} | 0x{hex_val}")
    
    print(f"\n{'='*70}")
    print("BIAS CONVERSION")
    print(f"{'='*70}")
    bias_val = fc_bias[0]
    fixed_point = float_to_fixed_point_16bit(bias_val)
    hex_val = format(int(fixed_point, 2), '04x')
    print(f"Float: {bias_val:12.8f}")
    print(f"Binary: {fixed_point}")
    print(f"Hex: 0x{hex_val}")
    
    # Verification: Convert back to float
    print(f"\n{'='*70}")
    print("VERIFICATION (First 5 weights)")
    print(f"{'='*70}")
    print(f"{'Index':>5} | {'Original':>12} | {'After Conv':>12} | {'Error':>12}")
    print("-"*70)
    
    def fixed_to_float(binary_str):
        """Convert 16-bit fixed point back to float"""
        sign_bit = int(binary_str[0])
        int_part = int(binary_str[1:8], 2)
        frac_part = int(binary_str[8:16], 2) / 256.0
        
        value = int_part + frac_part
        if sign_bit == 1:
            value = -value
        return value
    
    for i in range(min(5, len(fc_weight_flat))):
        original = fc_weight_flat[i]
        fixed_point = float_to_fixed_point_16bit(original)
        converted_back = fixed_to_float(fixed_point)
        error = abs(original - converted_back)
        print(f"{i:5d} | {original:12.8f} | {converted_back:12.8f} | {error:12.8e}")
    
    print(f"\n{'='*70}")
    print("✓ Memory files generated successfully!")
    print(f"{'='*70}")
    print(f"Files created:")
    print(f"  - fc_weights.mem (94 x 16-bit values)")
    print(f"  - fc_bias.mem (1 x 16-bit value)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    generate_fc_mem_files()