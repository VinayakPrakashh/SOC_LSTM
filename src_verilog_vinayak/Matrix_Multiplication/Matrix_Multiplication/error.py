import numpy as np
import matplotlib.pyplot as plt

def float_to_fixed_32bit(value):
    """Convert float to 32-bit fixed point (1 sign + 10 int + 21 frac)"""
    # Clamp to representable range [-1024, 1023.9999995231628]
    clamped = np.clip(value, -1024.0, 1023.9999995231628)
    
    # Scale by 2^21 (2097152) for 21 fractional bits
    scaled = clamped * 2097152
    
    # Round to nearest integer
    quantized = np.round(scaled)
    
    # Convert back to float
    fixed_point_value = quantized / 2097152
    
    return fixed_point_value

def analyze_fixed_point_error():
    """Analyze quantization error for 32-bit fixed point format"""
    
    # Test range covering typical LSTM values
    test_values = np.linspace(-1024, 1023.999, 20000)
    
    # Convert to fixed point
    fixed_point_values = np.array([float_to_fixed_32bit(val) for val in test_values])
    
    # Calculate quantization error
    quantization_error = test_values - fixed_point_values
    
    return test_values, fixed_point_values, quantization_error

def plot_fixed_point_analysis():
    """Create comprehensive fixed-point error analysis plots"""
    
    # Analyze quantization error
    original, fixed_point, errors = analyze_fixed_point_error()
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('32-bit Fixed Point Analysis for LSTM Processing Element\n(1 sign + 10 integer + 21 fractional bits)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Original vs Fixed Point Values
    ax1.plot(original, original, 'b-', label='Floating Point (Original)', alpha=0.7)
    ax1.plot(original, fixed_point, 'r-', label='32-bit Fixed Point', alpha=0.8)
    ax1.set_xlabel('Input Value')
    ax1.set_ylabel('Represented Value')
    ax1.set_title('Floating Point vs 32-bit Fixed Point')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-5, 5])  # Focus on common LSTM range
    ax1.set_ylim([-5, 5])
    
    # Plot 2: Quantization Error
    ax2.plot(original, errors * 1000000, 'r-', linewidth=1)  # Convert to micro-units
    ax2.set_xlabel('Input Value')
    ax2.set_ylabel('Quantization Error (×10⁻⁶)')
    ax2.set_title('Quantization Error vs Input Value')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-5, 5])
    
    # Add quantization step lines
    step_size = 1/2097152  # 2^-21
    ax2.axhline(y=step_size/2 * 1000000, color='g', linestyle='--', alpha=0.7, 
                label=f'Max Error = ±{step_size/2:.9f}')
    ax2.axhline(y=-step_size/2 * 1000000, color='g', linestyle='--', alpha=0.7)
    ax2.legend()
    
    # Plot 3: Error Distribution Histogram
    ax3.hist(errors * 1000000, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('Quantization Error (×10⁻⁶)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Quantization Errors')
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))
    ax3.text(0.05, 0.95, f'Mean |Error|: {mean_error*1000000:.6f}×10⁻⁶\nMax |Error|: {max_error*1000000:.6f}×10⁻⁶', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 4: LSTM-specific error analysis
    # Typical LSTM values: weights [-1, 1], activations [0, 1], inputs [battery range]
    lstm_ranges = {
        'Weights': np.linspace(-1, 1, 1000),
        'Sigmoid/Tanh': np.linspace(0, 1, 1000), 
        'Battery Voltage': np.linspace(2.5, 4.2, 1000),
        'Battery Current': np.linspace(-5, 5, 1000)
    }
    
    colors = ['blue', 'green', 'red', 'purple']
    for i, (name, values) in enumerate(lstm_ranges.items()):
        fixed_vals = np.array([float_to_fixed_32bit(val) for val in values])
        errors = values - fixed_vals
        ax4.plot(values, np.abs(errors) * 1000000, color=colors[i], label=name, alpha=0.8)
    
    ax4.set_xlabel('Input Value')
    ax4.set_ylabel('Absolute Error (×10⁻⁶)')
    ax4.set_title('Absolute Error for Different LSTM Data Types')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("="*60)
    print("32-BIT FIXED POINT ANALYSIS FOR YOUR PROCESSING ELEMENT")
    print("="*60)
    print(f"Format: 1 sign + 10 integer + 21 fractional bits")
    print(f"Range: [{-1024:.9f}, {1023.9999995231628:.9f}]")
    print(f"Resolution: {1/2097152:.9f} (2^-21)")
    print(f"Maximum quantization error: ±{1/4194304:.9f}")
    print()
    
    # Analyze specific LSTM scenarios
    test_cases = {
        'LSTM Weight (0.5)': 0.5,
        'LSTM Weight (-0.3)': -0.3, 
        'Sigmoid Output (0.7)': 0.7,
        'Battery Voltage (3.7V)': 3.7,
        'Battery Current (2.1A)': 2.1,
        'Small Weight (0.001)': 0.001,
        'Tiny Weight (0.0000001)': 0.0000001
    }
    
    print("QUANTIZATION ERROR FOR TYPICAL LSTM VALUES:")
    print("-" * 60)
    for name, value in test_cases.items():
        fixed_val = float_to_fixed_32bit(value)
        error = value - fixed_val
        relative_error = (error / value * 100) if value != 0 else 0
        print(f"{name:25s}: {value:12.9f} → {fixed_val:12.9f} "
              f"(error: {error:+15.9f}, {relative_error:+8.6f}%)")
    
    print()
    print("HARDWARE IMPLICATIONS:")
    print("-" * 30)
    print("✅ Extremely high precision for LSTM weights and activations")
    print("✅ Excellent range for battery measurement values")
    print("✅ Near-zero quantization noise for typical operations") 
    print("✅ Can handle large accumulated sums without overflow")
    print("✅ Minimal precision loss in long accumulation chains")
    print("⚠️  Requires 32-bit multipliers and wider data paths")
    print("⚠️  Higher memory bandwidth and storage requirements")

# Run the analysis
plot_fixed_point_analysis()

# Additional analysis: Accumulation error over multiple operations
def analyze_accumulation_error():
    """Analyze error accumulation in dot products (like in your PE)"""
    
    print("\n" + "="*60)
    print("DOT PRODUCT ACCUMULATION ERROR ANALYSIS")
    print("="*60)
    
    # Simulate dot product accumulation (like in your PE)
    np.random.seed(42)
    
    # Test different vector lengths (5 for Layer 0, 94 for other layers)
    vector_lengths = [5, 94, 1000]  # Added 1000 for stress test
    
    for length in vector_lengths:
        layer_type = "0" if length == 5 else ("1-3" if length == 94 else "Stress Test")
        print(f"\nVector length: {length} (Layer {layer_type} operation)")
        print("-" * 50)
        
        # Generate typical LSTM data
        if length == 5:  # Battery inputs
            data_vec = np.random.uniform(0, 5, length)  # Battery measurements
        elif length == 94:  # Hidden states
            data_vec = np.random.uniform(0, 1, length)  # Sigmoid/tanh outputs
        else:  # Stress test
            data_vec = np.random.uniform(0, 1, length)  # Large vector
            
        weight_vec = np.random.uniform(-1, 1, length)  # LSTM weights
        
        # Compute floating point result
        fp_result = np.dot(data_vec, weight_vec)
        
        # Compute fixed point result (simulate your PE accumulation)
        fixed_data = np.array([float_to_fixed_32bit(val) for val in data_vec])
        fixed_weights = np.array([float_to_fixed_32bit(val) for val in weight_vec])
        
        # Simulate accumulation with quantization at each step
        acc = 0.0
        for i in range(length):
            product = fixed_data[i] * fixed_weights[i]
            acc += product
            acc = float_to_fixed_32bit(acc)  # Quantize accumulator
        
        fixed_result = acc
        
        # Calculate errors
        total_error = fp_result - fixed_result
        relative_error = (total_error / fp_result * 100) if fp_result != 0 else 0
        
        print(f"Floating point result: {fp_result:15.9f}")
        print(f"Fixed point result:    {fixed_result:15.9f}")
        print(f"Absolute error:        {total_error:+15.9f}")
        print(f"Relative error:        {relative_error:+10.6f}%")
        snr = 20*np.log10(abs(fp_result/total_error)) if total_error != 0 else float('inf')
        print(f"SNR (dB):              {snr:10.2f}")

analyze_accumulation_error()

# Compare with original 12-bit format
def compare_precisions():
    """Compare 32-bit vs 12-bit fixed point precision"""
    
    print("\n" + "="*60)
    print("PRECISION COMPARISON: 32-bit vs 12-bit Fixed Point")
    print("="*60)
    
    test_values = [0.5, -0.3, 0.001, 0.0000001, 3.7, 2.1]
    
    print(f"{'Value':<12} {'12-bit Error':<15} {'32-bit Error':<15} {'Improvement':<12}")
    print("-" * 60)
    
    for value in test_values:
        # 12-bit fixed point (original)
        fixed_12 = float_to_fixed_12bit(value)
        error_12 = abs(value - fixed_12)
        
        # 32-bit fixed point
        fixed_32 = float_to_fixed_32bit(value)
        error_32 = abs(value - fixed_32)
        
        improvement = error_12 / error_32 if error_32 > 0 else float('inf')
        
        print(f"{value:<12.7f} {error_12:<15.9f} {error_32:<15.9f} {improvement:<12.1f}x")

def float_to_fixed_12bit(value):
    """Original 12-bit fixed point for comparison"""
    clamped = np.clip(value, -32.0, 31.984375)
    scaled = clamped * 64
    quantized = np.round(scaled)
    return quantized / 64

compare_precisions()