import math
import matplotlib.pyplot as plt
import numpy as np

def piecewise_tanh(x):
    """
    Piecewise tanh function:
    - For 0 < x < 0.25: tanh(x) = x
    - For 0.25 <= x <= 3: tanh(x) = table_value(x)
    - For x > 3: tanh(x) = 1
    - For x < 0: apply symmetric conditions (tanh(-x) = -tanh(x))
    """
    
    # Tanh lookup table for range [0.25, 3.0]
    # Pre-computed tanh values at specific points
    table_x = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    table_y = [0.24492, 0.46212, 0.63515, 0.76159, 0.84829, 0.90515, 0.94138, 
               0.96403, 0.97754, 0.98661, 0.99331, 0.99505]
    
    def get_table_value(x_val):
        """Get tanh value from table using linear interpolation"""
        if x_val <= 0.25:
            return table_y[0]
        if x_val >= 3.0:
            return table_y[-1]
        
        # Find the two points for interpolation
        for i in range(len(table_x) - 1):
            if table_x[i] <= x_val <= table_x[i + 1]:
                # Linear interpolation
                x1, y1 = table_x[i], table_y[i]
                x2, y2 = table_x[i + 1], table_y[i + 1]
                return y1 + (y2 - y1) * (x_val - x1) / (x2 - x1)
        
        return table_y[-1]  # fallback
    
    # Handle negative values using symmetry: tanh(-x) = -tanh(x)
    if x < 0:
        abs_x = -x
        if abs_x < 0.25:
            return -abs_x  # Linear region
        elif abs_x <= 3.0:
            return -get_table_value(abs_x)  # Table lookup
        else:
            return -1.0  # Saturation
    
    # Handle positive values
    elif x < 0.25:
        return x  # Linear region: tanh(x) = x
    elif x <= 3.0:
        return get_table_value(x)  # Table lookup
    else:
        return 1.0  # Saturation: tanh(x) = 1

def plot_tanh_comparison():
    """Plot comparison between piecewise and real tanh functions"""
    # Generate x values
    x = np.linspace(-4, 4, 1000)
    
    # Calculate y values for both functions
    y_piecewise = [piecewise_tanh(xi) for xi in x]
    y_real = [math.tanh(xi) for xi in x]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Main comparison plot
    plt.subplot(2, 1, 1)
    plt.plot(x, y_piecewise, 'b-', linewidth=2, label='Piecewise tanh', alpha=0.8)
    plt.plot(x, y_real, 'r--', linewidth=2, label='Real tanh (math.tanh)', alpha=0.8)
    
    # Add vertical lines to show boundaries
    plt.axvline(x=-3, color='gray', linestyle=':', alpha=0.7, label='Boundary at ±3')
    plt.axvline(x=3, color='gray', linestyle=':', alpha=0.7)
    plt.axvline(x=-0.25, color='orange', linestyle=':', alpha=0.7, label='Boundary at ±0.25')
    plt.axvline(x=0.25, color='orange', linestyle=':', alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add horizontal lines
    plt.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Saturation at ±1')
    plt.axhline(y=-1, color='green', linestyle=':', alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('tanh(x)')
    plt.title('Piecewise vs Real Tanh Function Comparison')
    plt.legend()
    plt.ylim(-1.2, 1.2)
    plt.xlim(-4, 4)
    
    # Error plot
    plt.subplot(2, 1, 2)
    error = [abs(y_piecewise[i] - y_real[i]) for i in range(len(x))]
    plt.plot(x, error, 'g-', linewidth=2, label='Absolute Error')
    plt.axvline(x=-3, color='gray', linestyle=':', alpha=0.7)
    plt.axvline(x=3, color='gray', linestyle=':', alpha=0.7)
    plt.axvline(x=-0.25, color='orange', linestyle=':', alpha=0.7)
    plt.axvline(x=0.25, color='orange', linestyle=':', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('|Error|')
    plt.title('Absolute Error between Piecewise and Real Tanh')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    max_error = max(error)
    mean_error = sum(error) / len(error)
    print(f"\nError Statistics:")
    print(f"Maximum absolute error: {max_error:.8f}")
    print(f"Mean absolute error: {mean_error:.8f}")
    
    # Show specific region errors
    linear_region = [(i, x[i]) for i in range(len(x)) if -0.25 <= x[i] <= 0.25]
    table_region = [(i, x[i]) for i in range(len(x)) if 0.25 <= abs(x[i]) <= 3.0]
    
    if linear_region:
        linear_errors = [error[i] for i, _ in linear_region]
        print(f"Linear region (-0.25 to 0.25) max error: {max(linear_errors):.8f}")
    
    if table_region:
        table_errors = [error[i] for i, _ in table_region]
        print(f"Table region (0.25 to 3.0) max error: {max(table_errors):.8f}")

def plot_regions_detail():
    """Plot detailed view of different regions"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Region 1: Linear region around zero
    x1 = np.linspace(-0.3, 0.3, 200)
    y1_pw = [piecewise_tanh(xi) for xi in x1]
    y1_real = [math.tanh(xi) for xi in x1]
    
    axes[0, 0].plot(x1, y1_pw, 'b-', linewidth=2, label='Piecewise')
    axes[0, 0].plot(x1, y1_real, 'r--', linewidth=2, label='Real tanh')
    axes[0, 0].axvline(x=-0.25, color='orange', linestyle=':', alpha=0.7)
    axes[0, 0].axvline(x=0.25, color='orange', linestyle=':', alpha=0.7)
    axes[0, 0].set_title('Linear Region (-0.3 to 0.3)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Region 2: Table region
    x2 = np.linspace(0.2, 3.2, 200)
    y2_pw = [piecewise_tanh(xi) for xi in x2]
    y2_real = [math.tanh(xi) for xi in x2]
    
    axes[0, 1].plot(x2, y2_pw, 'b-', linewidth=2, label='Piecewise')
    axes[0, 1].plot(x2, y2_real, 'r--', linewidth=2, label='Real tanh')
    axes[0, 1].axvline(x=0.25, color='orange', linestyle=':', alpha=0.7)
    axes[0, 1].axvline(x=3, color='gray', linestyle=':', alpha=0.7)
    axes[0, 1].set_title('Table Region (0.2 to 3.2)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Region 3: Saturation region
    x3 = np.linspace(2.5, 5, 200)
    y3_pw = [piecewise_tanh(xi) for xi in x3]
    y3_real = [math.tanh(xi) for xi in x3]
    
    axes[1, 0].plot(x3, y3_pw, 'b-', linewidth=2, label='Piecewise')
    axes[1, 0].plot(x3, y3_real, 'r--', linewidth=2, label='Real tanh')
    axes[1, 0].axvline(x=3, color='gray', linestyle=':', alpha=0.7)
    axes[1, 0].axhline(y=1, color='green', linestyle=':', alpha=0.5)
    axes[1, 0].set_title('Saturation Region (2.5 to 5)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Region 4: Negative region (symmetry)
    x4 = np.linspace(-3.2, -0.2, 200)
    y4_pw = [piecewise_tanh(xi) for xi in x4]
    y4_real = [math.tanh(xi) for xi in x4]
    
    axes[1, 1].plot(x4, y4_pw, 'b-', linewidth=2, label='Piecewise')
    axes[1, 1].plot(x4, y4_real, 'r--', linewidth=2, label='Real tanh')
    axes[1, 1].axvline(x=-0.25, color='orange', linestyle=':', alpha=0.7)
    axes[1, 1].axvline(x=-3, color='gray', linestyle=':', alpha=0.7)
    axes[1, 1].set_title('Negative Region (-3.2 to -0.2)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

# Test function
def test_piecewise_tanh():
    """Test the piecewise tanh function"""
    test_values = [-4, -3, -2, -1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1, 2, 3, 4]
    
    print("Testing Piecewise Tanh Function:")
    print("x\t\tPiecewise\tBuilt-in\tError")
    print("-" * 45)
    
    for x in test_values:
        pw_result = piecewise_tanh(x)
        builtin_result = math.tanh(x)
        error = abs(pw_result - builtin_result)
        print(f"{x:6.2f}\t\t{pw_result:8.5f}\t\t{builtin_result:8.5f}\t\t{error:8.6f}")

if __name__ == "__main__":
    # Test the function
    test_piecewise_tanh()
    
    # Plot comparisons
    print("\nGenerating plots...")
    plot_tanh_comparison()
    plot_regions_detail()
    
    # Example usage
    print(f"\nExample Usage:")
    print(f"piecewise_tanh(0.1) = {piecewise_tanh(0.1):.5f}")
    print(f"piecewise_tanh(1.0) = {piecewise_tanh(1.0):.5f}")
    print(f"piecewise_tanh(-1.0) = {piecewise_tanh(-1.0):.5f}")
    print(f"piecewise_tanh(3.5) = {piecewise_tanh(3.5):.5f}")
    print(f"piecewise_tanh(-3.5) = {piecewise_tanh(-3.5):.5f}")