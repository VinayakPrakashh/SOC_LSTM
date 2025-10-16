import numpy as np

print("=== LSTM Data Loader and Verifier ===\n")

# Load from CSV
print("Loading from CSV files...")
W_all = np.loadtxt("W_all_matrix.csv", delimiter=",")
x_concat = np.loadtxt("x_concat_vector.csv", delimiter=",")
bias = np.loadtxt("bias_vector.csv", delimiter=",")
z_original = np.loadtxt("z_result.csv", delimiter=",")

print(f"✓ W_all shape: {W_all.shape}")
print(f"✓ x_concat shape: {x_concat.shape}")
print(f"✓ bias shape: {bias.shape}")
print(f"✓ z_original shape: {z_original.shape}")

# Also load from NumPy binary (faster)
print("\nLoading from NumPy binary files...")
W_all_npy = np.load("W_all_matrix.npy")
x_concat_npy = np.load("x_concat_vector.npy")
bias_npy = np.load("bias_vector.npy")

print(f"✓ Binary files loaded successfully")

# Verify the computation
print("\n=== Computation Verification ===")
z_check = W_all @ x_concat + bias
computation_matches = np.allclose(z_original, z_check)
print("Computation matches:", computation_matches)

# Show the actual values
print("\n=== Input Values ===")
print("x_concat (input vector):")
print(x_concat)

print("\nW_all (weight matrix 16x9):")
print(W_all)

print("\nbias (bias vector):")
print(bias)

print("\n=== Computation Results ===")
print("Original z (from file):")
print(z_original)

print("\nRecomputed z (W_all @ x_concat + bias):")
print(z_check)

print("\nDifference (should be near zero):")
print(z_original - z_check)

# Split results into gates
print("\n=== Gate-wise Results ===")
z_i = z_check[0:4]
z_f = z_check[4:8]
z_c = z_check[8:12]
z_o = z_check[12:16]

print("z_i (Input gate pre-activations):")
print(z_i)
print("z_f (Forget gate pre-activations):")
print(z_f)
print("z_c (Cell candidate pre-activations):")
print(z_c)
print("z_o (Output gate pre-activations):")
print(z_o)

# Apply activations to see final gate values
print("\n=== Final Gate Activations ===")
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

i_gate = sigmoid(z_i)
f_gate = sigmoid(z_f)
g_gate = np.tanh(z_c)
o_gate = sigmoid(z_o)

print("i_gate (Input gate - sigmoid):")
print(i_gate)
print("f_gate (Forget gate - sigmoid):")
print(f_gate)
print("g_gate (Cell candidate - tanh):")
print(g_gate)
print("o_gate (Output gate - sigmoid):")
print(o_gate)

# Load and show final states if available
try:
    final_states = np.loadtxt("final_states.csv", delimiter=",")
    c_t = final_states[0:4]
    h_t = final_states[4:8]
    
    print("\n=== Final States ===")
    print("c_t (new cell state):")
    print(c_t)
    print("h_t (new hidden state):")
    print(h_t)
    
except FileNotFoundError:
    print("\nfinal_states.csv not found - skipping final states display")

print("\n=== Summary ===")
print(f"Matrix multiplication: {W_all.shape} @ {x_concat.shape} + {bias.shape} = {z_check.shape}")
print(f"Verification: {'✓ PASSED' if computation_matches else '✗ FAILED'}")