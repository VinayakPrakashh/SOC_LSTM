import numpy as np
import os

def lstm_with_concat():
    # -------------------------------
    # Inputs and previous states
    # -------------------------------
    x = np.array([3.7, -2.0, 25.0, -7.4, 2.5])     # Input vector, shape: (5,)
    h_prev = np.array([0.1, 0.2, 0.3, 0.4])        # Hidden state, shape: (4,)
    c_prev = np.array([0.5, 0.6, 0.7, 0.8])        # Cell state, shape: (4,)

    # -------------------------------
    # Concatenate input and hidden
    # -------------------------------
    x_concat = np.concatenate((x, h_prev))         # Shape: (5+4,) = (9,)
    print("x_concat shape:", x_concat.shape)

    # -------------------------------
    # Define weights for gates
    # -------------------------------
    Wxi = np.array([[0.1, 0.2, 0.0, -0.1, 0.1],
                    [0.2, 0.1, 0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.2, 0.1, 0.0],
                    [0.1, 0.0, 0.1, 0.2, 0.1]])   # Shape: (4,5)
    Whi = np.array([[0.1, 0.0, 0.1, 0.0],
                    [0.0, 0.1, 0.0, 0.1],
                    [0.1, 0.0, 0.1, 0.0],
                    [0.0, 0.1, 0.0, 0.1]])        # Shape: (4,4)

    Wxf = np.array([[0.5, 0.1, 0.0, 0.0, 0.0],
                    [0.1, 0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.1, 0.5, 0.0]])   # Shape: (4,5)
    Whf = np.array([[0.2, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.0, 0.2]])        # Shape: (4,4)

    Wxc = 0.5 * Wxi
    Whc = 0.3 * Whi
    Wxo = 0.8 * Wxi
    Who = 0.4 * Whi

    # -------------------------------
    # Concatenate input and hidden weights for each gate
    # Shape after hstack: (4, 5+4=9)
    # -------------------------------
    Wi = np.hstack((Wxi, Whi))  # Input gate weights, shape: (4,9)
    Wf = np.hstack((Wxf, Whf))  # Forget gate weights, shape: (4,9)
    Wc = np.hstack((Wxc, Whc))  # Cell candidate weights, shape: (4,9)
    Wo = np.hstack((Wxo, Who))  # Output gate weights, shape: (4,9)

    print("Wi shape:", Wi.shape)
    print("Wf shape:", Wf.shape)
    print("Wc shape:", Wc.shape)
    print("Wo shape:", Wo.shape)

    # -------------------------------
    # Stack all gates vertically: shape (16, 9)
    # -------------------------------
    W_all = np.vstack((Wi, Wf, Wc, Wo))
    print("W_all shape:", W_all.shape)

    # -------------------------------
    # Concatenate biases for all gates
    # Shape: (16,)
    # -------------------------------
    b_i = np.array([0.1, 0.0, 0.1, 0.2])
    b_f = np.array([1.0, 1.0, 1.0, 1.0])
    b_c = np.array([0.0, 0.1, 0.0, 0.1])
    b_o = np.array([0.2, 0.1, 0.2, 0.1])
    b_all = np.concatenate((b_i, b_f, b_c, b_o))
    print("b_all shape:", b_all.shape)

    # -------------------------------
    # Single matrix multiplication for all gates
    # z shape: (16,)
    # -------------------------------
    z = W_all @ x_concat + b_all
    print("z shape:", z.shape)

    # Split z into 4 parts (each 4 values)
    z_i, z_f, z_c, z_o = np.split(z, 4)
    print("z_i shape:", z_i.shape)
    print("z_f shape:", z_f.shape)
    print("z_c shape:", z_c.shape)
    print("z_o shape:", z_o.shape)

    # -------------------------------
    # Apply activations
    # -------------------------------
    i = 1 / (1 + np.exp(-z_i))
    f = 1 / (1 + np.exp(-z_f))
    g = np.tanh(z_c)
    o = 1 / (1 + np.exp(-z_o))

    # -------------------------------
    # Compute new cell and hidden states
    # -------------------------------
    c_t = f * c_prev + i * g
    h_t = o * np.tanh(c_t)

    # -------------------------------
    # Print results
    # -------------------------------
    print("\nInput gate (i):", i)
    print("Forget gate (f):", f)
    print("Cell candidate (g):", g)
    print("Output gate (o):", o)
    print("New cell state c_t:", c_t)
    print("New hidden state h_t:", h_t)

    # -------------------------------
    # Save to separate files
    # -------------------------------
    print("\n" + "="*50)
    print("SAVING FILES...")
    print("="*50)
    
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    try:
        # 1. Save W_all matrix (16x9)
        np.savetxt("W_all_matrix.csv", W_all, delimiter=",", fmt="%.6f")
        print("✓ W_all_matrix.csv saved successfully")
        print(f"  Shape: {W_all.shape}")
        
        # 2. Save x_concat vector (9,)
        np.savetxt("x_concat_vector.csv", x_concat, delimiter=",", fmt="%.6f")
        print("✓ x_concat_vector.csv saved successfully") 
        print(f"  Shape: {x_concat.shape}")
        
        # 3. Save bias vector (16,)
        np.savetxt("bias_vector.csv", b_all, delimiter=",", fmt="%.6f")
        print("✓ bias_vector.csv saved successfully")
        print(f"  Shape: {b_all.shape}")
        
        # 4. Save computation result z (16,)
        np.savetxt("z_result.csv", z, delimiter=",", fmt="%.6f")
        print("✓ z_result.csv saved successfully")
        print(f"  Shape: {z.shape}")
        
        # 5. Save final states
        final_states = np.concatenate((c_t, h_t))  # Combine c_t and h_t
        np.savetxt("final_states.csv", final_states, delimiter=",", fmt="%.6f")
        print("✓ final_states.csv saved successfully")
        print(f"  Shape: {final_states.shape} (first 4: c_t, last 4: h_t)")
        
        # 6. Save as NumPy binary files (more efficient for loading)
        np.save("W_all_matrix.npy", W_all)
        np.save("x_concat_vector.npy", x_concat)
        np.save("bias_vector.npy", b_all)
        np.save("z_result.npy", z)
        print("✓ NumPy binary files (.npy) also saved")
        
        # 7. Create a summary file
        with open("lstm_data_summary.txt", "w") as f:
            f.write("LSTM Data Files Summary\n")
            f.write("=====================\n\n")
            f.write(f"W_all_matrix.csv: Weight matrix, shape {W_all.shape}\n")
            f.write(f"x_concat_vector.csv: Input vector, shape {x_concat.shape}\n")
            f.write(f"bias_vector.csv: Bias vector, shape {b_all.shape}\n")
            f.write(f"z_result.csv: Matrix mult result, shape {z.shape}\n")
            f.write(f"final_states.csv: Final c_t and h_t, shape {final_states.shape}\n\n")
            f.write("Input values:\n")
            f.write(f"x = {x}\n")
            f.write(f"h_prev = {h_prev}\n")
            f.write(f"c_prev = {c_prev}\n\n")
            f.write("Gate results:\n")
            f.write(f"z_i = {z_i}\n")
            f.write(f"z_f = {z_f}\n")
            f.write(f"z_c = {z_c}\n")
            f.write(f"z_o = {z_o}\n\n")
            f.write("Final states:\n")
            f.write(f"c_t = {c_t}\n")
            f.write(f"h_t = {h_t}\n")
        print("✓ lstm_data_summary.txt saved successfully")
        
        # Verify files exist
        files_to_check = [
            "W_all_matrix.csv", "x_concat_vector.csv", "bias_vector.csv", 
            "z_result.csv", "final_states.csv", "lstm_data_summary.txt"
        ]
        
        print(f"\nFile verification:")
        for filename in files_to_check:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                print(f"  ✓ {filename} ({size} bytes)")
            else:
                print(f"  ✗ {filename} - NOT FOUND!")
                
    except Exception as e:
        print(f"ERROR saving files: {e}")

if __name__ == "__main__":
    lstm_with_concat()