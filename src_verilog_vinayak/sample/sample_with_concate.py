import numpy as np

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
    print("W_all shape:", W_all)

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
    z = W_all @ x_concat 
    print("z shape:", z)

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

lstm_with_concat()
