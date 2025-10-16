import numpy as np

# LSTM with hidden-to-hidden weights
def lstm_with_hidden():
    x = np.array([3.7, -2.0, 25.0, -7.4, 2.5])
    h_prev = np.array([0.1, 0.2, 0.3, 0.4])  # Previous hidden state
    c_prev = np.array([0.5, 0.6, 0.7, 0.8])  # Previous cell state
    
    # Input-to-hidden weights [4x5]
    W_ih_i = np.array([[0.1, 0.2, 0.0, -0.1, 0.1],
                       [0.2, 0.1, 0.1, 0.0, 0.0], 
                       [0.0, 0.1, 0.2, 0.1, 0.0],
                       [0.1, 0.0, 0.1, 0.2, 0.1]])
    
    # Hidden-to-hidden weights [4x4]
    W_hh_i = np.array([[0.1, 0.0, 0.1, 0.0],
                       [0.0, 0.1, 0.0, 0.1],
                       [0.1, 0.0, 0.1, 0.0], 
                       [0.0, 0.1, 0.0, 0.1]])
    
    W_ih_f = np.array([[0.5, 0.1, 0.0, 0.0, 0.0],
                       [0.1, 0.5, 0.0, 0.0, 0.0],
                       [0.0, 0.1, 0.5, 0.0, 0.0], 
                       [0.0, 0.0, 0.1, 0.5, 0.0]])
    
    W_hh_f = np.array([[0.2, 0.0, 0.0, 0.0],
                       [0.0, 0.2, 0.0, 0.0],
                       [0.0, 0.0, 0.2, 0.0],
                       [0.0, 0.0, 0.0, 0.2]])
    
    # Other gates similar...
    W_ih_c = W_ih_i * 0.5  # Simplified
    W_hh_c = W_hh_i * 0.3
    W_ih_o = W_ih_i * 0.8
    W_hh_o = W_hh_i * 0.4
    
    b_i = np.array([0.1, 0.0, 0.1, 0.2])
    b_f = np.array([1.0, 1.0, 1.0, 1.0])
    b_c = np.array([0.0, 0.1, 0.0, 0.1])
    b_o = np.array([0.2, 0.1, 0.2, 0.1])
    
    # LINEAR OUTPUTS BEFORE ACTIVATION
    linear_i = W_ih_i @ x + W_hh_i @ h_prev + b_i
    linear_f = W_ih_f @ x + W_hh_f @ h_prev + b_f
    linear_c = W_ih_c @ x + W_hh_c @ h_prev + b_c
    linear_o = W_ih_o @ x + W_hh_o @ h_prev + b_o
    
    print(f"Input gate linear:  {linear_i}")
    print(f"Forget gate linear: {linear_f}")
    print(f"Cell gate linear:   {linear_c}")
    print(f"Output gate linear: {linear_o}")
    
    # Gates with both input-to-hidden AND hidden-to-hidden
    i = 1/(1+np.exp(-linear_i))
    f = 1/(1+np.exp(-linear_f))
    c_tilde = np.tanh(linear_c)
    o = 1/(1+np.exp(-linear_o))
    print(f"Input gate :  {i}")
    print(f"Forget gate : {f}")
    print(f"Cell gate :   {c_tilde}")
    print(f"Output gate : {o}")
    # Update states
    c_t = f * c_prev + i * c_tilde
    h_t = o * np.tanh(c_t)
    
    print(f"Input x: {x}")
    print(f"h_prev: {h_prev}")
    print(f"c_prev: {c_prev}")
    print(f"c_t: {c_t}")
    print(f"h_t: {h_t}")

lstm_with_hidden()