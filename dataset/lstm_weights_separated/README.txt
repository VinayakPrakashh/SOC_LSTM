================================================================================
 LSTM WEIGHTS - SEPARATED BY GATE
================================================================================

DIRECTORY STRUCTURE:
--------------------------------------------------------------------------------

lstm_weights_separated/
├── input/                 # Input gate weights
│   ├── W_ih_input_94x5.npy       (input-to-hidden)
│   ├── W_hh_input_94x94.npy      (hidden-to-hidden)
│   ├── b_ih_input_94.npy         (input bias)
│   ├── b_hh_input_94.npy         (hidden bias)
│   ├── b_combined_input_94.npy   (combined bias)
│   ├── input_gate_94x100.npy     (all concatenated)
│   └── *.txt versions
│
├── forget/                # Forget gate weights
│   └── (same structure as input/)
│
├── cell/                  # Cell gate weights
│   └── (same structure as input/)
│
├── output/                # Output gate weights
│   └── (same structure as input/)
│
└── full/                  # Complete matrices (all 4 gates)
    ├── W_ih_full_376x5.npy
    ├── W_hh_full_376x94.npy
    ├── b_combined_full_376.npy
    ├── W_all_376x100.npy        (complete 376x100 matrix)
    └── *.txt versions

================================================================================
 MATRIX DIMENSIONS
================================================================================

Per Gate (94 units per gate):
  W_ih:       94 x 5   (input-to-hidden weights)
  W_hh:       94 x 94  (hidden-to-hidden weights)
  b_ih:       94 x 1   (input bias)
  b_hh:       94 x 1   (hidden bias)
  b_combined: 94 x 1   (b_ih + b_hh)
  Concatenated: 94 x 100 ([W_ih | W_hh | bias])

Full (All 4 gates):
  W_ih:       376 x 5   (4 × 94 rows)
  W_hh:       376 x 94  (4 × 94 rows)
  b_combined: 376 x 1
  Complete:   376 x 100

================================================================================
 GATE ORGANIZATION
================================================================================

Rows in full matrices:
  [0:94]      - Input Gate
  [94:188]    - Forget Gate
  [188:282]   - Cell Gate
  [282:376]   - Output Gate

================================================================================
 USAGE EXAMPLES
================================================================================

Python - Load individual gate:
--------------------------------------------------------------------------------
import numpy as np

# Load input gate weights
W_ih_input = np.load('input/W_ih_input_94x5.npy')
W_hh_input = np.load('input/W_hh_input_94x94.npy')
bias_input = np.load('input/b_combined_input_94.npy')

# Or load the concatenated matrix
input_gate_all = np.load('input/input_gate_94x100.npy')


Python - Load full matrix:
--------------------------------------------------------------------------------
# Load complete 376x100 matrix
W_all = np.load('full/W_all_376x100.npy')

# Split by gates
input_gate = W_all[0:94, :]
forget_gate = W_all[94:188, :]
cell_gate = W_all[188:282, :]
output_gate = W_all[282:376, :]


================================================================================
 STATISTICS
================================================================================

INPUT GATE:
----------------------------------------
  W_ih range: [-0.735237, 0.584072]
  W_hh range: [-0.814147, 0.892773]
  bias range: [-1.264924, 0.704314]
  W_ih mean:  0.054297
  W_hh mean:  -0.019223
  bias mean:  -0.396099

FORGET GATE:
----------------------------------------
  W_ih range: [-0.779909, 0.625779]
  W_hh range: [-1.206314, 0.809021]
  bias range: [-0.712529, 0.419879]
  W_ih mean:  0.008368
  W_hh mean:  -0.014605
  bias mean:  -0.185323

CELL GATE:
----------------------------------------
  W_ih range: [-0.391514, 0.299556]
  W_hh range: [-0.538537, 0.964126]
  bias range: [-0.419873, 0.373696]
  W_ih mean:  0.002926
  W_hh mean:  -0.000734
  bias mean:  0.040912

OUTPUT GATE:
----------------------------------------
  W_ih range: [-0.373505, 0.547347]
  W_hh range: [-0.686327, 0.780596]
  bias range: [-1.279081, 0.500547]
  W_ih mean:  0.026301
  W_hh mean:  -0.019865
  bias mean:  -0.384435

FULL MATRICES:
----------------------------------------
  W_ih range: [-0.779909, 0.625779]
  W_hh range: [-1.206314, 0.964126]
  bias range: [-1.279081, 0.704314]
