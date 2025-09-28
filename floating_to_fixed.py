# Fixed-point parameters
WIDTH = 12       # total bits (1 sign + 5 integer + 6 fraction)
FRACTION = 6     # fractional bits
SCALE = 1 << FRACTION  # 2^6 = 64

# 1. Float to Fixed (Q5.6, signed)
def float_to_fixed(value):
    scaled = int(round(value * SCALE))  # scale & round
    if scaled < 0:  # convert negative to two’s complement
        scaled = (1 << WIDTH) + scaled
    return scaled & ((1 << WIDTH) - 1)  # fit into WIDTH bits

# 2. Fixed-point addition (signed wrap-around like hardware)
def fixed_add(a, b):
    return (a + b) & ((1 << WIDTH) - 1)

# 3. Fixed to Float (decode Q5.6 signed)
def fixed_to_float(fixed_val):
    if fixed_val & (1 << (WIDTH - 1)):  # check sign bit
        fixed_val -= (1 << WIDTH)
    return fixed_val / SCALE

# ----------------- TEST -----------------
if __name__ == "__main__":
    a_float = 12.84
    c_float = -2.5

    # Encode
    a_fixed = float_to_fixed(a_float)
    c_fixed = float_to_fixed(c_float)

    # Add
    sum_fixed = fixed_add(a_fixed, c_fixed)
    print(sum_fixed)
    sum_float = fixed_to_float(1658)

    print(f"A = {a_float} -> fixed {format(a_fixed, '012b')}")
    print(f"C = {c_float} -> fixed {format(c_fixed, '012b')}")
    print(f"A + C (fixed) = {format(sum_fixed, '012b')}, decoded = {sum_float}")
