import torch

MODEL_PATH = "soc_lstm_model_1layer.pth"
OUTPUT_FILE = "fc_fixed.mem"

FRAC_BITS = 20
SCALE = 1 << FRAC_BITS

def float_to_s3_20_signmag(x):
    sign = 1 if x < 0 else 0
    mag = int(round(abs(x) * SCALE))
    word = (sign << 23) | mag
    return f"{word:06X}"

print("Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
state_dict = checkpoint["model_state_dict"]

# Extract FC layer
weights = state_dict["fc.weight"].cpu().numpy().flatten()
bias = state_dict["fc.bias"].cpu().numpy().flatten()

all_values = list(weights) + list(bias)

print("Total values:", len(all_values))

with open(OUTPUT_FILE, "w") as f:
    for v in all_values:
        hx = float_to_s3_20_signmag(v)
        f.write(f"{hx}   // {v:.8f}\n")

print("✓ MEM file created:", OUTPUT_FILE)