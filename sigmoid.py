import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Generate x values from -10 to 10
x_vals = [i * 0.5 for i in range(-20, 21)]
y_vals = [sigmoid(x) for x in x_vals]

# Simple ASCII plot
for x, y in zip(x_vals, y_vals):
    bar = '*' * int(y * 40)
    print(f"x={x:5.2f} | {bar}")

# If you want a graphical plot and have matplotlib, uncomment below:
import matplotlib.pyplot as plt
plt.plot(x_vals, y_vals)
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.title("Sigmoid Activation Function")
plt.grid(True)
plt.show()