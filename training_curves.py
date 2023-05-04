import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

with open("dsrnngan/logs/log_normal.out", "r") as file:
    data = file.read()

# Define the regular expression pattern
pattern = r"\b(D[0-3]|G[0-2])\s*:\s*([-+]?\d*\.\d+([eE][-+]?\d+)?|\d+([eE][-+]?\d+)?)\b"

# Find all matches and store them in a dictionary
matches = re.findall(pattern, data)
values = {"D0": [], "D1": [], "D2": [], "D3": [], "G0": [], "G1": [], "G2": []}

# Iterate over matches and append values to the corresponding key in the values dictionary
for match in matches:
    key, value = match[0], float(match[1])
    values[key].append(value)

# Store the values in separate arrays
D0 = np.array(values["D0"])
D1 = np.array(values["D1"])
D2 = np.array(values["D2"])
D3 = np.array(values["D3"])
G0 = np.array(values["G0"])
G1 = np.array(values["G1"])
G2 = np.array(values["G2"])

fig, ax = plt.subplots(figsize=(25, 6))

# Plot the values
ax.plot(np.arange(len(D0)), D0)
ax.set_xlabel("Iteration")
ax.set_ylabel("D0 Value")
ax.set_title("D0 vs. Iteration")
plt.savefig("D0", facecolor='white', transparent=False)
plt.show()