# This is a python script to run model by JR on Vertical transmission
# The Python code consists of 3 stages of vertical transmission: gene pool, proliferation and Holobiont selection


import numpy as np
import matplotlib.pyplot as plt

# Constants
Tm = 20  # Total number of time points
K = 20  # Carrying capacity
growth_factor = 0.02  # Bacterial growth factor

# Initialize the H matrix with random values for the shuffle stage
def shuffle_stage(Tm, K):
    return np.random.randint(0, K, size=(Tm + 1, K + 1))

# Apply the growth factor to each value in the H matrix for the proliferation stage
def proliferation_stage(H, growth_factor):
    return H * (1 + growth_factor)

# Selection stage: select the bacteria with the highest counts to pass to the next cycle
def selection_stage(H):
    # You might need more sophisticated logic here depending on your selection criteria
    return np.where(H >= np.max(H, axis=1, keepdims=True), H, 0)

# Main simulation loop
H = shuffle_stage(Tm, K)
for t in range(1, Tm + 1):
    H[t] = proliferation_stage(H[t - 1], growth_factor)
    H[t] = selection_stage(H[t])

    # Plotting the histogram for the current time point
    plt.figure(figsize=(8, 6))
    plt.hist(range(K + 1), bins=np.arange(K + 2) - 0.5, weights=H[t], edgecolor='black')
    plt.title(f'Histogram at time t={t}')
    plt.xlabel('Number of Bacteria')
    plt.ylabel('Count of Individuals')
    plt.xticks(range(K + 1))
    plt.grid(axis='y', alpha=0.5)
    plt.show()
