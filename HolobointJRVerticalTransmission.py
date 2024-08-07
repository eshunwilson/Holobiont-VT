# This is a python script to run model by JR on Vertical transmission
# The Python code consists of 3 stages of vertical transmission: gene pool, proliferation and Holobiont selection


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
Tm = 20  # Total number of time points
K = 20  # Carrying capacity
growth_factor = 0.02  # Bacterial growth factor

#Initiation conditions:

# Initialize the H matrix with random values for the shuffle stage
def shuffle_stage(Tm, K):
    return np.random.randint(0, K, size=(Tm + 1, K + 1))
#TM stage
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

# Apply the growth factor to each value in the H matrix for the proliferation stage
def proliferation_stage(H, growth_factor):
    return H * (1 + growth_factor)
#Application of Generalized Lotka Volterra Model
#Example where one species outcompetes another for resources (functional role)
# Define the generalized Lotka-Volterra model for two competing bacterial species
def lotka_volterra(y, t, r, A):
    """
    Generalized Lotka-Volterra model for two competing bacterial species
    
    Parameters:
        y : array_like
            Array containing the population of each species.
        t : float
            Time variable.
        r : array_like
            Array containing intrinsic growth rates of each species.
        A : array_like
            Interaction matrix representing interactions between species.
    Returns:
        dydt : array_like
# Array containing the derivative of population with respect to time for each species.
    """
    dydt = y * (r + np.dot(A, y))
    return dydt

# Parameters
r = np.array([0.1, 0.05])  # Intrinsic growth rates (Species 1 initially has higher growth rate)
A = np.array([[-0.01, 0.], [0., -0.02]])  # Interaction matrix (negative interaction between species)
y0 = np.array([10, 5])  # Initial population of each species

# Time points
t = np.linspace(0, 100, 1000)

# Solve the ODE
y = odeint(lotka_volterra, y0, t, args=(r, A))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, y[:, 0], label='Species 1')
plt.plot(t, y[:, 1], label='Species 2')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Competition Between Two Bacterial Species')
plt.legend()
plt.grid(True)
plt.show()


### Trait-based Model Version 2
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
r_i_max = 1.0       # Maximum intrinsic growth rate
R_alpha = 10.0      # Resource availability
K_i_alpha = 0.5     # Half-saturation constant
S_i = 0.1           # Sensitivity to temperature
T_ref = 20.0        # Reference temperature
T = 25.0            # Current temperature
m_i = 0.1           # Mortality rate
initial_B_i = 1.0   # Initial biomass/population size

# Define the function for the growth rate
def growth_rate(R_alpha, K_i_alpha, S_i, T, T_ref):
    return (r_i_max * R_alpha / (R_alpha + K_i_alpha)) * np.exp(S_i * (T - T_ref))

# Time parameters
time_steps = 100
dt = 0.1  # Time step size

# Initialize arrays to store population sizes and time
B_i = np.zeros(time_steps)
B_i[0] = initial_B_i
time = np.linspace(0, time_steps * dt, time_steps)

# Run the simulation using Euler's method
for t in range(1, time_steps):
    r_i = growth_rate(R_alpha, K_i_alpha, S_i, T, T_ref)
    dB_i_dt = (r_i - m_i) * B_i[t-1]
    B_i[t] = B_i[t-1] + dB_i_dt * dt

# Plot the results
plt.plot(time, B_i, label='Population Size')
plt.xlabel('Time')
plt.ylabel('Biomass (B_i)')
plt.title('Trait-based Model Simulation')
plt.legend()
plt.show()
