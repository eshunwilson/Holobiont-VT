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
