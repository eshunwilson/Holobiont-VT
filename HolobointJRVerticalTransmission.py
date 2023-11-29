# This is a python script to run model by JR on Vertical transmission
# The Python code consists of 3 stages of vertical transmission: gene pool, proliferation and Holobiont selection


#stage of proliferation:

def logistic_growth(K, R, F):
    population = [1]  # Initial population starts at 1 individual
    for step in range(1, F + 1):
        new_population = population[-1] + R * population[-1] * (1 - population[-1] / K)
        population.append(min(K, new_population))  # Cap population at K
    return population

# Example usage:
max_individuals = 1000  # K: Maximum number of individuals
growth_rate = 0.2  # R: Growth rate
num_generations = 20  # F: Number of steps of generations

result = logistic_growth(max_individuals, growth_rate, num_generations)
print(result)

#proliferation trial 2
def new_growth(K, R, F, P):
    for microbe in P:
        print("Microbe", microbe)

        for step in range(1, F + 1):
            new_population = population[-1] + R * population[-1] * (1 - population[-1] / K)
            population.append(min(K, new_population))  # Cap population at K



startpopulation = [1,3,4,3,2,1,4,3,2,1,2]
startpopulation1 = [1,3,4,3,2,1,4,3,2,1,2]

prolif = new_growth(1,2,3,startpopulation)


##trial november 21 2023 - how to plot the population
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Logistic growth function based on the differential equation
def logistic_growth(t, N, r, K):
    return r * N * (K - N) / K

# Parameters
r = 0.1  # rate of natural increase
K = 100  # carrying capacity
N0 = 20  # initial population density (changed to 20 individuals)

# Time span
t_span = (0, 50)  # time interval for simulation

# Solve the differential equation
sol = solve_ivp(logistic_growth, t_span, [N0], args=(r, K), t_eval=np.linspace(0, 50, 1000))

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(sol.t, sol.y[0], label='Population density (N)')
plt.xlabel('Time')
plt.ylabel('Population Density')
plt.title('Logistic Growth Model (Initial Population: 20)')
plt.legend()
plt.grid(True)
plt.show()

#converting the for loop into a function:
#first code
def logistic_growth(K, R, F):
    population = [20]  # Initial population starts at 20 individuals
    for step in range(1, F + 1):
        new_population = population[-1] + R * population[-1] * (1 - population[-1] / K)
        population.append(min(K, new_population))  # Cap population at K
    return population
    
#new function for population growth
def simulate_logistic_growth(max_individuals, growth_rate, num_generations):
    result = logistic_growth(max_individuals, growth_rate, num_generations)
    return result

# Example usage:
max_individuals = 1000  # K: Maximum number of individuals
growth_rate = 0.2  # R: Growth rate
num_generations = 20  # F: Number of steps of generations

result = simulate_logistic_growth(max_individuals, growth_rate, num_generations)
print(result)
#[20, 23.92, 28.58956672, 34.14400739895253, 40.73964623049109, 48.55563172159219, 57.79522819153405, 68.68621614949856, 81.47990012161115, 96.44808532116784, 113.87725575297756, 134.05910102800726, 157.27655271992123, 183.78468045681305, 213.7862547940531, 247.40259320509017, 284.6415032211875, 325.36564679422156, 369.26621533030146, 415.8479508394888, 464.4316373639062]

