# This is a python script to run model by JR on Vertical transmission
# The Python code consists of 3 stages of vertical transmission: gene pool, proliferation and Holobiont selection


#stage of proliferation:

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

####################################################
#Step 1 creating the Matrix and trial plots
#trial to create matrix using given indexes from JRs paper
import numpy as np

def simulate_population(Tm, K):
    # Initialize an empty matrix H with dimensions (Tm + 1) x (K + 1)
    H = np.zeros((Tm + 1, K + 1), dtype=int) #np.zeros gives a shell of a matrix with zeros

    # Loop through each time step
    for t in range(Tm + 1):
        # Generate the count of bacteria for each individual (0 to K)
        individuals_bacteria_count = np.random.randint(0, K + 1, size=100)  # Example: 100 individuals

        # Count the number of individuals with each count of bacteria and update H matrix
        unique, counts = np.unique(individuals_bacteria_count, return_counts=True)
        H[t, unique] = counts

    return H

# Set values for Tm and K
Tm_value = 10  # Example value for Tm
K_value = 20  # Example value for K

# Simulate population and create matrix H
resulting_matrix = simulate_population(Tm_value, K_value)
print(resulting_matrix)

import numpy as np
import matplotlib.pyplot as plt

# Assuming H is your matrix with dimensions (Tm + 1) x (K + 1)
# You have H matrix filled with counts of individuals having a certain number of bacteria

# Example matrix H with random data (replace this with your actual data)
Tm = 10  # Example value of Tm
K = 20  # Example value of K

# Generating random data for demonstration (replace this with your actual data)
H = np.random.randint(0, 20, size=(Tm + 1, K + 1))  # Random data between 0 and 20

# Plotting histograms for different time points
for t in range(1, Tm + 1):
    plt.figure(figsize=(8, 6))
    plt.hist(range(K + 1), bins=np.arange(K + 2) - 0.5, weights=H[t], edgecolor='black')
    plt.title(f'Histogram at time t={t}')
    plt.xlabel('Number of Bacteria')
    plt.ylabel('Count of Individuals')
    plt.xticks(range(K + 1))
    plt.grid(axis='y', alpha=0.5)
    plt.show()
#######################################################

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

###################################################
#trial example of applying exponential growth to H matrix
#trial 1 - resulted in fractions
#import numpy as np

# Example values
#max_individuals = 1000  # K: Maximum number of individuals
#growth_rate = 0.5  # R: Growth rate
#num_generations = 20  # F: Number of steps of generations

# Generate initial data for the H matrix
#Tm = num_generations
#K = max_individuals

# Initializing H matrix with zeros
#H = np.zeros((Tm + 1, K + 1))

# Assuming you have populated H matrix with initial counts of individuals

# Apply exponential growth to the H matrix
#for t in range(1, Tm + 1):
   # for i in range(K + 1):
        # Apply exponential growth to each cell in the matrix
      #  H[t][i] = min(max_individuals, H[t - 1][i] * (1 + growth_rate))

# Now, the H matrix contains updated counts after exponential growth
#print(H)

import numpy as np

Tm = 20  # Number of time steps
K = 1000  # Maximum number of bacteria
H = np.zeros((Tm + 1, K + 1))  # Initialize matrix with zeros

# Your code to populate the matrix H here...

# Apply exponential growth factor
max_individuals = 1000  # K: Maximum number of individuals
growth_rate = 2  # R: Growth rate

for t in range(1, Tm + 1):
    for i in range(K + 1):
        H[t][i] = np.ceil(H[t - 1][i] * (1 + growth_rate * (1 - H[t - 1][i] / max_individuals)))

print(H)
##results show a fraction even with np.floor used to round values to whole numbers
#################################################
