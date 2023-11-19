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

