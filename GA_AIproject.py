import time
import numpy as np
import matplotlib.pyplot as plt

# Load city coordinates from file
city_coords = np.loadtxt('city_coords.txt')

# Define GA parameters
pop_size = 100  # Population size
n_genes = 2  # Number of genes in each individual (airport coordinates)
n_gen = 1000  # Number of generations
mut_rate = 0.1  # Mutation rate
sel_rate = 0.5  # Selection rate
tourn_size = 5  # Tournament size
n_airports = 3  # Number of airports to place

# Define fitness function
def fitness(individual):
    # Compute distance to nearest airport for each city
    dists = np.min(np.sqrt(np.sum((city_coords - individual)**2, axis=1)))
    # Return sum of squared distances
    return np.sum(dists**2)

# Define mutation function
def mutate(individual):
    # Randomly add or subtract a small value from each gene
    return individual + np.random.normal(0, 0.1, size=n_genes)

# Define selection function
def select(population, fitnesses):
    # Tournament selection
    idxs = np.random.choice(len(population), size=tourn_size, replace=False)
    return population[idxs[np.argmin(fitnesses[idxs])]]

# Initialize population
population = np.random.uniform(size=(pop_size, n_genes))

# Start timer
start_time = time.time()

# Evolution loop
for i in range(n_gen):
    # Compute fitness of each individual
    fitnesses = np.array([fitness(individual) for individual in population])
    # Sort population by fitness
    sorted_idxs = np.argsort(fitnesses)
    population = population[sorted_idxs]
    fitnesses = fitnesses[sorted_idxs]
    # Select parents and create offspring
    offspring = []
    for j in range(pop_size):
        parent1 = select(population, fitnesses)
        parent2 = select(population, fitnesses)
        child = parent1 + sel_rate * (parent2 - parent1)
        # Mutate child with a certain probability
        if np.random.rand() < mut_rate:
            child = mutate(child)
        offspring.append(child)
    # Replace population with offspring
    population = np.array(offspring)

# Select best individuals as airports
airports = population[np.argsort(fitnesses)[:n_airports]]

# Stop timer and calculate elapsed time
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")

# Plot convergence
plt.figure()
plt.plot(fitnesses)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Convergence plot')
plt.show()

# Plot results
plt.figure()
plt.scatter(city_coords[:, 0], city_coords[:, 1], marker='.', color='gray')
plt.scatter(airports[:, 0], airports[:, 1], marker='*', color='red')
plt.show()

"""
/*

This code includes a timer that measures the elapsed time of the GA algorithm, and a plot of the convergence of the fitness function over generations. The elapsed time is printed to the console after the algorithm has completed.

Note that this code assumes that the city coordinates are stored in a text file called city_coords.txt, with one row per city and two columns for the x and y coordinates. You may need to modify this code to match your specific problem and data format.
*/



/* define mathematically:

The optimization problem of placing n new airports in Romania such that the sum of squared distances from each city on the map to its nearest airport is minimized can be mathematically defined as follows:

Decision variables:

Let X be a matrix of size n x 2, where each row represents the coordinates of an airport to be placed. Specifically, X<sub>i,1</sub> and X<sub>i,2</sub> denote the x and y coordinates of the i-th airport, respectively, where i = 1, 2, ..., n.
Objective function:

Minimize the sum of squared distances from each city on the map to its nearest airport, which can be computed as:
php
Copy code
f(X) = ∑<sub>j=1</sub><sup>m</sup> (min<sub>i=1,2,...,n</sub> √((X<sub>i,1</sub> - x<sub>j</sub>)<sup>2</sup> + (X<sub>i,2</sub> - y<sub>j</sub>)<sup>2</sup>))<sup>2</sup>
where m is the total number of cities on the map, and (x<sub>j</sub>, y<sub>j</sub>) are the coordinates of the j-th city.

Constraints:

None, since the problem does not have any explicit constraints.
The genetic algorithm can then be used to solve this optimization problem by defining appropriate selection, crossover, and mutation operators to create and evolve a population of potential solutions (i.e., matrices of airport coordinates X). The algorithm can be terminated when a satisfactory solution is found, or after a predetermined number of iterations or generations.

*/
"""
