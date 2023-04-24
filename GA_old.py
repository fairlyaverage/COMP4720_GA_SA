import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Define GA parameters
population_size = 50
mutation_rate = 0.02
generations = 1000
num_iterations = 1000


# Define fitness function
def fitness(individual, cities):
    distance = 0
    for i in range(len(individual) - 1):
        city1 = individual[i]
        city2 = individual[i + 1]
        distance += np.linalg.norm(cities[city1] - cities[city2])
    return 1 / distance


# Define GA functions
def create_individual(num_cities):
    return random.sample(range(num_cities), num_cities)


def create_population(num_cities):
    return [create_individual(num_cities) for i in range(population_size)]


def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual


def crossover(parent1, parent2):
    child = [-1] * len(parent1)
    start = random.randint(0, len(parent1) - 1)
    end = random.randint(0, len(parent1) - 1)
    if start > end:
        start, end = end, start
    for i in range(start, end + 1):
        child[i] = parent1[i]
    j = 0
    for i in range(len(parent2)):
        if not parent2[i] in child:
            while child[j] != -1:
                j += 1
            child[j] = parent2[i]
    return child


def select_parents(population, cities):
    fitnesses = [fitness(individual, cities) for individual in population]
    sum_fitnesses = sum(fitnesses)
    probabilities = [fitness / sum_fitnesses for fitness in fitnesses]
    return random.choices(population, weights=probabilities, k=2)


def evolve(population, cities):
    new_population = []
    for i in range(population_size):
        parent1, parent2 = select_parents(population, cities)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.append(child)
    return new_population


# Main loop
best_fitnesses = []
best_individuals = []
average_fitness = 0
st = time.time()
for i in range(num_iterations):
    # Generate random cities
    N = random.randint(5, 20)
    cities = np.multiply(np.random.rand(N, 2), 300)

    # Run GA
    population = create_population(N)
    best_fitness = 0
    for j in range(generations):
        population = evolve(population, cities)
        best_individual = max(population, key=lambda individual: fitness(individual, cities))
        current_fitness = fitness(best_individual, cities)
        if current_fitness > best_fitness:
            best_fitness = current_fitness

        # Store best individual and fitness for this iteration
    best_individuals.append(best_individual)
    best_fitnesses.append(best_fitness)

    # Calculate and store average fitness for this iteration
    fitnesses = [fitness(individual, cities) for individual in population]
    average_fitness = sum(fitnesses) / len(fitnesses)

    # Print progress
    print(
        f"Iteration {i + 1}/{num_iterations} - Best fitness: {best_fitness:.2f} - Average fitness: {average_fitness:.2f}")

et = time.time()
print(f"\nElapsed time: {et - st:.2f}s")

