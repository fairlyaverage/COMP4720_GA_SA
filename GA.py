import numpy as np
import matplotlib.pyplot as plt
import random
import time


start_time_total = time.time()  # Record the start time of the whole script


# Define GA parameters
population_size = 50
mutation_rate = 0.02
generations = 1000
num_iterations = 1000


# def plot_ga_convergence(fitness_values_all_runs, num_iterations):
#     plt.figure(figsize=(10, 5))
#     for i in range(len(fitness_values_all_runs)):
#         plt.plot(fitness_values_all_runs[i], alpha=0.1)
#     plt.title('All Convergence Plots for GA')
#     plt.xlabel('Number of iterations')
#     plt.ylabel('Fitness')
#     plt.show()
def plot_ga_convergence_avg(fitness_values_all_runs, num_iterations):
    avg_fitness_values = np.mean(fitness_values_all_runs, axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(avg_fitness_values)
    plt.title('Average Convergence Plot for GA')
    plt.xlabel('Number of iterations')
    plt.ylabel('Fitness')
    plt.show()

# Define accuracy function
def accuracy(individual, cities, num_cities, num_airports):
    distances = np.zeros(len(cities))
    for i in range(len(cities)):
        distances[i] = np.min(np.linalg.norm(cities[i] - individual, axis=1))
    return np.sum(distances) / (num_cities * num_airports)


# Define fitness function
def fitness(individual, cities):
    distances = np.zeros(len(cities))
    for i in range(len(cities)):
        distances[i] = np.min(np.linalg.norm(cities[i] - individual, axis=1)) ** 2
    return 1 / (np.sum(distances) + 1e-6)


# Define GA functions
def create_individual(num_cities, num_airports):
    return np.random.rand(num_airports, 2) * 100


def create_population(num_cities, num_airports):
    return [create_individual(num_cities, num_airports) for i in range(population_size)]


def mutate(individual, num_cities):
    if random.random() < mutation_rate:
        # Move airport to a new location
        i = random.randint(0, len(individual) - 1)
        individual[i] = np.random.rand(1, 2) * 100
    return individual


def crossover(parent1, parent2):
    child = np.zeros_like(parent1)
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    return child


def select_parents(population, cities):
    fitnesses = [fitness(individual, cities) for individual in population]
    sum_fitnesses = sum(fitnesses)
    if sum_fitnesses == 0:
        return random.sample(population, k=2)
    probabilities = [fitness / sum_fitnesses for fitness in fitnesses]
    parent1, parent2 = random.choices(population, weights=probabilities, k=2)
    return parent1, parent2


def plot_all_iterations_ga(best_individuals, num_runs, cities):
    plt.figure(figsize=(10, 10))

    all_city_coords = cities
    all_airports = np.vstack(best_individuals)

    plt.scatter(all_city_coords[:, 0], all_city_coords[:, 1], marker='.', color='gray', alpha=0.1)
    plt.scatter(all_airports[:, 0], all_airports[:, 1], marker='*', color='red', alpha=0.1)

    plt.title(f'All Iterations: Cities and Airport Locations (GA)')
    plt.show()


def run_experiment(num_runs, num_cities, num_airports):
    accuracies = []
    times = []  # Store the time taken for each iteration
    cities = np.random.rand(num_cities, 2) * 100
    fitness_values_all_runs = np.zeros((num_runs, num_iterations))
    best_individuals = []

    for i in range(num_runs):
        print(f"Run {i + 1} of {num_runs}")
        start_time = time.time()  # Start the timer
        best_individual, fitness_values = find_airports(num_cities, num_airports, cities)
        end_time = time.time()  # End the timer
        fitness_values_all_runs[i] = fitness_values
        elapsed_time = (end_time - start_time) / (num_cities * num_airports)  # Calculate the time per city and airport
        times.append(elapsed_time)  # Add the elapsed time to the list of times

        acc = accuracy(best_individual, cities, num_cities, num_airports)
        accuracies.append(acc)
        best_individuals.append(best_individual)
        print(f"Accuracy: {acc:.2f}")

    return accuracies, times, fitness_values_all_runs, best_individuals, cities


def find_airports(num_cities, num_airports, cities):
    population = create_population(num_cities, num_airports)
    best_individual = None
    best_fitness = None
    fitness_values = np.zeros(num_iterations)

    for iteration in range(num_iterations):
        parents = select_parents(population, cities)
        child = crossover(parents[0], parents[1])
        child = mutate(child, num_cities)

        fitnesses = [fitness(individual, cities) for individual in population]
        worst_index = np.argmin(fitnesses)
        population[worst_index] = child

        for individual in population:
            individual_fitness = fitness(individual, cities)
            if best_fitness is None or individual_fitness > best_fitness:
                best_individual = individual
                best_fitness = individual_fitness

        fitness_values[iteration] = best_fitness

    return best_individual, fitness_values


# Run the experiment 1000 times
# can modify N and n here --> run_experiment(iterations, N, n)
# for example if you wanted to run 5000 times you could change it to N=5000 and n=5
accuracies, times, fitness_values_all_runs, best_individuals, cities = run_experiment(1000, 10, 2)

# # Calculate mean and standard deviation of accuracies
# mean_accuracy = np.mean(accuracies)
# std_accuracy = np.std(accuracies)
#
# # Calculate mean and standard deviation of times
# mean_time = np.mean(times)
# std_time = np.std(times)
#
# plot_ga_convergence(fitness_values_all_runs, num_iterations)
# plot_all_iterations_ga(best_individuals, num_iterations, cities)
# Calculate mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

# Calculate mean and standard deviation of times
mean_time = np.mean(times)
std_time = np.std(times)

plot_ga_convergence_avg(fitness_values_all_runs, num_iterations)
plot_all_iterations_ga(best_individuals, num_iterations, cities)

end_time_total = time.time()  # Record the end time of the whole script
total_elapsed_time = end_time_total - start_time_total  # Calculate the total elapsed time
print(f"Total elapsed time: {total_elapsed_time:.2f} seconds")

print("Mean Accuracy: {:.4f}".format(mean_accuracy))
print("Standard Deviation of Accuracies: {:.4f}".format(std_accuracy))
print("Mean Time: {:.4f}".format(mean_time))
print("Standard Deviation of Times: {:.4f}".format(std_time))

# plt.figure(figsize=(10, 10))
# for i in range(1000):
#     plt.plot(fitness_values_all_runs[i], alpha=0.1)
# plt.title('All 1000 Convergence Plots for GA')
# plt.xlabel('Number of iterations')
# plt.ylabel('Fitness')
# plt.show()
