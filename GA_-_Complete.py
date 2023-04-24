import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Define GA parameters
population_size = 100
mutation_rate = 0.1
generations = 1000 # per run
num_iterations = 1000 # number of runs?

def plot_ga_convergence(population, num_iterations, cities):
    fitness_values = np.zeros(num_iterations)
    for i in range(num_iterations):
        fitness_values[i] = max([fitness_prime(individual, cities) for individual in population])
    return fitness_values


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

def fitness_prime(individual, cities):
    '''
        for each airport
            for each city
                calc distance
                if distance airport-city distance < best_distance_from_this_city_to_another_airport: update best_distance

        sum best_distances
        return 1/sum of best distances (want to minimize this score so ^-1)
    '''
    # num_airports might be the same as len(individual) try len(individual) - should output the length of the first dimension, or individual.shape[0] for sure works
    # print(np.zeros(len(cities),2))
    distances = np.zeros((len(cities), len(individual))) # 2d array for storing locations, return minimum for each city (this is the dist to the closest airport)
    # distances has each city each airport, use the min[city] to get best airport distance

    # distance_sum = 0
    for i in range(len(cities)): # each city
        city_x, city_y = cities[i]
        for j in range(len(individual)): # each airport
            airport_x, airport_y = individual[j]
            distances[i][j] = pow(city_x - airport_x, 2) + pow(city_y - airport_y, 2) # linear distance (x_c - x_a)^2 + (y_c - y_a)^2
        # debug
        # print(distances[i], f"after {i}th city")
    # for distance in distances: distance_sum += distance.min() # only care about distance to closest airport for each city
    # or
    return 1 / np.sum(distances.min(axis=1)) # possibly faster?


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
    fitnesses = [fitness_prime(individual, cities) for individual in population]
    sum_fitnesses = sum(fitnesses)
    if sum_fitnesses == 0:
        return random.sample(population, k=2)
    probabilities = [fitness / sum_fitnesses for fitness in fitnesses]
    parent1, parent2 = random.choices(population, weights=probabilities, k=2)
    return parent1, parent2


def run_experiment(num_runs, num_cities, num_airports):
    log = []
    accuracies = []
    times = []  # Store the time taken for each iteration
    cities = np.random.rand(num_cities, 2) * 100 # gen 2-d array for city locations?; update to read this from file or pass between GA and SA
    fitness_values_all_runs = np.zeros((num_runs, num_iterations)) # what is this? 'run 1000 trials', 'run 1000 generations' maybe?

    for i in range(num_runs): # run experiment this many times
        print(f"Run {i + 1} of {num_runs}")
        start_time = time.time()  # Start the timer
        best_individual, fitness_values = find_airports(num_cities, num_airports, cities)
        end_time = time.time()  # End the timer
        fitness_values_all_runs[i] = fitness_values
        elapsed_time = (end_time - start_time) # / (num_cities * num_airports)  # Calculate the time per city and airport
        times.append(elapsed_time)  # Add the elapsed time to the list of times

        acc = accuracy(best_individual, cities, num_cities, num_airports)
        accuracies.append(acc)
        print(f"Accuracy: {acc:.2f}")

        sum = 0
        for each in times:
            sum+=each
        average = sum/len(times)

        print(f"Run {i} took {elapsed_time:.2f} seconds, reached acuracy {acc:.10f}, the best individual was {best_individual}. The total time is {sum} with an average run time of {average}")

    # for each run want:
    run_statistics = {
        accuracy: acc,
        times:
    }
        # { accuracy: float, times: [], average_time: float, total_time: float, best_individual: [ n (x,y) tuples ], best_individual_each_run: { number_of_generations [ n (x,y) tuples ] } }
        # accuracy = -1 # solution (best individual) accuracy() = (Sum of the distance between each of the N cities and its nearest airport) / (N × n) = fitness^-1 / (N*n) because fitness is 1/(sum of distances)
            # derive
            # mean accuracy over all 1000 trials
            # standard deviation of accuracy over 1000 trials
        # times = [] # per generation
            # derive
            # average_time = sum / len(times)
            # total_time = sum
        # best_individuals_per_run = {}
            # derive
            # best_individual = {}



    return accuracies, times, fitness_values_all_runs


def find_airports(num_cities, num_airports, cities):
    population = create_population(num_cities, num_airports)
    best_individual = None
    best_fitness = None
    fitness_values = np.zeros(num_iterations)
    times = []

    for iteration in range(num_iterations):
        start = time.time()
        parents = select_parents(population, cities) # is this trying to change cities? it shouldn't, those are static locations, should only need population for this?
        child = crossover(parents[0], parents[1])
        child = mutate(child, num_cities) # num_cities? every

        fitnesses = [fitness_prime(individual, cities) for individual in population] # cities is a 2d array [ [ x, y ],..., [x_N, y_N] ] of city coordinates
        worst_index = np.argmin(fitnesses)
        population[worst_index] = child

        for individual in population:
            individual_fitness = fitness_prime(individual, cities)
            if best_fitness is None or individual_fitness > best_fitness:
                best_individual = individual
                best_fitness = individual_fitness

        fitness_values[iteration] = best_fitness
        # debug
        print(f"{iteration}th generation took {(time.time() - s):.2f} seconds, fittest individual is {(min(fitnesses)*10000):.5f} (dist: {(1/min(fitnesses)):.5f}) current best fitness is {(best_fitness*100000):.5f} which is a distance delta of {(1/best_fitness):.5f}")

        # generation run time
        end = time.time()
        times.append(end - start)
    return best_individual, fitness_values, times


# Run the experiment
# # runs, # cities, # airports
num_runs = 5
N = 1000
n = 3
accuracies, times, fitness_values_all_runs = run_experiment(num_runs, N, n)

# Calculate mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

# Calculate mean and standard deviation of times
mean_time = np.mean(times)
std_time = np.std(times)

print(f"fitness values all runs: {fitness_values_all_runs}")
plt.figure(figsize=(10, 10))
for i in range(fitness_values_all_runs.shape[0]): # range should be arr.shape[0] since 1st dimension is # runs, 2nd dimension is # iterations
    plt.plot(fitness_values_all_runs[i], alpha=0.1)
plt.title('All 1000 Convergence Plots for GA')
plt.xlabel('Number of iterations')
plt.ylabel('Fitness')
plt.show()

# set global vars
def set_default_global_vars():
    number_of_airports = 3
    number_of_cities = 1000
    number_of_generations = 1000
    mutation_rate = 0.1
    population_size = 25
    number_of_iterations = 1

    set_airports(number_of_airports)
    set_cities(number_of_cities)
    set_generations(number_of_generations)
    set_mutation_rate(mutation_rate)
    set_population_size(population_size)
    set_iterations(number_of_iterations)
    return True

def set_airports(n):
    global number_of_airports
    number_of_airports = n
    return number_of_airports

def set_cities(n):
    global number_of_cities
    number_of_cities = n
    return number_of_cities

def set_generations(n):
    global number_of_generations
    number_of_generations = n
    return number_of_generations

def set_mutation_rate(n):
    global mutation_rate
    mutation_rate = n
    return mutation_rate

def set_population_size(n):
    global population_size
    population_size = n
    return population_size

# number of generations
def set_iterations(n):
    global number_of_iterations
    number_of_iterations = n
    return number_of_iterations
