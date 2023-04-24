import numpy as np
import matplotlib.pyplot as plt
import random
import time
from progress import *

# Define GA parameters
# set global vars
def set_default_global_vars():
    number_of_airports = 3
    number_of_cities = 1000
    number_of_generations = 1000 # this is actually iterations?
    mutation_rate = 0.001
    population_size = 50
    number_of_iterations = 1000 # generations
    number_of_trials = 1

    set_airports(number_of_airports)
    set_cities(number_of_cities)
    set_generations(number_of_generations)
    set_mutation_rate(mutation_rate)
    set_population_size(population_size)
    set_iterations(number_of_iterations)
    set_trials(number_of_trials)
    return True

def set_trials(n):
    global number_of_trials
    number_of_trials = n
    return number_of_trials

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
    set_iterations(n)
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

# write useful stats to file every run
def save_stats():
    return True

# population_size = 100
# mutation_rate = 0.1
# number_of_generations = 1000 # per run
# number_of_iterations = 1000 # number of runs?

# pass log
# def plot_ga_convergence(population, number_of_iterations, cities):
#     fitness_values = np.zeros(number_of_iterations)
#     for i in range(number_of_iterations):
#         fitness_values[i] = max([fitness_prime(individual, cities) for individual in population])
#     return fitness_values

def plot_ga_convergence():
    plt.figure(figsize=(10, 10))
    for i in range(fitness_values_all_runs.shape[0]): # for each run; range should be arr.shape[0] since 1st dimension is # runs, 2nd dimension is # iterations
        plt.plot(fitness_values_all_runs[i], alpha=0.5)
    plt.title(f'All {number_of_trials} Convergence Plots for GA')
    plt.xlabel('Generations')
    plt.ylabel('Best Individual Fitness per Generation = (sum of straight-line distance)^2')
    plt.show()
    return

# Define accuracy function
def accuracy(individual, cities, number_of_cities, number_of_airports):
    distances = np.zeros(len(cities))
    for i in range(len(cities)):
        distances[i] = np.min(np.linalg.norm(cities[i] - individual, axis=1))
    return np.sum(distances) / (number_of_cities * number_of_airports)

# Define fitness function - old, using fitness_prime() now
# def fitness(individual, cities):
#     distances = np.zeros(len(cities))
#     for i in range(len(cities)):
#         distances[i] = np.min(np.linalg.norm(cities[i] - individual, axis=1)) ** 2
#     return 1 / (np.sum(distances) + 1e-6)

def fitness_prime(individual, cities):
    distances = np.zeros((len(cities), len(individual))) # 2d array for storing locations, return minimum for each city (this is the dist to the closest airport)
    for i in range(len(cities)): # each city
        city_x, city_y = cities[i]
        for j in range(len(individual)): # each airport
            airport_x, airport_y = individual[j]
            distances[i][j] = pow(city_x - airport_x, 2) + pow(city_y - airport_y, 2) # linear distance (x_c - x_a)^2 + (y_c - y_a)^2
    # fitness score: higher is better but this wants to minimize distances
    # input(f"in fitness_prime, distances = {distances}, \n distances.min(axis=1) = {distances.min(axis=1)} which should sum to ={np.sum(distances.min(axis=1))} and invert to {1/np.sum(distances.min(axis=1))}")
    return 1 / np.sum(distances.min(axis=1)) # returns an individual [solution's] fitness value

# Define GA functions
def create_individual(number_of_cities, number_of_airports):
    return np.random.rand(number_of_airports, 2) * 100

def create_population(number_of_cities, number_of_airports):
    return [create_individual(number_of_cities, number_of_airports) for i in range(population_size)]

def mutate(individual):
    mutated = False
    if random.random() < mutation_rate:
        # Move airport to a new location
        i = random.randint(0, len(individual) - 1)
        individual[i] = np.random.rand(1, 2) * 100
        mutated = True
    return individual, mutated

# def crossover(parent1, parent2):
#     # len >= 3
#     # acceptable range 1 to len() - 1

#     child = np.zeros_like(parent1)
#     for i in range(len(parent1)):
#         if random.random() < 0.5:
#             child[i] = parent1[i]
#         else:
#             child[i] = parent2[i]
#     return child

def crossover(mother, father):
    # input(f"mom = {mother}\ndad = {father}")
    # 1 < index < len() - 1 and always len() >= 3
    crossover_index = random.randint(1,len(mother) - 1) # mom and dad are the same length
    # slice and concatenate
    brother = np.concatenate((mother[:crossover_index], father[crossover_index:])) # LHS from mom RHS from dad
    sister = np.concatenate((father[:crossover_index], mother[crossover_index:])) # LHS from dad RHS from mom
    # input(f"brother={brother}\nsister={sister}")
    return brother, sister # children

def select_parents(population, cities):
    fitnesses = [fitness_prime(individual, cities) for individual in population]
    sum_fitnesses = sum(fitnesses)
    if sum_fitnesses == 0:
        return random.sample(population, k=2)
    probabilities = [fitness / sum_fitnesses for fitness in fitnesses]
    parents = []
    for i in range(population_size):
        parent1, parent2 = random.choices(population, weights=probabilities, k=2)
        parents.append(parent1)
        parents.append(parent2)
    return parents
    # return parent1, parent2

# def get_all_parents(population, cities):
#     parents = []
#     for i in range(len(population)):
#         parent1, parent2 = select_parents(population, cities)
#         parents.append(parent1)
#         parents.append(parent2)
#     return parents

def save_cities(cities):
    # np.array(number_of_cities, 2) * 100
    coordinate_file = open('coord.txt', 'w')
    # for coord in locations:
    coordinate_file.write(cities)
    coordinate_file.close()

    # coordinate_file = open('coord.txt', 'w')
    # for coord in locations:
    #     coordinate_file.write(f"{coord[0]} {coord[1]}\n")
    # coordinate_file.close()
    return True

def generate_cities(number_of_cities):
    return np.random.rand(number_of_cities, 2) * 100 # gen 2-d array for city locations?; update to read this from file or pass between GA and SA


def run_experiment(number_of_trials, number_of_cities, number_of_airports):
    log = []
    accuracies = []
    times = []  # Store the time taken for each iteration
    # cities = np.random.rand(number_of_cities, 2) * 100 # gen 2-d array for city locations?; update to read this from file or pass between GA and SA
    cities = generate_cities(number_of_cities)


    fitness_values_all_runs = np.zeros((number_of_trials, number_of_iterations)) # what is this? 'run 1000 trials', 'run 1000 generations' maybe?

    for i in range(number_of_trials): # run experiment this many times
        print(f"Run {i + 1} of {number_of_trials}") # replace progress bar here
        start_time = time.time()  # Start the timer
        best_solution, fitness_values, generation_times_list, best_solution_fitness = find_airports(number_of_cities, number_of_airports, cities) # most of the action here # sanity check: best_solution_fitness == fitness_values.max()
        end_time = time.time()  # End the timer
        # record fitness value(s?) for this generation

        #sanity:


        fitness_values_all_runs[i] = fitness_values # [ [ g_1 g_2 g_3 g_4 ... g_n ], ]
        # input(f"fitness_all_runs={fitness_values_all_runs}")
        elapsed_time = (end_time - start_time) # / (number_of_cities * number_of_airports)  # Calculate the time per city and airport
        times.append(elapsed_time)  # Add the elapsed time to the list of times

        acc = accuracy(best_solution, cities, number_of_cities, number_of_airports)
        accuracies.append(acc)
        print(f"Accuracy: {acc:.2f}")

        sum = 0
        for each in times:
            sum+=each
        average = sum/len(times)
        # progress.progress(i, number_of_trials)
        print(f"Run {i} took {elapsed_time:.2f} seconds, reached accuracy {acc:.10f}, the best individual was {best_solution}. The total time is {sum} with an average run time of {average}")

    # for each run want:
    run_statistics = {
        "accuracy": acc,
        "times": generation_times_list, # 2-d list of [each trial][each generation's runtime]

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

    return accuracies, times, fitness_values_all_runs, run_statistics

def find_airports(number_of_cities, number_of_airports, cities): # called once per trial
    population = create_population(number_of_cities, number_of_airports)
    # print(population)
    best_individual = None
    best_fitness = None
    fitness_values = np.zeros(number_of_iterations)
    times = []
    best_fitness_score_in_all_generations = 0
    best_solution_in_all_generations = None
    for iteration in range(number_of_iterations): # "number_of_generations"
        progress(iteration, number_of_iterations)
        # timer
        start = time.time()
        parents = select_parents(population, cities)
        # this only selects two parents but needs to happen for the entire population, twice essentially
        # parents = select_parents(population, cities) # is this trying to change cities? it shouldn't, those are static locations, should only need population for this?
        # input(f"parents={parents}, range(len(parents)) {range(len(parents))}")
        children = []
        for i in range(0,len(parents) - 1,2):
            # print(f"get p[{i}], p[{i+1}]")
            brother, sister=crossover(parents[i], parents[i+1]) # pass mom and dad, get two siblings back
            # twins, it's always twins
            children.append(brother)
            children.append(sister)
            # children.append(crossover(parents[i], parents[i+1]))
        # input(f"children={children}")
        for child in children:
            child, tf = mutate(child) # tf for debug
            # if tf: print(f"child {child} mutated={tf}")
        # print(f"pop_size={len(population)}, children_size={len(children)}")
        # input(f"{children}")
        population = children
        # child = crossover(parents[0], parents[1])
        # input(f"child={child}")
        # child = mutate(child)
        # child = mutate(child, number_of_cities) # number_of_cities? every

        # input(f"{population}")
        # all population fitnesses


        # population_fitnesses = [fitness_prime(individual, cities) for individual in population] # cities is a 2d array [ [ x, y ],..., [x_N, y_N] ] of city coordinates; list of every population member's fitness
        # input(f"fitnesses = {fitnesses}")

        best_fitness_score = 0 # in generation
        best_solution = None # in generation
        # in this generation
        for i in range(len(population)):
            fitness_score = fitness_prime(population[i], cities) # should store each individual's fitness score in an indexed array
            # population_fitnesses[i] = score
            if fitness_score > best_fitness_score:
                best_fitness_score = fitness_score # float
                best_solution = population[i] # i int, need the actual individual (set of coords)
                # print(f"found a new best individual score {best_fitness_score} from individual {population[i]}")
        # input(f"in generation {iteration} the best fitness score = {best_fitness_score} belongs to {best_solution}")

        # all generations so far this run
        if best_fitness_score > best_fitness_score_in_all_generations:
            best_fitness_score_in_all_generations = best_fitness_score
            best_solution_in_all_generations = best_solution

        # then take this entire array, maybe save it, but definitely get the best fitness out of it
        # best = fitness_values.max() # best score
        # still want to save the actual individual that did this


        # print("for individual in population = [")
        # all = []
        # for individual in population:fitnesses
        #     all.append(fitness_prime(individual, cities))
        #     # print(f"{fitness_prime(individual, cities)}, ")
        # print(f"{max(all)}]")
        # input(f"best_fitness_score={best_fitness_score} (is this the same value for every generation?)")
        fitness_values[iteration] = best_fitness_score # set best score this generation to fitness_values[generation]

        # input(f"fitness_values = {fitness_values}")
        # print("fitnesses", fitnesses, "len=()", len(fitnesses))
        # print(f"\n\nfitness_values", fitness_values, "len()=", len(fitness_values))

        # this sets the worst index to child?
        # worst_index = np.argmin(fitnesses)
        # population[worst_index] = child

        # for individual in population:
        #     # redundant call to fitness_prime()? fitnesses already has every member's fitness and nothing's changed since it was last set
        #     individual_fitness = fitness_prime(individual, cities)
        #     if best_fitness is None or individual_fitness > best_fitness:
        #         best_individual = individual
        #         best_fitness = individual_fitness


        # improve
        # set np array = population size
        # store fitnesses there (should contain k population fitnesses)
        # min(fitnesses) = current_best
        # index_of_best = fitness.index(min(fitnesses)) could return duplicate solution, but it doesn't really matter it'll still be a tie for best
        # if current_best > all_time_best: all_time_best_coordinates = current_best_coordinates to store a new better individual solution


        # fitness_values[iteration] = best_fitness


        # end generation run time
        end = time.time()
        times.append(end - start) # add this generation's delta time to times; len(times) == number_of_generations on return
        # debug
        # print(f"{iteration}th generation took {(end - start):.2f} seconds, fittest individual is {(best_fitness_score*10000):.5f} (dist: {(1/best_fitness_score):.5f}) current best fitness is {(best_fitness_score_in_all_generations*100000):.5f} which is a distance delta of {(1/best_fitness_score_in_all_generations):.5f}")
    # return best_individual, fitness_values, times
    return best_solution_in_all_generations, fitness_values, times, best_fitness_score_in_all_generations # need to find min fitness for plotting

# Run the experiment
# # runs, # cities, # airports
# number_of_trials = 5
# number_of_cities = 1000
# number_of_airports = 3
# set_default_global_vars() or manually call setters first
set_default_global_vars()
if (input("change defaults y") == "y"):
    N = input(f"cities")
    n = input(f"airports")
    g = input(f"generations")
    t = input(f"trials")
    m = input(f"mutation rate")
    k = input(f"population size")

    if N == '': N = number_of_cities
    if n == '': n = number_of_airports
    if g == '': g = number_of_generations
    if t == '': t = number_of_trials
    if m == '': m = mutation_rate
    if k == '': k = population_size

    set_cities(int(N))
    set_airports(int(n))
    set_generations(int(g))
    set_trials(int(t))
    set_mutation_rate(float(m))
    set_population_size(int(k))

accuracies, times, fitness_values_all_runs, run_statistics = run_experiment(number_of_trials, number_of_cities, number_of_airports)

# Calculate mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

# Calculate mean and standard deviation of times
mean_time = np.mean(times)
std_time = np.std(times)

# debug
# print(f"fitness values all runs: {fitness_values_all_runs} f.shape() = {fitness_values_all_runs.shape()}; f.size() = {fitness_values_all_runs.size()}")
# plt.figure(figsize=(10, 10))
# for i in range(fitness_values_all_runs.shape[0]): # range should be arr.shape[0] since 1st dimension is # runs, 2nd dimension is # iterations
#     plt.plot(fitness_values_all_runs[i], alpha=0.1)
# plt.title('All 1000 Convergence Plots for GA')
# plt.xlabel('Number of iterations')
# plt.ylabel('Fitness')
# plt.show()

plot_ga_convergence()
