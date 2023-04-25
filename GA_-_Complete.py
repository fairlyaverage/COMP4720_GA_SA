import numpy as np
import matplotlib.pyplot as plt
import random
import time
from progress import *
import json
import os

# set global vars
def set_default_global_vars():
    number_of_airports = 5
    number_of_cities = 500
    number_of_generations = 1000 # this is actually iterations?
    mutation_rate = 0.01 # .075 is pretty good
    population_size = 50 # even number please
    number_of_iterations = 500 # generations
    number_of_trials = 5

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
def save_data(trial_data, *args): # handle *args?
    """ save data to file
        try catch
        duplicate and then overwrite old one
        write to new file
        if success: rename new file to old file
        delete or overwrite new file for next save
        need to figure out how to append the object
    """
    trial_data['generation_population_fitness_scores'] = trial_data['generation_population_fitness_scores']
    PATH = "logs/"
    # if ./logs doesn't exist, create it

    unique_file_name = trial_data['type'] + str(trial_data['trial_number']) + ".json"
    # save run statistics
    try:
        if (os.path.exists(PATH+unique_file_name)): raise Exception(f"{unique_file_name} already exists")
        with open(PATH + unique_file_name, 'w+') as file:
            json.dump(trial_data, file)
    except:
        print("Something went wrong")
        return False
    finally:
        print(f"Successfully saved {unique_file_name}")
    return True

def save_cities(cities):
    coordinate_file = open('coord.txt', 'w')
    coordinate_file.write(cities)
    coordinate_file.close()
    return True

def plot_ga_convergence(graph_data):
    # get different values for convergence - best solution, population average, population total
    max = graph_data.max(axis=2)
    avg = graph_data.mean(axis=2)
    total = graph_data.sum(axis=2)

    _, ax = plt.subplots()
    # graph these separately
    for i in range(number_of_trials):
        ax.plot(np.arange(number_of_generations), max[i], label=f"Trial {i+1} - Best Solutions")
    for i in range(number_of_trials):
        ax.plot(np.arange(number_of_generations), avg[i], label=f"Trial {i+1} - Population Average")
    for i in range(number_of_trials):
        ax.plot(np.arange(number_of_generations), total[i], label=f"Trial {i+1} - Population Sum")

    ax.set_xlabel(f"{number_of_generations} Generations")
    ax.set_ylabel('Fitness Score')
    ax.set_title(f'Convergence Across {number_of_trials} Trials')
    ax.legend()
    plt.show()
    return

# Define accuracy function
def accuracy(individual, cities, number_of_cities, number_of_airports):
    distances = np.zeros(len(cities))
    for i in range(len(cities)):
        distances[i] = np.min(np.linalg.norm(cities[i] - individual, axis=1))
    return np.sum(distances) / (number_of_cities * number_of_airports)

def fitness_prime(individual, cities):
    distances = np.zeros((len(cities), len(individual))) # 2d array for storing locations, return minimum for each city (this is the dist to the closest airport)
    for i in range(len(cities)): # each city
        city_x, city_y = cities[i]
        for j in range(len(individual)): # each airport
            airport_x, airport_y = individual[j]
            distances[i][j] = pow(city_x - airport_x, 2) + pow(city_y - airport_y, 2) # linear distance (x_c - x_a)^2 + (y_c - y_a)^2
    # fitness score: higher is better but this wants to minimize distances
    return 1 / np.sum(distances.min(axis=1)) # returns an individual [solution's] fitness value

# Define GA functions
def create_individual(number_of_airports):
    return np.random.rand(number_of_airports, 2) * 100

def create_population(number_of_airports):
    return [create_individual(number_of_airports) for i in range(population_size)]

def mutate(individual):
    mutated = False
    if random.random() < mutation_rate:
        # Move airport to a new location
        i = random.randint(0, len(individual) - 1)
        individual[i] = np.random.rand(1, 2) * 100
        mutated = True
    return individual, mutated

def crossover(mother, father):
    mother, father = mother[0], father[0] # get the arrays out of the object
    # 1 < index < len() - 1 and always len() >= 3
    crossover_index = random.randint(1,len(mother) - 1) # mom and dad are the same length
    # slice and concatenate
    brother = np.concatenate((mother[:crossover_index], father[crossover_index:])) # LHS from mom RHS from dad
    sister = np.concatenate((father[:crossover_index], mother[crossover_index:])) # LHS from dad RHS from mom
    return brother, sister # children

def select_parents(population, cities):
    fitnesses = [fitness_prime(individual, cities) for individual in population]
    sum_fitnesses = sum(fitnesses)
    if sum_fitnesses == 0:
        return random.sample(population, k=2)
    probabilities = [fitness / sum_fitnesses for fitness in fitnesses]
    parents = []
    for i in range(population_size):
        parent = random.choices(population, weights=probabilities)
        parents.append(parent)
    return parents

def generate_cities(number_of_cities):
    return np.random.rand(number_of_cities, 2) * 100 # gen 2-d array for city locations?; update to read this from file or pass between GA and SA

def run_experiment(number_of_trials, number_of_cities, number_of_airports):
    data = {}
    accuracies = []
    times = []  # Store the time taken for each iteration
    # cities = np.random.rand(number_of_cities, 2) * 100 # gen 2-d array for city locations?; update to read this from file or pass between GA and SA
    cities = generate_cities(number_of_cities)

    fitness_values_all_runs = np.zeros((number_of_trials, number_of_iterations)) # for 1000 trials 1000 generations
    all_trials_all_generations_all_population_fitnesses = np.zeros((number_of_trials, number_of_iterations, population_size))
    for i in range(number_of_trials): # run experiment this many times
        progress(i, int(number_of_trials))
        print(f"Run {i + 1} of {number_of_trials}") # replace progress bar here
        start_time = time.time()  # Start the timer
        best_solution, fitness_values, generation_times_list, best_solution_fitness, generation_population_fitness_scores, runtime = find_airports(number_of_cities, number_of_airports, cities) # most of the action here # sanity check: best_solution_fitness == fitness_values.max()
        # use progress bar
        end_time = time.time()  # End the timer

        fitness_values_all_runs[i] = fitness_values # [ [ g_1 g_2 g_3 g_4 ... g_n ], ] # BEST INDIVIDUAL

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
        # print(f"Run {i} took {elapsed_time:.2f} seconds, reached accuracy {acc:.10f}, the best individual was {best_solution}. The total time is {sum} with an average run time of {average}")

        # save data
        # 2D arr at this point? [#generations][#individuals] but want to save each
        data[str(i)] = {
            'trial_number': i,
            'type': 'GA',
            'best_solution': best_solution.tolist(),
            'best_solution_score': best_solution_fitness,
            'runtimes': runtime, # can save progress list .copy()?
            # need to derive trial time, mean and standard deviation among all trials (not very useful unless it's normalized, since we're supposed to generate random numbers of cities and airports)
            'accuracy': None, # (Sum of the distance between each of the N cities and its nearest airport) / (N × n) but unclear if this is population accuracy or best accuracy, maybe get both?
            'number_of_cities': number_of_cities,
            'number_of_airports': number_of_airports,
            'number_of_iterations': number_of_iterations,
            # population_size is derivable
            'generation_population_fitness_scores': generation_population_fitness_scores.copy(),
            'cities': cities.tolist()
            # convergence plot data
            # global vars that didn't change over time?
            # populate what data you want
            # save it every trial so it can be accessed if this crashes
        }
    # for each run want:
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
        # print(data[str(i)])
        all_trials_all_generations_all_population_fitnesses[i] = generation_population_fitness_scores # i think this works
        save_data(data[str(i)])
    return accuracies, times, fitness_values_all_runs, data, all_trials_all_generations_all_population_fitnesses

def find_airports(number_of_cities, number_of_airports, cities): # called once per trial
    population = create_population(number_of_airports)
    fitness_values = np.zeros(number_of_iterations)
    generation_population_fitness_scores = np.zeros((number_of_iterations, population_size))
    # times = []
    best_fitness_score_in_all_generations = 0
    best_solution_in_all_generations = None
    # run each generation
    for iteration in range(number_of_iterations): # "number_of_generations"
        progress(iteration, number_of_iterations)
        # timer # start = time.time()

        # generational actions
        parents = select_parents(population, cities)
        children = []
        for i in range(0,len(parents),2):
            brother, sister=crossover(parents[i], parents[i+1]) # pass mom and dad, get two siblings back
            # twins, it's always twins
            children.append(brother)
            children.append(sister)
        for child in children:
            child, tf = mutate(child) # tf for debug # if tf: print(f"child {child} mutated={tf}")
        population = children # new population

        # generational stats
        best_fitness_score = 0 # in generation
        best_solution = None # in generation
        # in this generation
        for i in range(len(population)):
            fitness_score = fitness_prime(population[i], cities) # should store each individual's fitness score in an indexed array
            # input(f"{generation_population_fitness_scores.shape} ?= [{iteration}][{i}] <= [{number_of_iterations}][{population_size}]? len(pop) = {len(population)} pop={population}")
            generation_population_fitness_scores[iteration][i]=fitness_score # [iteration-th][i-th individual]
            # redundant - storing all population fitnesses, and can just use np.max() to get this value but it's probably best to save the actual solution instead of storing 999+ suboptimal solutions
            if fitness_score > best_fitness_score:
                best_fitness_score = fitness_score # float
                best_solution = population[i] # i int, need the actual individual (set of coords)
        # all generations so far this run
        if best_fitness_score > best_fitness_score_in_all_generations:
            best_fitness_score_in_all_generations = best_fitness_score
            best_solution_in_all_generations = best_solution

        fitness_values[iteration] = best_fitness_score # set best score this generation to fitness_values[generation]
        # maybe get times out of progress bar # end = time.time()
        # times.append(end - start) # add this generation's delta time to times; len(times) == number_of_generations on return
        # debug
        # print(f"{iteration}th generation took {(end - start):.2f} seconds, fittest individual is {(best_fitness_score*10000):.5f} (dist: {(1/best_fitness_score):.5f}) current best fitness is {(best_fitness_score_in_all_generations*100000):.5f} which is a distance delta of {(1/best_fitness_score_in_all_generations):.5f}")
    runtime = get_progress_info_intervals()
    return best_solution_in_all_generations, fitness_values, times, best_fitness_score_in_all_generations, generation_population_fitness_scores, runtime # need to find min fitness for plotting

# set_default_global_vars() or manually call setters first;
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

accuracies, times, fitness_values_all_runs, data, graph_data = run_experiment(number_of_trials, number_of_cities, number_of_airports)

# Calculate mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

# Calculate mean and standard deviation of times
mean_time = np.mean(times)
std_time = np.std(times)

plot_ga_convergence(graph_data)
