import numpy as np
import matplotlib.pyplot as plt
import random
import time
from progress import *
import json
import os
NUMBER_OF_TRIALS = 5
NUMBER_OF_AIRPORTS = 5
NUMBER_OF_CITIES = 500
NUMBER_OF_GENERATIONS = 350
MUTATION_RATE = 0.005
POPULATION_SIZE = 50

def set_default_global_vars():
    # number_of_airports = 5
    # number_of_cities = 500
    # number_of_generations = 250 # convergence always by 500, regularly by 250, both SA and GA by 350
    # mutation_rate = 0.005 # .075 is pretty good but introduces a lot of noise, 0.01 is effective
    # population_size = 50 # even number please
    # number_of_trials = 5

    # set_airports(number_of_airports)
    # set_cities(number_of_cities)
    # set_generations(number_of_generations)
    # set_mutation_rate(mutation_rate)
    # set_population_size(population_size)
    # set_trials(number_of_trials)

    # should set to UC constants now
    set_trials()
    set_airports()
    set_cities()
    set_generations()
    set_mutation_rate()
    set_population_size()

    set_log_path() # for saving
    return True

def set_trials(n=NUMBER_OF_TRIALS):
    global number_of_trials
    number_of_trials = n
    input(f"why doesn't this set {n}?")
    return number_of_trials

def set_airports(n=NUMBER_OF_AIRPORTS):
    global number_of_airports
    number_of_airports = n
    return number_of_airports

def set_cities(n=NUMBER_OF_CITIES):
    global number_of_cities
    number_of_cities = n
    return number_of_cities

def set_generations(n=NUMBER_OF_GENERATIONS):
    global number_of_generations
    number_of_generations = n
    return number_of_generations

def set_mutation_rate(n=MUTATION_RATE):
    global mutation_rate
    mutation_rate = n
    return mutation_rate

def set_population_size(n=POPULATION_SIZE):
    global population_size
    population_size = n
    return population_size

# don't call this twice
def set_log_path():
    global PATH
    PATH = f"logs_{str(math.floor(time.time()))}/"
    os.makedirs(PATH)
    return PATH

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
    if 'generation_population_fitness_scores' in trial_data.keys(): trial_data['generation_population_fitness_scores'] = trial_data['generation_population_fitness_scores'].tolist()

    unique_file_name = trial_data['type'] + str(trial_data['trial_number']) + ".json"
    # save run statistics
    try:
        if (os.path.exists(PATH+unique_file_name)): raise Exception(f"{unique_file_name} already exists")
        with open(PATH + unique_file_name, 'w+') as file:
            json.dump(trial_data, file)
    # except:
    #     print("Something went wrong")
    #     return False
    finally:
        print(f"Successfully saved {unique_file_name}")
    return True

def save_cities(cities):
    save_data(cities.tolist())
    return True

def accuracy(individual, cities, number_of_cities, number_of_airports):
    distances = np.zeros(len(cities))
    for i in range(len(cities)):
        distances[i] = np.min(np.linalg.norm(cities[i] - individual, axis=1))
    return np.sum(distances) / (number_of_cities * number_of_airports)

# energy is inverse
def fitness_prime(individual, cities):
    distances = np.zeros((len(cities), len(individual))) # 2d array for storing locations, return minimum for each city (this is the dist to the closest airport)
    for i in range(len(cities)): # each city
        city_x, city_y = cities[i]
        for j in range(len(individual)): # each airport
            airport_x, airport_y = individual[j]
            distances[i][j] = pow(city_x - airport_x, 2) + pow(city_y - airport_y, 2) # linear distance (x_c - x_a)^2 + (y_c - y_a)^2
    # fitness score: higher is better but this wants to minimize distances
    return 1 / np.sum(distances.min(axis=1)) # returns an individual [solution's] fitness value

def generate_cities(number_of_cities):
    return np.random.rand(number_of_cities, 2) * 100 # gen 2-d array for city locations?; update to read this from file or pass between GA and SA

