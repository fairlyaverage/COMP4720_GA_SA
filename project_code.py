import random
import numpy as np

MIN_COORDINATE, MAX_COORDINATE = 0, 10000
n_MIN, n_MAX, N_MIN, N_MAX = 3, 10, 1000, 10000
TEST_CITIES = [ (1,1), (2,2), (3,3), (4,4) ]
TEST_AIRPORTS = [ () ]
POPULATION_SIZE = 1000
MUTATION_PROBABILITY = 0.1 # test ranges from 0.001 to 0.1
cities = []
airports = []
X_MAX, Y_MAX = 10000, 10000


# initial population

# chromosome
chromosome = ()

# fitness function
def fitness(airport, city):
    # { 'x': int, 'y': int }
    return pow(airport['x'] - city['x'], 2) + pow(airport['y'] - city['y'], 2)

    # for element in chromosome:
    #     for city in cities:

# selection function
def selection(population):
    population = [
        [
            (), (), ()
        ],
        [
            (), (), ()
        ],
    ]
    # for each chromosome
    # fitness(chromosome)
    # probability = fitness(chromosome) / (sum of all fitness scores)
    # randomly select pairs

    for pair in range(len(population)):

    return

# crossover function
def crossover(father, mother): # chromosome parents
    selection_point = random.randint(1, len(father))
    return father[:selection_point] + mother[selection_point:] # returns new list

# mutation function
def mutate(chromosome):
    for location in chromosome:
        if (random.random() <= MUTATION_PROBABILITY):
            change_coordinate()

def change_coordinate(coordinate):
    # check that new coordinate doesn't already exist

    return ()

def unique_coordinate(coordinate, chromosome):
    # check against global cities & this chromosome
    return not (coordinate in cities and coordinate in chromosome)
    """
        read locations cities


    """

def random_coordinate():
    x,y = random.randint(0,X_MAX), random.randint(0,Y_MAX)
# main
def set_parameters(n, N, iterations):

    for i in range(iterations):
        # generate city coords
            # when to generate airport coords?
        # call GA
        # call SA
        break
    return -1

# my random coordinate generator from hw2
def generateLocations(n): # -> list:
    # locations = [] # list to hold n unique location tuples (x,y)
    coordinates = np.array([], dtype=np.int64)

    # generate n random x, y coordinates; discretize as integers
    # while len(locations) < n: # produce n unique locations
    while coordinates.shape[0] < n:
        # x = random.randint(MIN_COORDINATE, MAX_COORDINATE) # inclusive range
        x = np.random.randint(MIN_COORDINATE, MAX_COORDINATE + 1)
        # y = random.randint(MIN_COORDINATE, MAX_COORDINATE) # inclusive range
        y = np.random.randint(MIN_COORDINATE, MAX_COORDINATE + 1)
        coordinate = np.array([x, y], dtype=np.int64)
        # if (x,y) in locations: continue # prevent duplicate locations
        if np.isin(coordinate, coordinates).all(axis=1).any(): continue
        # locations.append((x,y)) # add new location tuple (x,y) to list of locations
        coordinates = np.append(coordinates, [coordinate], axis=0)
    return coordinates
    # can also return coordinates.tolist() if numpy array doesn't behave right

def generateTextLocations(n) -> list:
    locations = [] # list to hold n unique location tuples (x,y)
    number_of_locations = 0
    # generate n random x, y coordinates; discretize as integers
    while len(locations) < n: # produce n unique locations
        x = random.randint(MIN_COORDINATE, MAX_COORDINATE) # inclusive range
        y = random.randint(MIN_COORDINATE, MAX_COORDINATE) # inclusive range
        if (x,y) in locations: continue # prevent duplicate locations
        number_of_locations+=1
        locations.append((x,y)) # add new location tuple (x,y) to list of locations
        print(f"#{number_of_locations} ({x},{y})")
    coordinate_file = open('coord.txt', 'w')
    for coord in locations:
        coordinate_file.write(f"{coord[0]} {coord[1]}\n")
    coordinate_file.close()

    return locations
