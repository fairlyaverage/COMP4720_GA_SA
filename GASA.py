import random, math
import matplotlib.pyplot as plt
import numpy as np

MIN_COORDINATE, MAX_COORDINATE = 0, 10000 # location map size; range for x and y values
GENERATIONS = 10000 # number of iterations to run GA and SA algorithms
INITIAL_TEMPERATURE = 90


def generateLocations(n) -> list:
    locations = [] # list to hold n unique location tuples (x,y)

    # generate n random x, y coordinates; discretize as integers
    while len(locations) < n: # produce n unique locations
        x = random.randint(MIN_COORDINATE, MAX_COORDINATE) # inclusive range
        y = random.randint(MIN_COORDINATE, MAX_COORDINATE) # inclusive range
        if (x,y) in locations: continue # prevent duplicate locations
        locations.append((x,y)) # add new location tuple (x,y) to list of locations

    return locations

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

def getRandomPath(locations) -> list:
    """
        accepts list of location coordinates (x,y) and returns a list in random order
        only pass a copy() of the original location list (python is pass by reference only)
    """
    state = []
    while locations:
        state.append(locations.pop(random.randint(0, len(locations) - 1)))
    return state

# Genetic Algorithm
class GA:
    def __init__():
        return

    def runGA(locations):
        return

# Simulated Annealing
class SA:
    locations = None
    SA_history = []
    best_state = None # (cost(state), state, generation)

    # initialize state
    state = None
    new_state = None

    """
        1. initial random state
        2. create new state with random change (swap 2 cities)
        3. SA calculates probability to move from s to s'
        4. decrease temperature (likelihood) of randomly selecting different option
        5. # Generations
    """

    def __init__(self, locations):
        self.locations = locations
        self.state = getRandomPath(locations.copy())
        self.SA_history = []
        self.best_state = (cost(self.state), self.state, -1)

    def probability(state, new_state, temperature):
        """
            The probability of making the transition from the current state s to a candidate new state s'
            probability of moving to new state:
                cost(state), cost(new_state) are the energy costs E(s) and E(s')
                P = 1 if cost(new_state) < cost(state)
                P = e^(-(cost(new_state) - cost(state))/T)
                T is temperature, integer which reduces every iteration/generation
                math.exp(v) is e^(v) function


                delta = cost(new_state) - cost(state)
                    <= 0 sometimes pick
                    > 0  always pick

            returns
                probability value?
                either one of state/new state?
                boolean value? T if transition occurs, F otherwise?
        """
        delta = cost(new_state) - cost(state)
        if (delta > 0): # new_state is a better path, always pick this one
            return new_state
        # probability = math.exp(delta/temperature) == e^(-(cost(new_state) - cost(state))/T)
        # rand = random.random() # returns 0 <= random.random() <= 1
        if (random.random() <= math.exp(-delta/temperature)): # if random value 0<r<1 is less than probability: do it
            return new_state
        return state # otherwise, don't transition to a new state

    def swap(state) -> list:
        """ randomly select two locations in the state and swap their positions """
        i = random.randint(0, len(state) - 1)
        # j = random.randint(0, len(state))
        while not (i == (j:=random.randint(0, len(state) - 1))): # should keep trying to assign a random j until i!=j
            # DEBUG
            print(f"swapping {state[i]} and {state[j]} in state={state}")
            # once i != j, swap and return
            state[i], state[j] = state[j], state[i]

            #DEBUG
            print(f"now state={state}; returning")

            return state

        # while not (j:= random.randint(len(state)) == i): #
        #     state[i], state[j] = state[j], state[i]

        #DEBUG
        print("If you see this, while loop is broken")

        return state

    def runSA(self):
        for t in range(GENERATIONS):
            # t increments every generation, so INITIAL_TEMPERATURE - t decrements toward zero

            # create candidate state
            self.new_state = SA.swap(self.state.copy()) # getRandomPath(locations.copy())

            # probabilistically determine if state = probability(state, new_state, TEMPERATURE - t)
            self.state = SA.probability(self.state, self.new_state, INITIAL_TEMPERATURE - t*.0001)

            path_cost = cost(self.state)

            # record log
            self.SA_history.append((path_cost, self.state, t)) # (cost(state), state, generation) but generation should equal index

            # update best
            if (path_cost < self.best_state[0]): self.best_state = (path_cost, self.state, t) # always keep track of best solution

            #begin next generation

        plotData(self.SA_history)

        return self.best_state

def plotData(history):
    data = np.array([[i, history[i][0]] for i in range(len(history))])
    x,y = data.T
    plt.scatter(x,y)
    # matplotlib.plot()
    # matplotlib.title()
    plt.show()

# both GA and SA need this
def cost(state):
    """ utility function,scatte returns the cost of a path from state[0] to state[len(state)]

        distance calculation sqrt((x_i+1 - x_i)^2 + (y_i+1 - y_i)^2)

    """
    # initialize path_cost to the [return] cost between first and final location
    path_cost = math.sqrt(math.pow(state[-1][0] - state[0][0], 2) + math.pow(state[-1][1] - state[0][1], 2))

    # sum the path cost for each subsequent location
    for i in range(len(state) - 1): # len() - 1 for readability
        path_cost += math.sqrt(math.pow(state[i+1][0] - state[i][0], 2) + math.pow(state[i+1][1] - state[i][1], 2))

    return path_cost
