import random
import math
import matplotlib.pyplot as plt
import time

# Define the distance function between two cities
def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

# Define the cost function for a given tour
def cost(tour, cities):
    distance_sum = 0
    for i in range(len(tour)):
        distance_sum += distance(cities[tour[i - 1]], cities[tour[i]])
    return distance_sum

# Define the simulated annealing function
def simulated_annealing(cities, temp, cooling_rate):
    current_tour = list(range(len(cities)))
    random.shuffle(current_tour)
    current_cost = cost(current_tour, cities)
    best_tour = current_tour
    best_cost = current_cost

    while temp > 1e-8:
        # Generate a new solution by randomly swapping two cities in the current tour
        new_tour = current_tour.copy()
        i = random.randint(0, len(cities) - 1)
        j = random.randint(0, len(cities) - 1)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_cost = cost(new_tour, cities)

        # Calculate the acceptance probability
        delta_cost = new_cost - current_cost
        acceptance_prob = math.exp(-delta_cost / temp)

        # Decide whether to accept the new solution
        if acceptance_prob > random.random():
            current_tour = new_tour
            current_cost = new_cost

        # Update the best solution if necessary
        if current_cost < best_cost:
            best_tour = current_tour
            best_cost = current_cost

        # Cool down the temperature
        temp *= cooling_rate

    return best_tour, best_cost

st = time.time()

# Set the number of iterations and the range of N
num_iterations = 1000
N_min = 5
N_max = 30

# Loop over the iterations
for i in range(num_iterations):
    # Generate a random value of N
    N = random.randint(N_min, N_max)

    # Generate some random cities
    cities = [(random.uniform(0, 1), random.uniform(0, 1)) for i in range(N)]

    # Solve the TSP using simulated annealing
    temp = 100
    cooling_rate = 0.999
    best_tour, best_cost = simulated_annealing(cities, temp, cooling_rate)

    # Plot the cities and the best tour
    plt.figure(figsize=(8, 8))
    plt.scatter([city[0] for city in cities], [city[1] for city in cities], s=100)
    for i in range(N):
        plt.annotate(str(i), (cities[i][0] + 0.01, cities[i][1] + 0.01), fontsize=12)
    for i in range(N):
        plt.plot([cities[best_tour[i - 1]][0], cities[best_tour[i]][0]],
                 [cities[best_tour[i - 1]][1], cities[best_tour[i]][1]], 'r-', linewidth=2)
    plt.title(f'TSP using Simulated Annealing (cost={best_cost:.3f})')
    plt.show()
    plt.close()



