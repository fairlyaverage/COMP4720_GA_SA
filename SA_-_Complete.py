import numpy as np
import matplotlib.pyplot as plt
import time

def plot_all_iterations(results):
    plt.figure(figsize=(10, 10))

    for iteration, (N, n, city_coords, airports, energies) in enumerate(results):
        plt.scatter(city_coords[:, 0], city_coords[:, 1], marker='.', color='gray', alpha=0.1)
        plt.scatter(airports[:, 0], airports[:, 1], marker='*', color='red', alpha=0.1)

    plt.title(f'All Iterations: Cities and Airport Locations')
    plt.show()

def plot_convergence(results):
    plt.figure(figsize=(10, 10))

    for iteration, (N, n, city_coords, airports, energies) in enumerate(results):
        plt.plot(energies, alpha=0.1)

    plt.title(f'All Convergence Plots for SA')
    plt.xlabel('Number of steps')
    plt.ylabel('Energy')
    plt.show()

# fitness?
def energy(state, city_coords):
    # sqrt adds overhead, not necessary b/c if sqrt(a^2+b^2) = c, sqrt(a'^2 + b'^2) = c' and c < c', then c^2 < c'^2
    dists = np.sqrt(np.sum(np.min((city_coords.reshape(-1, 1, 2) - state.reshape(1, -1, 2)) ** 2, axis=1)))
    return np.sum(dists ** 2)

def transition(state, T):
    new_state = state.copy()
    idx = np.random.choice(n)
    new_state[idx * 2:idx * 2 + 2] += np.random.normal(0, T, size=2)
    return new_state

def accuracy(state, city_coords, N, n):
    distances = np.sqrt(np.sum(np.min((city_coords.reshape(-1, 1, 2) - state.reshape(1, -1, 2)) ** 2, axis=1))) # calculate distances
    sum_distances = np.sum(distances)
    return sum_distances / (N * n)# (Sum of the distance between each of the N cities and its nearest airport) / (N × n)

def run_simulation(iterations, N, n, plot_results=True):
    T_start = 1.0
    T_end = 0.001
    n_steps = 10000

    results = []
    times = []
    accuracies = []

    for iteration in range(iterations):
        start_time = time.time()

        np.random.seed(None)
        city_coords = np.random.uniform(low=0, high=100, size=(N, 2))

        state = np.random.uniform(size=n * 2)

        T = T_start
        energies = [energy(state, city_coords)]
        for i in range(n_steps):
            T = T_start * (T_end / T_start) ** (i / n_steps)
            new_state = transition(state, T)
            delta_E = energy(new_state, city_coords) - energy(state, city_coords)
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                state = new_state
                energies.append(energy(state, city_coords))

        airports = state.reshape(-1, 2)
        results.append((N, n, city_coords, airports, energies))
        acc = accuracy(state, city_coords, N, n)
        accuracies.append(acc)

        # log duration of each run
        end_time = time.time()
        elapsed_time = (end_time - start_time) # / (N * n)
        times.append(elapsed_time)

        print(
            f"Iteration {iteration + 1} / {iterations} | Energy: {energies[-1]:.2f} | Accuracy: {acc:.2f} | Time: {elapsed_time:.2f} seconds")

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    mean_time = np.mean(times)
    std_time = np.std(times)

    if plot_results:
        plot_convergence(results)

    return mean_accuracy, std_accuracy, mean_time, std_time


# Can replace the values of N and n in the run_simulation function call to experiment with different scenarios.
#  i.e. want to run 5000 times : N=5000 and n=5
iterations = 2
N = 3
n = 1
mean_accuracy, std_accuracy, mean_time, std_time = run_simulation(iterations, N, n)

# Print the results
print(f"\nMean accuracy: {mean_accuracy}")
print(f"Standard deviation of accuracy: {std_accuracy}")
print(f"\nMean time: {mean_time}")
print(f"Standard deviation of time: {std_time:.10f} seconds")

