from matplotlib import pyplot as plt
import numpy as np

from genetic_methods import build_next_generation, calc_fitness
from graph import describe_graph, generate_population, generate_random_graph


def run_algorithm(graph: np.ndarray, population_size: int, mutation_prob: float, crossover_prob: float):
    graph_size: int = graph[0].size
    current_num_colors = graph_size # colors number will be decreased in the loop, now we start with easiest solution
    population = generate_population(population_size, graph_size, current_num_colors)
    gens_without_improvement_limit = 100 
    gens_without_improvement = 0
    best_solution = None, graph_size + 1, current_num_colors # (chromosome, fitness, colors number)
    gen_number = 1
    iterations = []
    
    while gens_without_improvement < gens_without_improvement_limit:
        fitnesses = [calc_fitness(x, graph) for x in population]
        current_best_fitness = min(fitnesses)
        current_best = population[fitnesses.index(current_best_fitness)]

        iterations.append((gen_number, best_solution[2], best_solution[1]))

        if current_best_fitness < best_solution[1] or current_num_colors < best_solution[2]:
            best_solution = current_best, current_best_fitness, current_num_colors
            gens_without_improvement = 0

        gens_without_improvement += 1
        gen_number += 1

        if current_best_fitness == 0:
            # in next iteration try less colors
            current_num_colors -= 1 
            # generate new population for fewer colors
            population = generate_population(population_size, graph_size, current_num_colors)
        else:
            population = build_next_generation(population, graph, mutation_prob, crossover_prob, current_num_colors)

    return best_solution, iterations

def run_single_params_set(graph: np.ndarray, population_size: int, mut_prob: float, crossover_prob: float):
    print(f'Start testing: mut_prob={mut_prob}, crossover_prob={crossover_prob}')
    best_solution, iterations = run_algorithm(graph, population_size, mut_prob, crossover_prob)
    print(f'Best solution: {best_solution[2]} colors in {len(iterations)} generations')
    return best_solution, iterations

def plot_iterations(iterations, mutation_prob, crossover_prob, ax):
    gen_numbers = [gen for gen, _, _ in iterations]
    colors_numbers = [colors for _, colors, _ in iterations]
    
    ax.plot(gen_numbers, colors_numbers, label=f'Mut: {mutation_prob}, Cross: {crossover_prob}')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Colors Number')
    ax.set_title(f'Mutation: {mutation_prob}, Crossover: {crossover_prob}')
    ax.legend()

if __name__ == "__main__":
    graph = generate_random_graph(48)
    describe_graph(graph)
    
    mutation_prob = [0.05, 0.075, 0.1, 0.125, 0.15]
    crossover_prob = [0.1, 0.3, 0.5, 0.7, 0.9]
    population_size = 20
    fig, axes = plt.subplots(len(mutation_prob), len(crossover_prob), figsize=(15, 10))
    axes = axes.flatten()
    solutions = []

    for i, mut_prob in enumerate(mutation_prob):
        for j, cross_prob in enumerate(crossover_prob):
            best_solution, iterations = run_single_params_set(graph, population_size, mut_prob, cross_prob)
            plot_iterations(iterations, mut_prob, cross_prob, axes[i * len(crossover_prob) + j])
            solutions.append((best_solution[2], len(iterations), (mut_prob, cross_prob)))

    sorted_solutions = sorted(solutions, key=lambda x: x[0])

    print("Solutions ranking:")
    for i, sol in enumerate(sorted_solutions):
        print(f"{i+1}. {sol[0]} colors in {sol[1]} generations (mut_prob={sol[2][0]}, cross_prob={sol[2][1]})")

    plt.tight_layout(h_pad=2.0)
    plt.savefig('graph_coloring_gen_alg_results.png')

