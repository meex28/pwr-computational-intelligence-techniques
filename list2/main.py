from matplotlib import pyplot as plt
import numpy as np

from genetic_methods import build_next_generation, calc_fitness, AlgorithmMethods, AlgorithmParams
from genetic_methods_selection import SelectionFunction, rank_selection, roulette_wheel_selection, tournament_selection
from graph import describe_graph, generate_population, generate_random_graph
from list2.genetic_methods_crossover import CrossoverFunction, single_point_crossover, two_point_crossover, \
    uniform_crossover
from list2.util import save_json


def run_algorithm(
        graph: np.ndarray,
        algorithm_params: AlgorithmParams,
        algorithm_methods: AlgorithmMethods
):
    population_size = algorithm_params[0]
    graph_size: int = graph[0].size
    current_num_colors = graph_size # colors number will be decreased in the loop, now we start with the easiest solution
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
            population = build_next_generation(
                population,
                graph,
                current_num_colors,
                algorithm_params,
                algorithm_methods
            )

    return best_solution, iterations

def run_single_params_set(
        graph: np.ndarray,
        algorithm_params: AlgorithmParams,
        algorithm_methods: AlgorithmMethods,
):
    population_size, mutation_prob, crossover_prob, inv_prob = algorithm_params
    print(f'Start testing: mut_prob={mut_prob}, crossover_prob={crossover_prob}, inv_prob={inv_prob}')
    best_solution, iterations = run_algorithm(graph, algorithm_params, algorithm_methods)
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

    # params
    population_size = 20
    tested_mutation_prob = [0.125, 0.15]
    tested_crossover_prob = [0.3, 0.7]
    tested_inversion_prob = [0.05, 0.075, 0.1, 0.125, 0.15]
    tested_selection_functions = [rank_selection, roulette_wheel_selection, tournament_selection]
    tested_crossover_functions = [single_point_crossover, two_point_crossover, uniform_crossover]

    solutions = []

    for i, mut_prob in enumerate(tested_mutation_prob):
        for j, cross_prob in enumerate(tested_crossover_prob):
            for inv_prob in tested_inversion_prob:
                for selection_function in tested_selection_functions:
                    for crossover_function in tested_crossover_functions:
                        current_alg_params: AlgorithmParams = population_size, mut_prob, cross_prob, inv_prob
                        current_alg_methods: AlgorithmMethods = selection_function, crossover_function
                        best_solution, iterations = run_single_params_set(graph, current_alg_params, current_alg_methods)
                        solution = {
                            "params": {
                                "mut_prob": mut_prob,
                                "cross_prob": cross_prob,
                                "inv_prob": inv_prob,
                                "selection_function": selection_function.__name__,
                                "crossover_function": crossover_function.__name__
                            },
                            "colors_num": best_solution[2],
                            "generations": [colors_num for _, colors_num, _ in iterations],
                            "generations_num": len(iterations),
                        }
                        solutions.append(solution)

    sorted_solutions = sorted(solutions, key=lambda x: x["colors_num"])
    save_json(sorted_solutions, 'list2/results.json')
