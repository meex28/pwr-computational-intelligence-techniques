import numpy as np

from genetic_methods import build_next_generation, calc_fitness
from graph import generate_population, generate_random_graph


def run_algorithm(graph: np.ndarray, population_size: int, mutation_prob: float, crossover_prob: float):
    graph_size: int = graph[0].size
    population = generate_population(population_size, graph_size, 3)
    gens_without_improvement_limit = 100 
    gens_without_improvement = 0
    current_num_colors = graph_size # colors number will be decreased in the loop, now we start with easiest solution
    best_solution = None, graph_size + 1, current_num_colors # (chromosome, fitness, colors number)
    gen_number = 1
    
    while gens_without_improvement < gens_without_improvement_limit:
        fitnesses = [calc_fitness(x, graph) for x in population]
        current_best_fitness = min(fitnesses)
        current_best = population[fitnesses.index(current_best_fitness)]

        print(f'GENERATION {gen_number}: colors_number={best_solution[2]}, best_fitness={best_solution[1]}, current_best_fitness={current_best_fitness}')

        if current_best_fitness < best_solution[1] or current_num_colors < best_solution[2]:
            best_solution = current_best, current_best_fitness, current_num_colors
            gens_without_improvement = 0

        gens_without_improvement += 1
        gen_number += 1

        if current_best_fitness == 0:
            # in next iteration try less colors
            current_num_colors -= 1 
            # generate new population for less colors
            population = generate_population(population_size, graph_size, current_num_colors)
        else:
            population = build_next_generation(population, graph, mutation_prob, crossover_prob, current_num_colors)

    return best_solution

if __name__ == "__main__":
    graph = generate_random_graph(16)
    best_solution = run_algorithm(graph, 100, 0.1, 0.7)
    print(f'Best solution: {best_solution[0]} with fitness {best_solution[1]} and colors number {best_solution[2]}')
