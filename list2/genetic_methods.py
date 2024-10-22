from typing import List, Tuple
import random

import numpy as np
from genetic_methods_selection import SelectionFunction
from list2.genetic_method_fitness import calc_fitness
from list2.genetic_methods_crossover import CrossoverFunction

from util import random_choice

# [population_size, mutation_prob, crossover_prob, inversion_prob]
AlgorithmParams = Tuple[int, float, float, float]

# [selection_function, crossover_function]
AlgorithmMethods = Tuple[SelectionFunction, CrossoverFunction]

def mutate(chromosome: List[int], num_colors: int) -> List[int]:
    chromosome_copy = chromosome[::]

    position = random.randint(0, len(chromosome) - 1)
    chromosome_copy[position] = random.randint(0, num_colors - 1)

    return chromosome_copy

def inversion_operator(permutation: List[int]) -> List[int]:
    if len(permutation) < 2:
        return permutation.copy()
    start, end = sorted(random.sample(range(len(permutation)), 2))
    new_permutation = permutation[:start] + permutation[start:end + 1][::-1] + permutation[end + 1:]
    return new_permutation

def tournament_selection(population: List[List[int]], graph: np.ndarray, tournament_size: int = 2) -> List[List[int]]:
    population_size = len(population)
    new_population = []

    for _ in range(population_size):
        tournament_chromosomes = random.sample(population, tournament_size)
        winner = min(tournament_chromosomes, key=lambda x: calc_fitness(x, graph))
        new_population.append(winner)

    return new_population

def build_next_generation(
        population: List[List[int]],
        graph: np.ndarray,
        num_colors: int,
        algorithm_params: AlgorithmParams,
        algorithm_methods: AlgorithmMethods
    ) -> List[List[int]]:
    population_size, mutation_prob, crossover_prob, inv_prob = algorithm_params
    selection_function, crossover_function = algorithm_methods
    selected_chromosomes = selection_function(population, graph)

    # run crossover
    population_after_crossover = []
    random.shuffle(selected_chromosomes)
    for parent1, parent2 in zip(selected_chromosomes[::2], selected_chromosomes[1::2]):
        if random_choice(crossover_prob):
            child1, child2 = crossover_function(parent1, parent2)
        else:
            child1, child2 = parent1, parent2
        population_after_crossover.append(child1)
        population_after_crossover.append(child2)

    # run mutations
    population_after_mutations = []
    for chromosome in population_after_crossover:
        muted_chromosome = chromosome
        if random_choice(mutation_prob):
            muted_chromosome = mutate(chromosome, num_colors)
        if random_choice(inv_prob):
            muted_chromosome = inversion_operator(muted_chromosome)
        population_after_mutations.append(muted_chromosome)

    return population_after_mutations
