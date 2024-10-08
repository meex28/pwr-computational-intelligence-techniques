from typing import List
import random

import numpy as np

from util import random_choice

def calc_fitness(chromosome: List[int], graph: np.ndarray) -> float:
    conflicts: int = 0 # case when neighbour nodes has same color
    num_nodes = len(chromosome)

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if graph[i][j] == 1 and chromosome[i] == chromosome[j]:
                conflicts += 1

    # we want to minimize fitness
    return conflicts

def single_point_crossover(parent1: List[int], parent2: List[int]) -> tuple[List[int], List[int]]:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")

    crossover_point: int = random.randint(1, len(parent1) - 1)

    child1: List[int] = parent1[:crossover_point] + parent2[crossover_point:]
    child2: List[int] = parent2[:crossover_point] + parent1[crossover_point:]

    return child1, child2

def mutate(chromosome: List[int], num_colors: int) -> List[int]:
    chromosome_copy = chromosome[::]

    position = random.randint(0, len(chromosome) - 1)
    chromosome_copy[position] = random.randint(0, num_colors - 1)

    return chromosome_copy

def tournament_selection(population: List[List[int]], graph: np.ndarray, tournament_size: int = 2) -> List[List[int]]:
    population_size = len(population)
    new_population = []

    for _ in range(population_size):
        tournament_chromosomes = random.sample(population, tournament_size)
        winner = min(tournament_chromosomes, key=lambda x: calc_fitness(x, graph))
        new_population.append(winner)

    return new_population

def build_next_generation(population: List[List[int]], graph: np.ndarray, mutation_prob: float, crossover_prob: float, num_colors: int) -> List[List[int]]:
    selected_chromosomes = tournament_selection(population, graph)

    # run crossover
    population_after_crossover = []
    random.shuffle(selected_chromosomes)
    for parent1, parent2 in zip(selected_chromosomes[::2], selected_chromosomes[1::2]):
        if random_choice(crossover_prob):
            child1, child2 = single_point_crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2
        population_after_crossover.append(child1)
        population_after_crossover.append(child2)

    # run mutations
    population_after_mutations = []
    for chromosome in population_after_crossover:
        if random_choice(mutation_prob):
            population_after_mutations.append(mutate(chromosome, num_colors))
        else:
            population_after_mutations.append(chromosome)

    return population_after_mutations
