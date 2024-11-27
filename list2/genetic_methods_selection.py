from typing import Callable, List
import random

import numpy as np

from list2.genetic_method_fitness import calc_fitness

SelectionFunction = Callable[[List[List[int]], np.ndarray], List[List[int]]]

def tournament_selection(population: List[List[int]], graph: np.ndarray) -> List[List[int]]:
    tournament_size = 2
    population_size = len(population)
    new_population = []

    for _ in range(population_size):
        tournament_chromosomes = random.sample(population, tournament_size)
        winner = min(tournament_chromosomes, key=lambda x: calc_fitness(x, graph))
        new_population.append(winner)

    return new_population

def roulette_wheel_selection(population: List[List[int]], graph: np.ndarray) -> List[List[int]]:
    population_size = len(population)
    fitness_scores = [calc_fitness(chromosome, graph) for chromosome in population]

    # adjust fitnesses valuesto maximaze it (instead of minimize)
    max_fitness = max(fitness_scores)
    adjusted_fitness = [max_fitness - score + 1 for score in fitness_scores]

    total_fitness = sum(adjusted_fitness)
    selection_probs = [score / total_fitness for score in adjusted_fitness]

    # Roulette wheel selection: 'spin' the wheel for each new member
    # Higher fitness (lower conflicts) means higher chance of selection
    new_population = []
    for _ in range(population_size):
        spin = random.random()
        cumulative_prob = 0
        for i, prob in enumerate(selection_probs):
            cumulative_prob += prob
            if spin <= cumulative_prob:
                new_population.append(population[i])
                break

    return new_population

def rank_selection(population: List[List[int]], graph: np.ndarray) -> List[List[int]]:
    population_size = len(population)
    
    # Calculate fitness scores and sort population
    fitness_and_index = [(calc_fitness(chrom, graph), i) for i, chrom in enumerate(population)]
    fitness_and_index.sort()  # Sort by fitness (lower is better)
    
    # Assign ranks (1 is best, population_size is worst)
    ranks = list(range(1, population_size + 1))
    
    # Calculate selection probabilities based on ranks
    total_rank = sum(ranks)
    selection_probs = [(population_size - rank + 1) / total_rank for rank in ranks]
    
    new_population = []
    # Rank selection: select based on the assigned ranks
    # Higher rank (lower index in sorted list) means higher chance of selection
    for _ in range(population_size):
        spin = random.random()
        cumulative_prob = 0
        for i, prob in enumerate(selection_probs):
            cumulative_prob += prob
            if spin <= cumulative_prob:
                selected_index = fitness_and_index[i][1]
                new_population.append(population[selected_index])
                break
    
    return new_population
