import numpy as np
import random
from typing import List

# Generates a random undirected graph with N nodes. Returns an adjacency matrix representing the graph
def generate_random_graph(N: int) -> np.ndarray:
    graph: np.ndarray = np.zeros((N, N), dtype=int)

    for i in range(N):
        # Exclude self-loops
        for j in range(i+1, N):
            # Randomly decide if an edge should exist
            edge: int = random.choice([0, 1])
            graph[i][j] = edge
            graph[j][i] = edge

    return graph

def generate_population(population_size: int, num_nodes: int, num_colors: int) -> List[List[int]]:
    population: List[List[int]] = []

    for _ in range(population_size):
        # Assign random colors to each node
        chromosome: List[int] = [random.randint(0, num_colors - 1) for _ in range(num_nodes)]
        population.append(chromosome)

    return population
