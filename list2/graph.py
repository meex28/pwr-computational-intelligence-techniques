import numpy as np
import random
from typing import List

from util import random_choice

# Generates a random undirected graph with N nodes. Returns an adjacency matrix representing the graph
def generate_random_graph(N: int, edge_prob: float = 0.3) -> np.ndarray:
    graph: np.ndarray = np.zeros((N, N), dtype=int)

    for i in range(N):
        # Exclude self-loops
        for j in range(i+1, N):
            # Randomly decide if an edge should exist
            edge = 0
            if(random_choice(edge_prob)):
                edge = 1
            graph[i][j] = edge
            graph[j][i] = edge

    return graph

def describe_graph(graph: np.ndarray) -> None:
    num_nodes = len(graph[0])
    num_edges = 0
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if graph[i][j] == 1:
                num_edges += 1
    highest_degree = max([sum(row) for row in graph])
    print(f'Graph: {num_nodes} nodes, {num_edges} edges, {highest_degree} highest degree')

def generate_population(population_size: int, num_nodes: int, num_colors: int) -> List[List[int]]:
    population: List[List[int]] = []

    for _ in range(population_size):
        # Assign random colors to each node
        chromosome: List[int] = [random.randint(0, num_colors - 1) for _ in range(num_nodes)]
        population.append(chromosome)

    return population
