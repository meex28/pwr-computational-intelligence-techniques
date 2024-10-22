from typing import List
import numpy as np

def calc_fitness(chromosome: List[int], graph: np.ndarray) -> float:
    conflicts: int = 0 # case when neighbour nodes has same color
    num_nodes = len(chromosome)

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if graph[i][j] == 1 and chromosome[i] == chromosome[j]:
                conflicts += 1

    # we want to minimize fitness
    return conflicts
