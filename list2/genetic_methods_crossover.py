import random
from typing import List, Tuple, Callable

CrossoverFunction = Callable[
    [List[int], List[int]],
    Tuple[List[int], List[int]]
]

def single_point_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")

    crossover_point: int = random.randint(1, len(parent1) - 1)

    child1: List[int] = parent1[:crossover_point] + parent2[crossover_point:]
    child2: List[int] = parent2[:crossover_point] + parent1[crossover_point:]

    return child1, child2

def two_point_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")

    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)

    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    return child1, child2


# randomly select each gene from both parents
def uniform_crossover(parent1: List[int], parent2: List[int], crossover_prob: float = 0.5) -> Tuple[
    List[int], List[int]]:
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")

    child1 = []
    child2 = []

    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < crossover_prob:
            child1.append(gene2)
            child2.append(gene1)
        else:
            child1.append(gene1)
            child2.append(gene2)

    return child1, child2
