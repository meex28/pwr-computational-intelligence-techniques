from typing import List
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from list2.util import save_csv


@dataclass
class Params:
    mut_prob: float
    cross_prob: float
    inv_prob: float
    selection_function: str
    crossover_function: str

@dataclass
class Solution:
    params: Params
    colors_num: int
    generations: List[int]
    generations_num: int

def read_solutions(file_path: str) -> list[Solution]:
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return [Solution(
                params=Params(
                    mut_prob=sol['params']['mut_prob'],
                    cross_prob=sol['params']['cross_prob'],
                    inv_prob=sol['params']['inv_prob'],
                    selection_function=sol['params']['selection_function'],
                    crossover_function=sol['params']['crossover_function']
                ),
                colors_num=sol['colors_num'],
                generations=sol['generations'],
                generations_num=sol['generations_num']
            ) for sol in data]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading file: {e}")
        raise

def create_solutions_ranking(solutions: list[Solution]):
    ordered_solutions = sorted(solutions, key=lambda x: x.colors_num)

    data = [[sol.colors_num, sol.generations_num, sol.params.mut_prob, sol.params.cross_prob, sol.params.inv_prob, sol.params.selection_function, sol.params.crossover_function] for sol in ordered_solutions]
    header = ['colors_num', 'generations_num', 'mut_prob', 'cross_prob', 'inv_prob', 'selection_function', 'crossover_function']

    save_csv('solutions_ranking.csv', data, header)

def plot_average_by_field(solutions: list[Solution], field_name: str, field_name_localized: str):
    # Group solutions by field values
    grouped_solutions = {}
    for solution in solutions:
        grouped_solutions.setdefault(getattr(solution.params, field_name), []).append(getattr(solution, 'colors_num'))

    # Calculate averages
    field_averages = {str(key): float(np.mean(value)) for key, value in grouped_solutions.items()}
    # Sort the keys numerically
    sorted_keys = field_averages.keys()
    if field_name == 'inv_prob':
        sorted_keys = sorted(field_averages.keys(), key=lambda x: float(x))

    print(field_averages)

    # Create a column plot
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_keys, [field_averages[str(k)] for k in sorted_keys])
    plt.xlabel(field_name)
    plt.ylabel(f'Srednia liczba kolorów')
    plt.title(f'Srednia liczba kolorów w zależności od {field_name_localized}')
    if field_name == 'inv_prob':
        plt.xticks(sorted_keys)
    plt.tight_layout()
    plt.show()

def boxplot_by_field(solutions: list[Solution], field_name: str, field_name_localized: str):
    # Group solutions by field values
    grouped_solutions = defaultdict(list)
    for solution in solutions:
        grouped_solutions[getattr(solution.params, field_name)].append(getattr(solution, 'colors_num'))

    # Check if we have data to plot
    if not grouped_solutions:
        print(f"No data to plot for {field_name}")
        return

    # Extract field values for labels
    field_values = sorted(grouped_solutions.keys())

    # Create the box plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot([grouped_solutions[value] for value in field_values],
                    labels=field_values)

    # Set title and axis labels
    ax.set_title(f'Liczba kolorów w zależności od {field_name_localized}')
    ax.set_xlabel(field_name_localized)
    ax.set_ylabel('Liczba kolorów')

    # Show the plot
    plt.tight_layout()
    plt.show()

def average_by_two_fields(solutions: list[Solution], field_name_1: str, field_name_2: str):
    # Group solutions by field values
    grouped_solutions = {}
    for solution in solutions:
        grouped_solutions.setdefault((getattr(solution.params, field_name_1), getattr(solution.params, field_name_2)), []).append(getattr(solution, 'colors_num'))

    # Calculate averages
    field_averages = {str(key): float(np.mean(value)) for key, value in grouped_solutions.items()}
    # Create a column plot
    plt.figure(figsize=(10, 6))
    plt.bar(field_averages.keys(), field_averages.values())
    plt.ylabel(f'Srednia liczba kolorów')
    plt.title(f'Srednia liczba kolorów w zależności od {field_name_1} i {field_name_2}')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.5)
    plt.xticks(rotation=80)
    plt.show()

def main():
    try:
        solutions = read_solutions('results.json')
        fields = [
            ('inv_prob', "prawd. inwersji"),
            ('selection_function', "funkcja selekcji"),
            ('crossover_function', "funkcja krzyzowania")
        ]
        for i in range(len(fields)):
            for j in range(i+1, len(fields)):
                average_by_two_fields(solutions, fields[i][0], fields[j][0])
        # for field_name, field_name_localized in fields:
        #     plot_average_by_field(top_30_solutions, field_name, field_name_localized)
        #     boxplot_by_field(top_30_solutions, field_name, field_name_localized)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()