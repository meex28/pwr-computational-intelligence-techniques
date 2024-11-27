import random
import json
import csv

def random_choice(success_probability: float) -> bool:
    return random.random() < success_probability

def save_json(data, filename: str):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=2)

def save_csv(filename, data, header):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data)

