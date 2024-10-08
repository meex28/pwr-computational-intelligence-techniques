import random


def random_choice(success_probability: float) -> bool:
    return random.random() < success_probability