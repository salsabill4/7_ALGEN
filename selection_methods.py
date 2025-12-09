#ini baru
# selection_methods.py
import random
import numpy as np

# RWS expects fitness_values as list/array aligned with population
def rws_selection(population, fitness_values):
    total_fit = float(np.sum(fitness_values))
    if total_fit == 0:
        return random.choice(population)
    pick = random.uniform(0, total_fit)
    current = 0.0
    for chromosome, fit in zip(population, fitness_values):
        current += fit
        if current >= pick:
            return chromosome
    return population[-1]


# SUS: produce n selections but we return one (as earlier design)
def sus_selection(population, fitness_values):
    total_fit = float(np.sum(fitness_values))
    n = len(population)
    if total_fit == 0:
        return random.choice(population)
    point_distance = total_fit / n
    start_point = random.uniform(0, point_distance)
    points = [start_point + i * point_distance for i in range(n)]

    parents = []
    cumulative = 0.0
    idx = 0
    for p in points:
        while cumulative < p and idx < len(fitness_values):
            cumulative += fitness_values[idx]
            idx += 1
        parents.append(population[max(0, idx - 1)])
    return parents[0]


def tournament_selection(population, fitness_values, k=3):
    # pick k random individuals and choose best by fitness
    zipped = list(zip(population, fitness_values))
    selected = random.sample(zipped, min(k, len(zipped)))
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]


def rank_selection(population, fitness_values):
    n = len(population)
    if n == 0:
        return None
    rank_idx = np.argsort(fitness_values)
    ranks = np.empty_like(rank_idx, dtype=float)
    ranks[rank_idx] = np.arange(1, n + 1)
    probs = ranks / ranks.sum()
    idx = np.random.choice(n, p=probs)
    return population[idx]