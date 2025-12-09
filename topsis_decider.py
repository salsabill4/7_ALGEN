#ini baru
# topsis_decider.py
import numpy as np
import random

def calculate_statistics(fitness_values):
    fitness = np.array(fitness_values, dtype=float)
    M = np.mean(fitness)
    L = np.max(fitness) - np.min(fitness)
    K = np.std(fitness)
    return M, L, K

def build_ddm(M, L, K):
    ddm = np.array([
        [M, L, K],
        [M, L, K],
        [M, L, K],
        [M, L, K],
    ], dtype=float)
    return ddm

def get_weights():
    return np.array([1/3, 1/3, 1/3], dtype=float)

def topsis_selection(ddm, weights):
    # avoid zero-division
    denom = np.sqrt((ddm**2).sum(axis=0))
    denom[denom == 0] = 1.0
    norm = ddm / denom
    weighted = norm * weights
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)
    dist_best = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))
    scores = dist_worst / (dist_best + dist_worst + 1e-12)
    best_idx = int(np.argmax(scores))
    return best_idx, scores

def select_best_method(fitness_values):
    M, L, K = calculate_statistics(fitness_values)
    if (M == 0 and L == 0 and K == 0) or np.isnan(M) or np.isnan(L) or np.isnan(K):
        methods = ["RWS", "SUS", "TS", "RS"]
        return random.choice(methods), np.array([0,0,0,0], dtype=float)
    ddm = build_ddm(M, L, K)
    weights = get_weights()
    best_idx, scores = topsis_selection(ddm, weights)
    methods = ["RWS", "SUS", "TS", "RS"]
    return methods[best_idx], scores