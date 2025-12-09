#ini baru
# ga_core.py
import random
import numpy as np

# ========================
# Decode chromosome -> routes (split-sequential greedy)
# chromosome is a permutation of customers indices (1..N)
# demands: array indexed by node id (0 is depot demand=0)
# capacity: vehicle capacity
# coords: array of coordinates including depot at index 0
# distance_matrix: full matrix including depot at index 0
# ========================
def decode_chromosome_to_routes(chromosome, demands, capacity):
    """
    Greedy sequential split:
    read chromosome left-to-right, add customer to current route until adding would exceed capacity,
    then start new route. Each route is a list of nodes EXCLUDING depot; we'll add depot when computing distance.
    """
    routes = []
    current_route = []
    current_load = 0

    for customer in chromosome:
        d = demands[customer]
        if current_load + d <= capacity:
            current_route.append(customer)
            current_load += d
        else:
            # finish current route
            if current_route:
                routes.append(current_route)
            # start new route with this customer
            current_route = [customer]
            current_load = d

    if current_route:
        routes.append(current_route)
    return routes


# ========================
# compute route distance (route excludes depot index)
# distance_matrix includes depot at index 0; customers use their index accordingly
# route e.g. [3,5,2]
# return distance including depot->first and last->depot
# ========================
def route_distance(route, distance_matrix):
    if len(route) == 0:
        return 0.0
    dist = 0.0
    # depot assumed index 0
    prev = 0  # start at depot (0)
    for cust in route:
        dist += distance_matrix[prev][cust]
        prev = cust
    # return to depot
    dist += distance_matrix[prev][0]
    return dist


# ========================
# total distance for a chromosome
# ========================
def total_distance_chromosome(chromosome, distance_matrix, demands, capacity):
    routes = decode_chromosome_to_routes(chromosome, demands, capacity)
    total = 0.0
    for r in routes:
        total += route_distance(r, distance_matrix)
    return total, routes


# ========================
# fitness: minimize total distance
# We also add heavy penalty if any route exceeds capacity (shouldn't occur with greedy split,
# but keep penalty if user changes decode)
# ========================
def fitness(chromosome, distance_matrix, demands, capacity, penalty_factor=1e6):
    total_dist, routes = total_distance_chromosome(chromosome, distance_matrix, demands, capacity)
    # check overload
    overload = 0.0
    for r in routes:
        load = sum(demands[c] for c in r)
        if load > capacity:
            overload += (load - capacity)
    # smaller distance -> larger fitness
    fit = 1.0 / (total_dist + 1e-9)
    if overload > 0:
        fit = fit / (1.0 + penalty_factor * overload)
    return fit


# ========================
# initialize population: permutation of customers only (exclude depot 0)
# customers indices should be 1..N (assuming 0 is depot)
# ========================
def init_population(pop_size, num_customers):
    population = []
    base = list(range(1, num_customers + 1))
    for _ in range(pop_size):
        chrom = base.copy()
        random.shuffle(chrom)
        population.append(chrom)
    return population


# ========================
# OX Crossover for permutations (works on customer-only permutations)
# ========================
def ox_crossover(parent1, parent2):
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b] = parent1[a:b]
    pos = b % n
    for city in parent2:
        if city not in child:
            child[pos] = city
            pos = (pos + 1) % n
    return child


# ========================
# inversion mutation
# ========================
def inversion_mutation(chromosome):
    a, b = sorted(random.sample(range(len(chromosome)), 2))
    return chromosome[:a] + chromosome[a:b][::-1] + chromosome[b:]


# ========================
# Replacement: mix parents + offspring, keep best pop_size individuals
# ========================
def replace_population(population, offspring, distance_matrix, demands, capacity):
    combined = population + offspring
    # sort by fitness (desc)
    combined_sorted = sorted(
        combined,
        key=lambda c: fitness(c, distance_matrix, demands, capacity),
        reverse=True
    )
    return combined_sorted[:len(population)]