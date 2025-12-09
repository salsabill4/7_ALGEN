#ini yg baru
# app.py
# app.py (fixed, full file)
import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns

from ga_core import (
    init_population,
    fitness,
    ox_crossover,
    inversion_mutation,
    replace_population,
    total_distance_chromosome,
    decode_chromosome_to_routes
)
from selection_methods import (
    rws_selection,
    sus_selection,
    tournament_selection,
    rank_selection
)
from topsis_decider import select_best_method

# ---------- Page config ----------
st.set_page_config(page_title="GA + SODGA for CVRP (use CVRPLIB files)", layout="wide")

# ---------- Session state init ----------
for key in [
    "coords", "demands", "capacity", "num_customers",
    "distance_matrix", "mapping_orig_to_int", "mapping_int_to_orig"
]:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("GA + SODGA for CVRP — load CVRPLIB instance (folder A)")

# ---------------------------
# Utilities: parse CVRPLIB format (robust)
# Returns: coords_arr, demands_arr, capacity, num_customers, mapping_orig_to_int, mapping_int_to_orig
# ---------------------------
def parse_cvrplib_text(text):
    lines = text.splitlines()
    coords = {}
    demands = {}
    capacity = None
    depot_candidates = []
    dimension = None

    reading_coords = False
    reading_demands = False
    reading_depot = False

    for raw in lines:
        line = raw.strip()
        if line == "":
            continue
        up = line.upper()
        # DIMENSION
        if up.startswith("DIMENSION"):
            parts = line.replace(":", " ").split()
            for p in parts:
                if p.isdigit():
                    dimension = int(p)
                    break
            continue
        # CAPACITY
        if up.startswith("CAPACITY"):
            parts = line.replace(":", " ").split()
            for p in parts:
                if p.isdigit():
                    capacity = int(p)
                    break
            continue
        if up.startswith("NODE_COORD_SECTION"):
            reading_coords = True
            reading_demands = False
            reading_depot = False
            continue
        if up.startswith("DEMAND_SECTION"):
            reading_demands = True
            reading_coords = False
            reading_depot = False
            continue
        if up.startswith("DEPOT_SECTION"):
            reading_depot = True
            reading_coords = False
            reading_demands = False
            continue

        if reading_coords:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    idx = int(float(parts[0]))
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[idx] = (x, y)
                except:
                    pass
            continue

        if reading_demands:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    idx = int(float(parts[0]))
                    dem = float(parts[1])
                    demands[idx] = dem
                except:
                    pass
            continue

        if reading_depot:
            parts = line.split()
            for p in parts:
                try:
                    v = int(float(p))
                    if v == -1:
                        reading_depot = False
                    else:
                        depot_candidates.append(v)
                except:
                    pass
            continue

    if len(coords) == 0:
        raise ValueError("No NODE_COORD_SECTION found or coordinates missing in the file.")

    # choose depot: prefer first depot candidate if any, else use common defaults
    if len(depot_candidates) > 0:
        depot_file_idx = depot_candidates[0]
    elif 1 in coords:
        depot_file_idx = 1
    elif 0 in coords:
        depot_file_idx = 0
    else:
        depot_file_idx = min(coords.keys())

    # If DIMENSION present, try to use it; else infer from coords (best-effort)
    if dimension is None:
        dimension = max(coords.keys())

    file_indices = sorted(coords.keys())

    # Build mapping so internal indexing (0..N-1) has depot at 0
    mapping_orig_to_int = {}
    mapping_int_to_orig = {}

    ordered = [depot_file_idx] + [idx for idx in file_indices if idx != depot_file_idx]
    N = len(ordered)
    coords_arr = np.zeros((N, 2), dtype=float)
    demands_arr = np.zeros(N, dtype=float)

    for new_idx, orig_idx in enumerate(ordered):
        mapping_orig_to_int[orig_idx] = new_idx
        mapping_int_to_orig[new_idx] = orig_idx
        x, y = coords[orig_idx]
        coords_arr[new_idx, 0] = x
        coords_arr[new_idx, 1] = y
        demands_arr[new_idx] = demands.get(orig_idx, 0.0)

    # depot demand enforced to 0 internally
    demands_arr[0] = 0.0
    num_customers = N - 1

    return coords_arr, demands_arr, capacity, num_customers, mapping_orig_to_int, mapping_int_to_orig

def build_distance_matrix(coords):
    n = len(coords)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            mat[i, j] = np.linalg.norm(coords[i] - coords[j])
    return mat

# ---------------------------
# UI for loading instance
# ---------------------------
st.sidebar.header("Load CVRPLIB instance")
st.sidebar.write("Upload a .vrp/.txt file or paste direct URL to instance file.")

uploaded = st.sidebar.file_uploader("Upload CVRPLIB instance file (.vrp/.txt)", type=["vrp","txt"])
url_input = st.sidebar.text_input("Or paste direct URL to instance file (optional)")
load_button = st.sidebar.button("Load Instance")

# local references (may be updated from session_state)
coords = st.session_state.coords
demands = st.session_state.demands
capacity = st.session_state.capacity
num_customers = st.session_state.num_customers
distance_matrix = st.session_state.distance_matrix
mapping_orig_to_int = st.session_state.mapping_orig_to_int
mapping_int_to_orig = st.session_state.mapping_int_to_orig

if load_button:
    data_text = None
    if uploaded is not None:
        bytes_data = uploaded.read()
        try:
            data_text = bytes_data.decode("utf-8", errors="ignore")
        except:
            data_text = str(bytes_data)
    elif url_input:
        try:
            r = requests.get(url_input, timeout=10)
            r.raise_for_status()
            data_text = r.text
        except Exception as e:
            st.error(f"Failed to download from URL: {e}")
            data_text = None
    else:
        st.warning("Please upload a file or paste a direct URL to instance file.")
        data_text = None

    if data_text:
        try:
            coords_new, demands_new, capacity_new, num_customers_new, mapping_orig_to_int_new, mapping_int_to_orig_new = parse_cvrplib_text(data_text)
            distance_matrix_new = build_distance_matrix(coords_new)
            # SAVE to session state
            st.session_state.coords = coords_new
            st.session_state.demands = demands_new
            st.session_state.capacity = capacity_new
            st.session_state.num_customers = num_customers_new
            st.session_state.distance_matrix = distance_matrix_new
            st.session_state.mapping_orig_to_int = mapping_orig_to_int_new
            st.session_state.mapping_int_to_orig = mapping_int_to_orig_new

            # update local refs
            coords = st.session_state.coords
            demands = st.session_state.demands
            capacity = st.session_state.capacity
            num_customers = st.session_state.num_customers
            distance_matrix = st.session_state.distance_matrix
            mapping_orig_to_int = st.session_state.mapping_orig_to_int
            mapping_int_to_orig = st.session_state.mapping_int_to_orig

            st.success("Instance loaded successfully.")
            st.sidebar.write(f"Customers: {num_customers}, Vehicle capacity: {capacity}")
        except Exception as e:
            st.error(f"Parsing error: {e}")

# update local refs from session state (in case not just loaded)
coords = st.session_state.coords
demands = st.session_state.demands
capacity = st.session_state.capacity
num_customers = st.session_state.num_customers
distance_matrix = st.session_state.distance_matrix
mapping_orig_to_int = st.session_state.mapping_orig_to_int
mapping_int_to_orig = st.session_state.mapping_int_to_orig

if coords is None:
    st.info("Please load a CVRPLIB instance to run GA.")
    st.stop()

# ---------------------------
# GA parameters
# ---------------------------
st.sidebar.header("GA Parameters")
population_size = st.sidebar.number_input("Population Size", 10, 500, 50)
generations = st.sidebar.number_input("Generations", 10, 1000, 200)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.8)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1)
show_anim = st.sidebar.checkbox("Show route animation (best-so-far)", True)
anim_speed = st.sidebar.slider("Animation speed (s)", 0.01, 1.0, 0.1)

run = st.sidebar.button("Run GA")

# ---------------------------
# Show instance summary
# ---------------------------
st.subheader("Instance summary")
st.write(f"Number of customers (internal): {num_customers}")
st.write(f"Vehicle capacity: {capacity}")
st.write("First 10 demands (internal index : demand) :")
dem_list = [{"idx_internal": int(i), "demand": float(d)} for i, d in enumerate(demands)]
st.write(pd.DataFrame(dem_list).head(10))

fig_coords, axc = plt.subplots(figsize=(6, 6))
axc.scatter(coords[:, 0], coords[:, 1])
for i, (x, y) in enumerate(coords):
    axc.text(x, y, str(i))
axc.set_title("Node coordinates (0 = depot, internal index)")
st.pyplot(fig_coords)

st.subheader("Distance matrix heatmap")
fig_hm, axhm = plt.subplots(figsize=(6, 5))
sns.heatmap(distance_matrix, ax=axhm)
st.pyplot(fig_hm)

# ---------------------------
# helpers: convert internal route to file indices for display
# ---------------------------
def internal_route_to_file(route_internal, mapping_int_to_orig_local):
    # mapping_int_to_orig_local: internal_index -> original_file_index
    if mapping_int_to_orig_local is None:
        # fallback: identity mapping
        return [int(c) for c in route_internal]
    return [int(mapping_int_to_orig_local.get(int(c), int(c))) for c in route_internal]

def route_distance(route, distance_matrix_local):
    # route: list of internal customer indices (excludes depot)
    if route is None or len(route) == 0:
        return 0.0
    d = 0.0
    prev = 0
    for c in route:
        d += distance_matrix_local[prev, c]
        prev = c
    d += distance_matrix_local[prev, 0]
    return float(d)

def evaluate_routes_fileindex(routes_internal, distance_matrix_local, demands_local, capacity_local, mapping_int_to_orig_local):
    total = 0.0
    route_strs = []
    feasible = True
    for i, r in enumerate(routes_internal, start=1):
        dist = route_distance(r, distance_matrix_local)
        load = sum(demands_local[c] for c in r) if (r and len(r) > 0) else 0.0
        total += dist
        overloaded = False
        if capacity_local is not None:
            overloaded = load > capacity_local
        if overloaded:
            feasible = False
        # convert to file indices for display
        r_file = internal_route_to_file(r, mapping_int_to_orig_local)
        route_strs.append({
            "route_no": i,
            "route_internal": r,
            "route_file_idx": r_file,
            "load": float(load),
            "distance": float(dist),
            "overloaded": bool(overloaded)
        })
    return float(total), feasible, route_strs

# ---------------------------
# RUN GA
# ---------------------------
if run:
    # ensure we have mapping available (should be set when file loaded)
    mapping_int_to_orig = st.session_state.mapping_int_to_orig
    if mapping_int_to_orig is None:
        # fallback identity: internal index == file index
        mapping_int_to_orig = {i: i for i in range(len(coords))}
        st.session_state.mapping_int_to_orig = mapping_int_to_orig

    # init population
    population = init_population(population_size, num_customers)
    best_fitness_per_gen = []
    best_solution = None
    best_fit_val = -1.0
    topsis_log = []
    route_frames = []

    for gen in range(generations):
        fitness_vals = np.array([fitness(ind, distance_matrix, demands, capacity) for ind in population])
        gen_best_idx = int(np.argmax(fitness_vals))
        gen_best_val = float(fitness_vals[gen_best_idx])
        gen_best = population[gen_best_idx]
        if gen_best_val > best_fit_val:
            best_fit_val = gen_best_val
            best_solution = gen_best.copy()
        best_fitness_per_gen.append(best_fit_val)
        route_frames.append(best_solution.copy())

        best_method, scores = select_best_method(fitness_vals)
        if isinstance(scores, (list, tuple, np.ndarray)) and len(scores) >= 4:
            rws_s, sus_s, ts_s, rs_s = scores[:4]
        else:
            rws_s, sus_s, ts_s, rs_s = (0, 0, 0, 0)
        topsis_log.append({
            "gen": gen, "RWS": float(rws_s), "SUS": float(sus_s),
            "TS": float(ts_s), "RS": float(rs_s), "best": best_method
        })

        parents = []
        for _ in range(population_size):
            if best_method == "RWS":
                selected = rws_selection(population, fitness_vals)
            elif best_method == "SUS":
                selected = sus_selection(population, fitness_vals)
            elif best_method == "TS":
                selected = tournament_selection(population, fitness_vals)
            else:
                selected = rank_selection(population, fitness_vals)
            parents.append(selected.copy())

        offspring = []
        for i in range(0, population_size, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % population_size]
            if np.random.rand() < crossover_rate:
                c1 = ox_crossover(p1, p2)
                c2 = ox_crossover(p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring.append(c1)
            offspring.append(c2)

        for i_m in range(len(offspring)):
            if np.random.rand() < mutation_rate:
                offspring[i_m] = inversion_mutation(offspring[i_m])

        population = replace_population(population, offspring, distance_matrix, demands, capacity)

    # ---------------------------
    # Final results
    # ---------------------------
    st.header("Results")
    st.write("Best chromosome (permutation of internal customer indices):")
    st.write(best_solution)

    total_dist, routes_internal = total_distance_chromosome(best_solution, distance_matrix, demands, capacity)
    st.write(f"Total distance (internal calc): {total_dist:.4f}")

    # Evaluate & display routes in file indices
    total_cost, feasible, route_infos = evaluate_routes_fileindex(routes_internal, distance_matrix, demands, capacity, mapping_int_to_orig)

    # Print routes in CVRPLIB-like format (file indices)
    st.subheader("Routes (displayed as file indices)")
    for info in route_infos:
        route_file = " ".join(str(x) for x in info["route_file_idx"])
        st.write(f"Route #{info['route_no']}: {route_file}")

    st.write(f"Cost {int(round(total_cost))}")
    st.write("Feasible (no overload):", feasible)

    st.subheader("Decoded routes (internal indices with loads and distance):")
    for info in route_infos:
        st.write({
            "route_internal": info["route_internal"],
            "route_file_idx": info["route_file_idx"],
            "load": info["load"],
            "distance": info["distance"],
            "overloaded": info["overloaded"]
        })

    st.subheader("Fitness progression")
    st.line_chart(best_fitness_per_gen)

    st.subheader("TOPSIS log sample")
    topsis_df = pd.DataFrame(topsis_log)
    st.dataframe(topsis_df)

    if show_anim:
        st.subheader("Route animation (best-so-far)")
        placeholder = st.empty()
        for gen_idx, chrom in enumerate(route_frames):
            _, routes_i = total_distance_chromosome(chrom, distance_matrix, demands, capacity)
            fig, ax = plt.subplots(figsize=(6,6))
            ax.scatter(coords[:,0], coords[:,1])
            for j,(x,y) in enumerate(coords):
                ax.text(x,y,str(j))
            for route in routes_i:
                if len(route) == 0:
                    continue
                xs = [coords[0,0]] + [coords[c,0] for c in route] + [coords[0,0]]
                ys = [coords[0,1]] + [coords[c,1] for c in route] + [coords[0,1]]
                ax.plot(xs, ys, linewidth=2)
            ax.set_title(f"Gen {gen_idx}")
            placeholder.pyplot(fig)
            time.sleep(anim_speed)
        placeholder.empty()

    # prepare results for download (convert numpy lists to python lists)
    results = {
        "best_chromosome": list(best_solution) if best_solution is not None else None,
        "best_distance": float(total_cost),
        "routes_internal": routes_internal,
        "routes_file_indices": [info["route_file_idx"] for info in route_infos],
        "fitness_per_generation": best_fitness_per_gen,
        "topsis_log": topsis_log
    }
    st.download_button("Download JSON results", data=json.dumps(results, indent=2), file_name="ga_cvrp_results.json")















    """
    Very lightweight parser for common CVRPLIB instances.
    Expected blocks:
    - NAME:
    - VEHICLES/CAPACITY or CAPACITY
    - NODE_COORD_SECTION (index x y)
    - DEMAND_SECTION (index demand)
    - DEPOT_SECTION (index 0 or depot index)
    Returns: coords (Nx2 with depot first), demands (array with index aligned), capacity, num_customers
    """