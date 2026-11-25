# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Genetic Algorithm
#
# Starting from a population of connected networks.

# %% [markdown]
# ## Setup

# %%
import json
import numpy as np
import geopandas as gpd
import bicycle_network.genetic_algorithm as ga
from bicycle_network import irsg
import bicycle_network.plot as bcnwplt

# %%
SEED: int = 3581054352
RNG: np.random.Generator = np.random.default_rng(seed=SEED)
ROOT = "../"
PLOT_PATH = ROOT + "results/plots/"
DATA_PATH = ROOT + "results/network-data/"
INITIAL_POP_FILE = f"{DATA_PATH}start_pop_abs_connected.json"
DEMAND_MATRIX_FILE = f"{DATA_PATH}demand_matrix.geojson"
GRID_FILE = f"{DATA_PATH}grid.geojson"
SEED_POINTS_FILE = f"{DATA_PATH}seed_points.geojson"

# %% [markdown]
# ### Load data

# %%
demand_matrix = gpd.read_file(DEMAND_MATRIX_FILE)
grid = gpd.read_file(GRID_FILE)
seed_points = gpd.read_file(SEED_POINTS_FILE)

# %% [markdown]
# #### Initial population

# %%
with open(INITIAL_POP_FILE, "r") as fp:
    data = json.load(fp)
start_pop_old = irsg.json_to_pop(data)
start_pop_new = ga.json_to_pop(data)

# %%
print(type(start_pop_old))
print((start_pop_old))
print(start_pop_old[0])

# %%
# bcnwplt.plot_population_dict(
#     start_pop_old,
#     seed_points,
#     grid,
#     demand_type="absolute",
#     save=True,
#     output_path=PLOT_PATH,
#     filename="261124_start_pop",
#     title="",
#     gen=0,
# )

# %% [markdown]
# ## Run GA

# %%
N_ITE = 100  # iterations
MUTATE_PROB = 0.4
M = 50
MAX_EDGES = int(seed_points["neighbours"].apply(len).sum() // 2)
BUDGET = MAX_EDGES // 5
PENALTY = 1.3

generations = ga.run(
    start_pop_new,
    N_ITE,
    MUTATE_PROB,
    M,
    BUDGET,
    MAX_EDGES,
    demand_matrix,
    seed_points,
    grid,
    PENALTY,
    SEED,
)

print(f"Iterations: {N_ITE}")
print(f"Population size: {len(start_pop_new)}")
print(f"m: {M}")
print(f"Max edges: {MAX_EDGES}")
print(f"Budget: {BUDGET}")
print(f"Penalty weight: {PENALTY}")
print(f"Mutation probability: {MUTATE_PROB}")

# %% [markdown]
# ## Save generations as JSON

# %%
filename = "generations_261125"
with open(f"{DATA_PATH}{filename}.json", "w") as json_file:
    json.dump(ga.generations_to_jsonable(generations), json_file, indent=4)

# %% [markdown]
# # Evaluation/plots

# %% [markdown]
# ## Plot

# %% [markdown]
# ### Scatter plot
# Scatter plot with x-axis the total length of a network,
# y-axis the percentage of demand met

# %%
bcnwplt.scatter(generations, BUDGET, save=True, output_path=PLOT_PATH)

# %%
for pop_id in [0, N_ITE // 4, N_ITE // 2, (N_ITE * 3) // 4, N_ITE]:
    bcnwplt.plot_population_nx(
        generations[pop_id],
        seed_points,
        grid,
        demand_type="absolute",
        save=True,
        output_path=PLOT_PATH,
        filename="mutate_and_crossover",
        gen=pop_id,
    )

# %% [markdown]
# ### Violin plot
# One violin per generation to show demand coverage distribution for all networks in the generation.
# Should include at least 5 gens, maybe all if there aren't that many
