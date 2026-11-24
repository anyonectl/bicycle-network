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

# import networkx as nx

# %%
N_ITE: int = 100  # iterations
N_POP: int = 20
MAX_PATHS: int = 10
MAX_NODES: int = 15
SEED: int = 3581054352
RNG: np.random.Generator = np.random.default_rng(seed=SEED)
ROOT = "../"
PLOT_PATH = ROOT + "results/plots/"
DATA_PATH = ROOT + "results/network-data/"
EXT = ".json"
INITIAL_POP_FILE = f"{DATA_PATH}start_pop_abs_connected{EXT}"
DEMAND_MATRIX_FILE = f"{DATA_PATH}demand_matrix.geojson"
GRID_FILE = f"{DATA_PATH}grid.geojson"
SEED_POINTS_FILE = f"{DATA_PATH}seed_points.geojson"

# %% [markdown]
# ### Get demand_matrix etc

# %%
demand_matrix = gpd.read_file(DEMAND_MATRIX_FILE)
grid = gpd.read_file(GRID_FILE)
seed_points = gpd.read_file(SEED_POINTS_FILE)

# %% [markdown]
# ### Get initial population

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
total_degree = seed_points["neighbours"].apply(len).sum()
assert total_degree % 2 == 0

# %%
max_edges = int(seed_points["neighbours"].apply(len).sum() // 2)

# %%
N_ITE = 30  # iterations
MUTATE_PROB = 1.0
M = 15
BUDGET = max_edges // 4
PENALTY = 0.1

print(f"Iterations: {N_ITE}")
print(f"Population size: {len(start_pop_new)}")
print(f"m: {M}")
print(f"Max edges: {max_edges}")
print(f"Budget: {BUDGET}")
print(f"Penalty weight: {PENALTY}")
print(f"Mutation probability: {MUTATE_PROB}")


generations = ga.run(
    start_pop_new,
    N_ITE,
    MUTATE_PROB,
    M,
    BUDGET,
    max_edges,
    demand_matrix,
    seed_points,
    grid,
    PENALTY,
)

# %% [markdown]
# ### Check last generation
#
# # %%
# n_gens = len(generations)
# final_pop = generations[n_gens - 1]
#
# # %%
# bcnwplt.plot_population_dict(
#     final_pop,
#     seed_points,
#     grid,
#     demand_type="absolute",
#     save=True,
#     output_path=PLOT_PATH,
#     filename="finalpop_allmutated",
#     gen=n_gens,
# )
#
# # %%
# print(type(final_pop))
# print(final_pop)
# print(final_pop[0])

# %% [markdown]
# ## Evaluate generations
# Scatter plot with x-axis the total length of a network,
# y-axis the percentage of demand met
#
# Can use the shapely objects to calc length - just need to map
# the cell ids to their shapely points first
#
# Can maybe also save a metric demand per length

# %%
# TODO Placeholder for getting the right data/structure before plotting

# %%
# TODO Placeholder for scatter plot

# %% [markdown]
# ## Save generations as JSON

# %%
filename = "generations_261124"
with open(f"{DATA_PATH}{filename}.json", "w") as json_file:
    json.dump(ga.generations_to_jsonable(generations), json_file, indent=4)

# %% [markdown]
# # Evaluation/plots
# ## Get number of edges
#
# Can use the shapely objects to calc length - just need to map
# the cell ids to their shapely points first
#
# Can maybe also save a metric demand per length
#
# When calculating length, we should not count the edges that are used by more than one path.

# %% [markdown]
# For a grid of 272 regular hexagons, we have:
# - 210 interior cells with 6 neighbours
# - 28 side cells with 4 neighbours
# - 15 protruding top/bottom non-corner cells with 3 neighbours
# - 15 non-protruding top/bottom non-corner cells with 5 neighbours
# - 2 protruding corner cells (bottom) with 2 neighbours
# - 2 non-protruding corner cells (top) with 3 neighbours
#
# Which gives a total cell count of 272, and a total number of edges
# in the dual (?) graph of 751 edges.

# %%
# Calculate number of edges in each network
n_edges_dict = {}
for pop in generations.values():
    for nw_id, nw in pop.items():
        n_edges_dict[nw_id] = nw["graph"].number_of_edges()

# %% [markdown]
# ## Plot

# %% [markdown]
# ### Scatter plot
# Scatter plot with x-axis the total length of a network,
# y-axis the percentage of demand met

# %%
bcnwplt.scatter(generations, n_edges_dict, save=True, output_path=PLOT_PATH)

# %%
for pop_id in [0, N_ITE // 2, N_ITE - 1]:
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
