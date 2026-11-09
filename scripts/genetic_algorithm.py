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
start_pop = ga.json_to_pop(data)

# %%
print(type(start_pop))
print((start_pop))
print(start_pop[0])

# %%
bcnwplt.plot_population_dict(
    start_pop,
    seed_points,
    grid,
    demand_type="absolute",
    save=False,
    output_path=PLOT_PATH,
    filename="start_pop",
    title="",
    gen=0,
)

# %% [markdown]
# ## Run GA

# %%
final_pop = ga.run(start_pop, 10, demand_matrix, seed_points, grid)

# %%
bcnwplt.plot_population_dict(
    final_pop,
    seed_points,
    grid,
    demand_type="absolute",
    save=False,
    output_path=PLOT_PATH,
    filename="end_pop",
    title="",
    gen=-1,
)
