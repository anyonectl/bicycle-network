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
# # Initial Route Set Generation (IRSG)

# %% [markdown]
# ## Setup

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import json
from bicycle_network import irsg, plot
import numpy as np
import geopandas as gpd

# %% [markdown]
# ### Variables

# %%
N_POP: int = 20
MAX_PATHS: int = 10
MAX_NODES: int = 15
SEED: int = 3581054352
RNG: np.random.Generator = np.random.default_rng(seed=SEED)
ROOT = "../"
PLOT_PATH = ROOT + "results/plots/"
DATA_PATH = ROOT + "results/network-data/"
GRID_FILE = f"{DATA_PATH}grid.geojson"
SEED_POINTS_FILE = f"{DATA_PATH}seed_points.geojson"
DEMAND_MATRIX_FILE = f"{DATA_PATH}demand_matrix.geojson"

# %% [markdown]
# ### Load data

# %%
grid = gpd.read_file(GRID_FILE)
seed_points = gpd.read_file(SEED_POINTS_FILE)
demand_matrix = gpd.read_file(DEMAND_MATRIX_FILE)

# %% [markdown]
# ## Run IRSG

# %%
# how many nodes should be used in the INS (initial node set)
k = int(len(seed_points) / 2)

# %% [markdown]
# ### Absolute demand

# %%
start_pop = irsg.generate_population(
    N_POP, k, MAX_PATHS, MAX_NODES, seed_points, demand_matrix, RNG
)

# %%
print(start_pop)

# %%
filename = "start_population"
with open(f"{DATA_PATH}{filename}.json", "w") as json_file:
    json.dump(start_pop, json_file, indent=4)

# %%
plot.start_population(
    start_pop,
    seed_points,
    grid,
    save=True,
    output_path=PLOT_PATH,
)
