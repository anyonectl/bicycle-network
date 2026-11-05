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
# ### Dataframes

# %%
grid = gpd.read_file(GRID_FILE)
seed_points = gpd.read_file(SEED_POINTS_FILE)
demand_matrix = gpd.read_file(DEMAND_MATRIX_FILE)

# %% [markdown]
# ## Run IRSG

# %%
n_start_nodes = int(
    len(seed_points) / 2
)  # how many start nodes do we initialise our routes with

# %%
start_pop_abs = irsg.generate_population(
    N_POP,
    n_start_nodes,
    MAX_PATHS,
    MAX_NODES,
    seed_points,
    demand_matrix,
    RNG,
    demand_type="absolute",
)

# %%
start_pop_abs = start_pop_abs[
    [
        "cell_id",
        "node_position",
        "path_id",
        "network_id",
        "demand_coverage",
        "geometry",  # for the path
    ]
].copy()
print(start_pop_abs.info())
start_pop_abs.head(20)

# %%
filename = "start_pop_abs_connected"
ext = "geojson"
start_pop_abs.to_file(f"{DATA_PATH}{filename}.{ext}", driver="GeoJSON")

# %%
start_pop_share = irsg.generate_population(
    N_POP,
    n_start_nodes,
    MAX_PATHS,
    MAX_NODES,
    seed_points,
    demand_matrix,
    RNG,
    demand_type="share",
)

# %%
start_pop_share = start_pop_share[
    [
        "cell_id",
        "node_position",
        "path_id",
        "network_id",
        "demand_coverage",
        "geometry",  # for the path
    ]
].copy()
print(start_pop_share.info())
start_pop_share.head(20)

# %%
filename = "start_pop_share_connected"
ext = "geojson"
start_pop_share.to_file(f"{DATA_PATH}{filename}.{ext}", driver="GeoJSON")

# %%
plot.plot_population(
    start_pop_abs,
    seed_points,
    grid,
    demand_type="absolute",
    save=False,
    output_path=PLOT_PATH,
    title="connected",
)

# %%
plot.plot_population(
    start_pop_share,
    seed_points,
    grid,
    demand_type="share",
    save=False,
    output_path=PLOT_PATH,
    title="connected",
)
