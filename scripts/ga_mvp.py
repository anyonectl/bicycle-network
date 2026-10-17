# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: bicycle-network:default (3.14.8)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Imports

# %%
from bicycle_network import hex, irsg, demand
import numpy as np
from shapely import Point, LineString
import geopandas as gpd
import pandas as pd
from math import comb, ceil
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Tuple

# %% [markdown]
# ## Grid

# %%
hex_radius = 4
grid = hex.make_test_grid(hex_radius)
grid["centroid"] = grid["geometry"].centroid
grid.plot(figsize=(10, 10))

# %%
grid["centroid"].plot(figsize=(10, 10))


# %% [markdown]
# ## Demand matrix

# %% [markdown]
# ### Create all seed point pairs


# %%
test_pairs = demand.od_pairs(grid)
print(f"no. of seed points: {grid.shape[0]}")
print(f"no. of pairs is actually: {test_pairs.shape[0]}")
print(f"no. of pairs should be: {comb(grid.shape[0], 2)}")
test_pairs.head()


# %% [markdown]
# ### Generate random demand


# %%
n_pairs = test_pairs.shape[0]  # number of seed point pairs
test_demand = demand.generate_weird_demand(n_pairs)
demand.plot_demand_distribution(test_demand)

# %% [markdown]
# ### Create demand matrix

# %%
demand_matrix = demand.create_matrix(test_pairs, test_demand)
print(demand_matrix.info())
demand_matrix.head()

# %%
# With Anastassia's plot:
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
grid.boundary.plot(ax=ax, color="black", zorder=0, lw=0.5)
demand_matrix.plot(ax=ax, lw=demand_matrix.demand_share * 20, alpha=0.2)
ax.set_axis_off()
ax.set_title("Seed point pairs")

# %% [markdown]
# ## Seed point GDF

# %%
seed_points = grid
seed_points.columns = ["id", "geometry", "centroid"]
seed_points["total_demand"] = seed_points.apply(
    lambda x: sum(
        demand_matrix[(demand_matrix.u == x.id) | (demand_matrix.v == x.id)].demand
    ),
    axis=1,
)
print(seed_points.info())
seed_points.head()

# %%
fig, ax = plt.subplots(1, 1)
seed_points.plot(ax=ax, column="total_demand", cmap="Blues", legend=True)
ax.set_axis_off()
ax.set_title("Total demand per hexagon")

# %% [markdown]
# ## Initial Route Set Generation

# %% [markdown]
# ### Variables

# %%
N_ITE: int = 100  # iterations
N_POP: int = 20
MAX_PATHS: int = 10
MAX_NODES: int = 15
SEED: int = 3581054352
RNG: np.random.Generator = np.random.default_rng(seed=SEED)
ROOT = "../../"
PATH = ROOT + "results/plots/"


# %% [markdown]
# ### Helper functions
#
# See src/bicycle_network/hex.py
#
# TODO: Use spatial indexing to find neighbours instead? Sth with trees?


# %%
# TODO: should only add neighbours to vns if they are in the dataset, i.e. within the bounding box
def vns(node: Tuple[int, int]) -> List[Tuple[int, int]]:
    q, r = hex.double_to_axial(node, hex_radius)
    vns = []
    for i in range(6):
        a, b = hex.hex_neighbour((q, r), i)
        u, v = hex.axial_to_double(a, b)
        vns.append(hex.str_id((u, v)))
    return vns


# %% [markdown]
# ### Create Vicinity Node Set (VNS) for each seed point / hexagon

# %%
seed_points["vns"] = seed_points["id"].apply(lambda s: hex.vns(hex.numeric_id(s), hex_radius))


# %% [markdown]
# ### Functions to create single network
# See src/bicycle_network/irsg.py


# %%
initial_node_set = (
    seed_points.copy()
    .sort_values("total_demand", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
type(initial_node_set["total_demand"])

# %% [markdown]
# ### Run IRSG

# %%
n_start_nodes = int(
    len(seed_points) / 2
)  # how many start nodes do we initialise our routes with
start_pop = irsg.generate_population(N_POP, n_start_nodes, MAX_PATHS, MAX_NODES, seed_points, RNG)

# %% [markdown]
# ### Plot start population

# %%
n_cols = 4
n_rows = ceil(N_POP / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
axes = axes.ravel()

cmap = mpl.colormaps.get_cmap("tab10")

for i in range(N_POP):
    ax = axes[i]
    network = start_pop[start_pop["network_id"].values == i]
    colors = cmap(np.linspace(0, 1, len(network)))
    seed_points.boundary.plot(ax=ax, color="grey", lw=1)
    grid["centroid"].plot(ax=ax, color="black", markersize=1)

    for path_id in network["path_id"]:
        network[network["path_id"] == path_id].plot(
            ax=ax, lw=5, alpha=0.5, label=path_id, color=colors[path_id]
        )

    # ax.legend()
    ax.set_title(f"Network {i}")
    ax.set_axis_off()

fig.suptitle("Generation 0", fontsize=16)
plt.tight_layout()

filename = "start_population"
# plt.savefig(f"{PATH}{filename}.png", dpi=300, bbox_inches="tight")
plt.show()
