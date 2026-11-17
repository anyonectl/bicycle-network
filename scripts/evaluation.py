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
# # Evaluate GA-produced generations and networks

# %% [markdown]
# ## Setup

# %%
import json

from matplotlib import pyplot as plt
from shapely import MultiLineString
from bicycle_network.genetic_algorithm import Network
import bicycle_network.genetic_algorithm as ga
import bicycle_network.graph
import geopandas as gpd

# %%
ROOT = "../results/"
PLOT_PATH = ROOT + "plots/"
DATA_PATH = ROOT + "network-data/"
GEN_FILE = f"{DATA_PATH}generations_261113.json"
# DEMAND_MATRIX_FILE = f"{DATA_PATH}demand_matrix.geojson"
# GRID_FILE = f"{DATA_PATH}grid.geojson"
SEED_POINTS_FILE = f"{DATA_PATH}seed_points.geojson"

# %% [markdown]
# ### Load data

# %%
# demand_matrix = gpd.read_file(DEMAND_MATRIX_FILE)
# grid = gpd.read_file(GRID_FILE)
seed_points = gpd.read_file(SEED_POINTS_FILE)

# %%
with open(GEN_FILE, "r") as fp:
    data = json.load(fp)
gens = ga.json_to_gens(data)

# %%
print(type(gens))
print((gens))
print(gens[0])

# %%
seed_points.head()


# %% [markdown]
# ## Plot
# Scatter plot with x-axis the total length of a network,
# y-axis the percentage of demand met
#
# Can use the shapely objects to calc length - just need to map
# the cell ids to their shapely points first
#
# Can maybe also save a metric demand per length
#
# When calculating length, we should not count the edges that are used by more than one path.

# %%
# TODO Placeholder for getting the right data/structure before plotting
id_to_centroid = seed_points.copy().set_index("id")["geometry"].to_dict()
print(id_to_centroid)


# %%
# Uses multilinestring, counts overlapping lines which we don't want
def network_len(nw: Network) -> float:
    geom = MultiLineString(
        [
            [id_to_centroid[point_id] for point_id in path]
            for path in nw["paths"].values()
        ]
    )
    return geom.length


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

for pop in gens.values():
    for nw_id, nw in pop.items():
        n_edges_dict[nw_id] = bicycle_network.graph.network_len(nw)


# %%
xmin = 0
xmax = 800
ymin = 0
ymax = 100

max_edges = 751
max_demand = 100

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
for pop_id, pop in gens.items():
    ax.scatter(
        [n_edges_dict[nw_id] for nw_id in pop],  # number of edges in each network
        [nw["demand_covered"] for nw in pop.values()],
        label=f"Generation {pop_id}",
        s=10,
    )
ax.hlines(
    y=max_demand,
    xmin=xmin,
    xmax=max_edges,
    color="blue",
    linestyle="dashed",
    label="Full demand coverage",
)
ax.vlines(
    x=max_edges,
    ymin=ymin,
    ymax=ymax,
    color="orange",
    linestyle="dashed",
    label="Full triangulation",
)
ax.set_xlabel("Number of edges")
ax.set_ylabel("Demand coverage in %")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.legend()
