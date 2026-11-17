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

from shapely import LineString, MultiLineString, intersection
from bicycle_network.genetic_algorithm import Network
import bicycle_network.genetic_algorithm as ga
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
# Uses multilinestring, counts overlapping lines
def network_len(nw: Network) -> float:
    geom = MultiLineString(
        [
            [id_to_centroid[point_id] for point_id in path]
            for path in nw["paths"].values()
        ]
    )
    return geom.length

# %%
# This won't work - we need to check all pairs of paths
# Alternative idea: Get the total length first, as above.
# Then calculate what needs to be subtracted (find out how to find
# the overlap)
# generate all pairs of pairs in a network, then check intersection 
# between all pairs
# seems easier to create a networkx graph with a distance attribute, no?
def network_len2(nw: Network) -> float:
    total_length = 0.0
    prev = LineString([id_to_centroid[point_id] for point_id in nw["paths"][0]])

    while i < n_paths:
        while j < n_paths:

        j = j + 1
    for i, path in enumerate(nw["paths"].values()):

        current = LineString([id_to_centroid[point_id] for point_id in path])
        overlap = intersection(current, prev)
        total_length = total_length + current.length - overlap.length
        prev = current


        if i == n_paths:
            break


            current = nw["paths"][i+1]

    
    id, path in nw["paths"].items():
        prev = current
        current = LineString([id_to_centroid[point_id] for point_id in nw["paths"][0]])
        length = current.length
        prev = current
        if id == 0:



def 



# %%
for i in range(len(gens)):
    for j in range(len(gens[i])):
        print(network_len(gens[i][j]) == network_len2(gens[i][j]))

# %%
# TODO Placeholder for scatter plot
