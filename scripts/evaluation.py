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
import bicycle_network.graph
import bicycle_network.plot
import bicycle_network.genetic_algorithm as ga
import geopandas as gpd

# %%
ROOT = "../results/"
PLOT_PATH = ROOT + "plots/"
DATA_PATH = ROOT + "network-data/"
GEN_FILE = f"{DATA_PATH}generations_allmutated_261123.json"
GRID_FILE = f"{DATA_PATH}grid.geojson"
SEED_POINTS_FILE = f"{DATA_PATH}seed_points.geojson"

# %% [markdown]
# ### Load data

# %%
grid = gpd.read_file(GRID_FILE)
seed_points = gpd.read_file(SEED_POINTS_FILE)

# %%
with open(GEN_FILE, "r") as fp:
    data = json.load(fp)
gens = ga.jsonable_to_gens(data)

# %%
print(type(gens))
print((gens))
print(gens[0])

# %% [markdown]
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
for pop in gens.values():
    for nw_id, nw in pop.items():
        n_edges_dict[nw_id] = nw["graph"].number_of_edges()

# %% [markdown]
# ## Plot
# Scatter plot with x-axis the total length of a network,
# y-axis the percentage of demand met

# %%
bicycle_network.plot.scatter(gens, n_edges_dict, output_path=PLOT_PATH)

# %%
for pop_id, pop in gens.items():
    bicycle_network.plot.plot_population_nx(
        pop,
        seed_points,
        grid,
        demand_type="absolute",
        save=False,
        output_path=PLOT_PATH,
        filename="mutation",
        gen=pop_id,
    )
