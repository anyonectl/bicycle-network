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
GEN_FILE = f"{DATA_PATH}generations_261124.json"
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

# %% [markdown]
# ### Scatter plot
# Scatter plot with x-axis the total length of a network,
# y-axis the percentage of demand met

# %%
bicycle_network.plot.scatter(gens, n_edges_dict, save=True, output_path=PLOT_PATH)

# %%
for pop_id in [0, 10, 19]:
    bicycle_network.plot.population(
        gens[pop_id],
        seed_points,
        grid,
        save=True,
        output_path=PLOT_PATH,
        filename="mutate_and_crossover",
        gen=pop_id,
    )

# %% [markdown]
# ### Violin plot
# One violin per generation to show demand coverage distribution for all networks in the generation.
# Should include at least 5 gens, maybe all if there aren't that many
