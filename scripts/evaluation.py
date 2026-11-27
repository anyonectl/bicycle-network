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
from bicycle_network import plot
import bicycle_network.genetic_algorithm as ga
import geopandas as gpd

# %%
ROOT = "../results/"
PLOT_PATH = ROOT + "plots/"
DATA_PATH = ROOT + "network-data/"
GEN_FILE = f"{DATA_PATH}generations.json"
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
generations = ga.jsonable_to_gens(data)

# %%
iterations = len(generations)
demand_weight = 0.8
directness_weight = 0.2
max_edges = int(seed_points["neighbours"].apply(len).sum() // 2)
budget = max_edges // 5

# %% [markdown]
# ## Plots

# %% [markdown]
# ### Scatter plot
# Scatter plot with x-axis the total length of a network,
# y-axis the percentage of demand met

# %%
plot.scatter(
    generations,
    demand_weight,
    directness_weight,
    "fitness",
    budget,
    save=False,
    output_path=PLOT_PATH,
)
plot.scatter(
    generations,
    demand_weight,
    directness_weight,
    "demand_coverage",
    budget,
    save=False,
    output_path=PLOT_PATH,
)
plot.scatter(
    generations,
    demand_weight,
    directness_weight,
    "directness",
    budget,
    save=False,
    output_path=PLOT_PATH,
)

# %%
plot.progress(generations, budget, save=False, output_path=PLOT_PATH)

# %%
front_networks = plot.tradeoff(
    generations,
    demand_weight,
    directness_weight,
    budget,
    save=False,
    output_path=PLOT_PATH,
)

# %%
sorted_fronts = sorted(
    front_networks,
    key=lambda nw: nw[1]["demand_coverage"],
    reverse=True,
)
best_networks = dict(enumerate(item[1] for item in sorted_fronts))

# %%
top_20 = {i: best_networks[i] for i in range(20)}
plot.population(
    top_20,
    seed_points,
    grid,
    save=False,
    output_path=PLOT_PATH,
    filename="top20",
    gen=-99,
)

# %% [markdown]
# ## Save top 20 as JSON

# %%
filename = "top_networks"
with open(f"{DATA_PATH}{filename}.json", "w") as json_file:
    json.dump(ga.generations_to_jsonable(best_networks), json_file, indent=4)
