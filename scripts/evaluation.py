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
GEN_FILE = f"{DATA_PATH}ga_result.json"
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
result = ga.jsonable_to_ga_run(data)

# %%
iterations = result["iterations"]
budget = result["budget"]
elitism = result["elitism"]
n_elites = result["n_elites"]
tournament_size = result["tournament_size"]
mutation_probability = result["mutation_probability"]
demand_weight = result["demand_weight"]
directness_weight = result["directness_weight"]

print(f"Iterations: {iterations}")
print(f"Population size: {len(result["generations"][0])}")
print(f"Budget: {budget}")
print(f"Elitism: {elitism}")
print(f"Number of elites: {n_elites}")
print(f"Tournament size: {tournament_size}")
print(f"Mutation probability: {mutation_probability}")
print(f"Demand weight: {demand_weight}")
print(f"Directness weight: {demand_weight}")

# %% [markdown]
# ## Plots

# %% [markdown]
# ### Scatter plot
# Scatter plot with x-axis the total length of a network,
# y-axis the percentage of demand met

# %%
plot.scatter(
    result,
    "fitness",
    budget,
    save=False,
    output_path=PLOT_PATH,
)
plot.scatter(
    result,
    "demand_coverage",
    budget,
    save=False,
    output_path=PLOT_PATH,
)
plot.scatter(
    result,
    "directness",
    budget,
    save=False,
    output_path=PLOT_PATH,
)

# %%
plot.progress(result, budget, save=False, output_path=PLOT_PATH)

# %%
front_networks = plot.tradeoff(
    result,
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
