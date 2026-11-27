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
from bicycle_network import plot

# %%
SEED: int = 3581054352
RNG: np.random.Generator = np.random.default_rng(seed=SEED)
ROOT = "../"
PLOT_PATH = ROOT + "results/plots/"
DATA_PATH = ROOT + "results/network-data/"
INITIAL_POP_FILE = f"{DATA_PATH}start_population.json"
DEMAND_MATRIX_FILE = f"{DATA_PATH}demand_matrix.geojson"
GRID_FILE = f"{DATA_PATH}grid.geojson"
SEED_POINTS_FILE = f"{DATA_PATH}seed_points.geojson"

# %% [markdown]
# ### Load data

# %%
demand_matrix = gpd.read_file(DEMAND_MATRIX_FILE)
grid = gpd.read_file(GRID_FILE)
seed_points = gpd.read_file(SEED_POINTS_FILE)


# %% [markdown]
# #### Initial population

# %%
with open(INITIAL_POP_FILE, "r") as fp:
    data = json.load(fp)
start_pop = ga.convert_start_json_to_pop(data)

# %% [markdown]
# ## Run GA

# %%
iterations = 100
tournament_size = 3
elitism = True
n_elites = 2
mutation_probability = 0.3
demand_weight = 0.8
directness_weight = 0.2
max_edges = int(seed_points["neighbours"].apply(len).sum() // 2)
budget = max_edges // 5

generations = ga.run(
    start_pop,
    iterations,
    budget,
    tournament_size,
    elitism,
    n_elites,
    mutation_probability,
    seed_points,
    grid,
    demand_matrix,
    SEED,
)

print(f"Iterations: {iterations}")
print(f"Population size: {len(start_pop)}")
print(f"Budget: {budget}")
print(f"Elitism: {elitism}")
print(f"Number of elites: {n_elites}")
print(f"Tournament size: {tournament_size}")
print(f"Mutation probability: {mutation_probability}")
print(f"Max edges: {max_edges}")

# %% [markdown]
# ## Save generations as JSON

# %%
filename = f"generations_{plot.current_time()}"
with open(f"{DATA_PATH}{filename}.json", "w") as json_file:
    json.dump(ga.generations_to_jsonable(generations), json_file, indent=4)

# %% [markdown]
# # Plots

# %% [markdown]
# ### Scatter plot
# Scatter plot with x-axis the total length of a network,
# y-axis the percentage of demand met

# %%
plot.scatter(generations, budget, save=True, output_path=PLOT_PATH)

# %% [markdown]
# ### Network plots

# %%
gens_to_plot = [0, iterations // 4, iterations // 2, (iterations * 3) // 4, iterations]
for pop_id in gens_to_plot:
    plot.population(
        generations[pop_id],
        seed_points,
        grid,
        save=False,
        output_path=PLOT_PATH,
        filename="mutate_and_crossover",
        gen=pop_id,
    )

# %% [markdown]
# ### Violin plot
# One violin per generation to show demand coverage distribution for all networks in the generation.
# Should include at least 5 gens, maybe all if there aren't that many
