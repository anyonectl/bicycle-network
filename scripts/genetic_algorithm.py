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

# import networkx as nx

# %%
N_ITE: int = 100  # iterations
N_POP: int = 20
MAX_PATHS: int = 10
MAX_NODES: int = 15
SEED: int = 3581054352
RNG: np.random.Generator = np.random.default_rng(seed=SEED)
ROOT = "../"
PLOT_PATH = ROOT + "results/plots/"
DATA_PATH = ROOT + "results/network-data/"
EXT = ".json"
INITIAL_POP_FILE = f"{DATA_PATH}start_pop_abs_connected{EXT}"

# %% [markdown]
# ## Get initial population

# %%
with open(INITIAL_POP_FILE, "r") as fp:
    start_pop = json.load(fp)

# %%
print(type(start_pop))
print((start_pop))
print(start_pop["0"])

# %%
