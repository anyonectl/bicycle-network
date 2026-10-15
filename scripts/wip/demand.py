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

# %% [markdown]
# # Assign random demand to seed point pairs
#
# Steps:
# 1. Load seed points into list of size N (no. of seed points)
# 2. Create pair combinations in a new list of size N choose 2 (pair member can be referred to by their index in list 1)
# 3. Function that assigns random demand to each pair

# %%
from typing import List, Tuple


# %% [markdown]
# ## Compute pairs
#
# We find all possible pairs of seed points with 2-combination, $n$ choose $r$. We assume undirectedness (and just compute total demand for each edge, including travel both ways), so we need unordered pairs withon repetition (because we don't allow self-loops):
#
# $$
# {N \choose r} = \frac{N!}{r!(N-r)!}
# $$
#
# $$
# {N \choose 2} = \frac{N!}{2!(N-2)!}
# $$
#

# %%
def pairs(n: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
    return pairs


# %%
S: int = 9 # number of seed points
test_pairs = pairs(S) # n=9 should give 36 unique pairs
print(len(test_pairs))
test_pairs

# %% [markdown]
# ## Generate random demand

# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
# import secrets
# secrets.randbits(64)

# %%
def generate_demand(mean: float, std: float, n: int) -> np.ndarray:
    seed: int = 8920299146682435938
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    return rng.normal(mean, std, size=n)

# Code from NumPy docs
def plot_demand_distribution(mean: float, std: float, demand: np.ndarray):
    count, bins, _ = plt.hist(demand, 30, density=True)
    plt.plot(bins, 1/(std * np.sqrt(2 * np.pi)) *
                   np.exp( - (bins - mean)**2 / (2 * std**2) ),
             linewidth=2, color='r')
    plt.show()


# %%
P: int = len(test_pairs) # number of seed point pairs
mean: float = 0.5
std: float = 0.1
test_demand = generate_demand(mean, std, P)
print(type(test_demand))
print(len(test_demand))
print(len(test_demand))
test_demand

# %%
plot_demand_distribution(mean, std, test_demand)


# %% [markdown]
# ## Map demand to seed point pairs

# %%
def total_demand(demand: np.ndarray) -> float:
    return np.sum(demand)    

def normalised_demand(demand: np.ndarray) -> np.ndarray:
    demand_normalised: np.ndarray = np.empty_like(demand)
    total: float = total_demand(demand)
    for i in range(demand.size):
        demand_normalised[i] = (demand[i] * 100) / total
    return demand_normalised

# get the absolute demand for the ith seed point pair
def get_pair_demand_abs(demand: np.ndarray, i: int) -> float:
    return demand[i]

# get the demand percentage for the ith seed point pair
def get_pair_demand_percent(demand: np.ndarray, i: int) -> float:
    return normalised_demand(demand)[i]

# get the total demand that seed point v is involved in
# either 1. store in a map - lookup is cheap, storage is not, or 2. compute it as needed
# here's a naive, expensive version:
def get_point_demand_abs(v: int, pairs: List[Tuple[int,int]], demand: np.ndarray) -> float:
    point_demand: float = 0.0
    for i in range(len(pairs)):
        fst: int = pairs[i][0]
        snd: int = pairs[i][1]
        if v == fst or v == snd:
            point_demand += demand[i]
    return point_demand

# get the total demand share that seed point v is involved in
def get_point_demand_percent(v: int, pairs: List[Tuple[int,int]], demand: np.ndarray) -> float:
    point_demand: float = 0.0
    for i in range(len(pairs)):
        fst: int = pairs[i][0]
        snd: int = pairs[i][1]
        if v == fst or v == snd:
            point_demand += demand[i]
    return (point_demand * 100) / total_demand(demand)



# %%
print(f"Total demand: {total_demand(test_demand)}")
test_demand_percent = normalised_demand(test_demand)
print(type(test_demand_percent))
print(len(test_demand_percent))
test_demand_percent

# %%
plot_demand_distribution(mean, std, test_demand_percent)

# %%
for i in range(P):
    pair = test_pairs[i]
    abs_demand = get_pair_demand_abs(test_demand, i)
    share_demand = get_pair_demand_percent(test_demand, i)
    print(f"Pair: {pair} \t Abs demand: {abs_demand:.2f} \t Share of demand: {share_demand:.2f}")

# %%
for seed_point in range(S):
    abs_demand = get_point_demand_abs(seed_point, test_pairs, test_demand)
    share_demand = get_point_demand_percent(seed_point, test_pairs, test_demand)
    print(f"Seed point: {seed_point} \t Abs demand: {abs_demand:.2f} \t Share of total demand: {share_demand:.2f}")

# %% [markdown]
# ## Sort pairs after demand

# %%
