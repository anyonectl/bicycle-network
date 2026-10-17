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
from bicycle_network import hex
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
def od_pairs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    seed_points = gdf["cell_id"].values.tolist()
    geo: List[Point] = gdf["centroid"].tolist()
    n = len(seed_points)
    ids = []
    origins = []
    dests = []
    lines = []
    for i in range(n):
        for j in range(i + 1, n):
            ids.append((seed_points[i], seed_points[j]))
            origins.append(seed_points[i])
            dests.append(seed_points[j])
            lines.append(LineString([geo[i], geo[j]]))
    return gpd.GeoDataFrame({"pair": ids, "u": origins, "v": dests, "geometry": lines})


# %%
test_pairs = od_pairs(grid)
print(f"no. of seed points: {grid.shape[0]}")
print(f"no. of pairs is actually: {test_pairs.shape[0]}")
print(f"no. of pairs should be: {comb(grid.shape[0], 2)}")
test_pairs.head()


# %% [markdown]
# ### Generate random demand


# %%
def generate_uniform_demand(n: int) -> np.ndarray:
    seed: int = 8920299146682435931
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    return rng.uniform(low=0.0, high=1.0, size=n)


def make_demand_weird(demand: np.ndarray) -> np.ndarray:
    seed: int = 8920299146682435931
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    n = len(demand)
    indices = [x for x in range(n)]
    doubles = rng.choice(a=indices, size=int(n / 4), replace=False)
    halves = rng.choice(a=indices, size=int(n / 4), replace=False)
    for i in doubles:
        demand[i] = demand[i] * 2
    for i in halves:
        demand[i] = demand[i] / 1.5
    return demand


# Code from NumPy docs
def plot_demand_distribution(demand: np.ndarray):
    count, bins, _ = plt.hist(demand, 15, density=True)
    plt.plot(bins, np.ones_like(bins), linewidth=2, color="r")
    plt.show()


# %%
n_pairs = test_pairs.shape[0]  # number of seed point pairs
test_demand = make_demand_weird(generate_uniform_demand(n_pairs))
plot_demand_distribution(test_demand)

# %% [markdown]
# ### Create demand matrix

# %%
demand_matrix = pd.concat([test_pairs, pd.Series(test_demand)], axis=1)
demand_matrix.columns = ["pair", "u", "v", "geometry", "demand"]
total_demand = float(demand_matrix["demand"].sum())
demand_matrix["demand_share"] = demand_matrix["demand"] * 100 / total_demand
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
seed_points["vns"] = seed_points["id"].apply(lambda s: vns(hex.numeric_id(s)))


# %% [markdown]
# ### Functions to create single network


# %%
def normalise_probabilities(df):
    p = df.to_numpy(dtype=np.float64)
    return p / np.sum(p, dtype=np.float64)


# TODO: bias terms for each node in the VNS?
# TODO: needs some way to stop if no node in VNS is possible. boolean in irs dict?
def next_node(p: int, previous_node: str, irs: dict):
    # Calculate probabilities for each node in the VNS
    filter = seed_points.loc[
        seed_points["id"] == previous_node, ["vns"]
    ].values.tolist()[0][0]
    vns = seed_points.loc[seed_points["id"].isin(filter)].sort_values(
        by=["total_demand"], ascending=False
    )
    activity_sum = float(vns["total_demand"].sum())
    vns["probability"] = vns["total_demand"] / activity_sum

    # Select next node and add to dict
    tries = vns.shape[0]
    next_node = ""
    for _ in range(tries):
        next_node = RNG.choice(
            a=vns["id"], p=normalise_probabilities(vns["probability"])
        )
        if next_node not in irs[p]["nodes"]:
            irs[p]["nodes"].append(next_node)
            break
        else:
            vns = vns[(vns["id"] != next_node)]
    finished = vns.shape[0] == 0  # stop if no node in VNS is possible
    return (next_node, irs, finished)


def generate_route(k: int):
    """
    Form Initial Node Set (INS) from the first k nodes. k must be larger than MAX_PATHS
    """
    irs = {}
    ins = (
        seed_points.copy()
        .sort_values("total_demand", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )

    # Select first nodes
    for i in range(MAX_PATHS):
        # Calculate probabilities of selection a node j as first node
        activity_sum = float(ins["total_demand"].sum())
        ins["probability"] = ins["total_demand"] / activity_sum

        # Select first node
        first_node = RNG.choice(
            a=ins["id"], p=normalise_probabilities(ins["probability"])
        )

        # Add to dict
        irs[i] = {"nodes": [first_node]}

        # Remove selected node from INS
        ins = ins[(ins["id"] != first_node)]

        # Build paths
        previous_node = first_node
        for _ in range(MAX_NODES - 1):  # -1 because we already added first_node
            tmp_next, tmp_irs, finished = next_node(i, previous_node, irs)
            if finished:  # stop if no node in VNS is possible
                break
            previous_node = tmp_next
            irs = tmp_irs

    return irs


def create_gdf(irs: dict) -> gpd.GeoDataFrame:
    # Create dict for network
    d = {
        "path_id": [path_id for path_id in irs],
        "start_id": [path.get("nodes")[:][0] for _, path in irs.items()],
        "end_id": [path.get("nodes")[:][-1] for _, path in irs.items()],
        "path": [path.get("nodes") for _, path in irs.items()],
        # 'demand_covered': [],
        # 'geometry': []
    }

    # Create geometries
    id_to_centroid = seed_points.set_index("id")["centroid"].to_dict()
    geom = [
        LineString([id_to_centroid[point_id] for point_id in path])
        for path in d["path"]
    ]

    # Create dataframe
    network_gdf = gpd.GeoDataFrame(d, geometry=geom, crs=None)

    return network_gdf


def generate_population(n: int, k: int) -> gpd.GeoDataFrame:
    """
    Create population of n networks (route sets).
    Each network will contain the starting nodes selected from the k nodes with highest demand.
    """
    pop_list = []
    for i in range(n):
        gdf = create_gdf(generate_route(k))
        gdf["network_id"] = i
        # save_plot(gdf)
        pop_list.append(gdf)

    pop_gdf = pd.concat(pop_list)
    # save as geojson here
    return pop_gdf


# %% [markdown]
# ### Run IRSG

# %%
n_start_nodes = int(
    len(seed_points) / 2
)  # how many start nodes do we initialise our routes with
start_pop = generate_population(N_POP, n_start_nodes)

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
