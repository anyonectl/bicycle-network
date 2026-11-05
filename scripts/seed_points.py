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
# # Hex grid, demand matrix and Initial Route Set Generation (IRSG)

# %% [markdown]
# ## Setup

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from bicycle_network import hex, demand
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ### Variables

# %%
SEED: int = 3581054352
RNG: np.random.Generator = np.random.default_rng(seed=SEED)
ROOT = "../"
PLOT_PATH = ROOT + "results/plots/"
DATA_PATH = ROOT + "results/network-data/"

# %% [markdown]
# ## Grid and seed points

# %%
hex_radius = 4
grid = hex.make_test_grid(hex_radius)
grid["centroid"] = grid["geometry"].centroid

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
grid.boundary.plot(ax=axes[0])
axes[0].set_title("Hex grid")
grid["centroid"].plot(ax=axes[1])
axes[1].set_title("Centroids")
plt.tight_layout()
plt.show()

# %%
ax = grid.boundary.plot(figsize=(10, 10))
for _, row in grid.iterrows():
    centroid = row["centroid"]
    ax.annotate(
        row["cell_id"],
        (centroid.x, centroid.y),
        ha="center",
        va="center",
        fontsize=8,
        color="grey",
    )
ax.set_axis_off()
ax.set_title("Grid with IDs")
plt.show()

# %% [markdown]
# ## Demand

# %% [markdown]
# ### Create all seed point pairs

# %%
test_pairs = demand.od_pairs(grid)
test_pairs.head()
# TODO: put these tests into a tests/test_demand.py file
# print(f"no. of seed points: {grid.shape[0]}")
# print(f"no. of pairs is actually: {test_pairs.shape[0]}")
# print(f"no. of pairs should be: {comb(grid.shape[0], 2)}")


# %% [markdown]
# ### Generate random demand and create demand matrix

# %%
n_pairs = test_pairs.shape[0]  # number of seed point pairs
test_demand = demand.generate_weird_demand(n_pairs)
demand_matrix = demand.create_matrix(test_pairs, test_demand)
print(demand_matrix.info())
demand_matrix.head()

# %%
fig, (ax_hist, ax_map) = plt.subplots(
    1,
    2,
    figsize=(16, 8),
)

count, bins, _ = ax_hist.hist(test_demand, 15, density=True)
ax_hist.plot(bins, np.ones_like(bins), linewidth=2, color="r")
ax_hist.set_title("Demand distribution")

grid.boundary.plot(ax=ax_map, color="black", zorder=0, lw=0.5)
demand_matrix.plot(
    ax=ax_map,
    lw=demand_matrix.demand_share * 2000,
    alpha=0.2,
)
ax_map.set_axis_off()
ax_map.set_title("Pairwise demand")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Aggregate demand per seed point

# %%
seed_points = grid.copy()
seed_points.columns = ["id", "geometry", "centroid"]
seed_points["total_demand"] = seed_points.apply(
    lambda x: sum(
        demand_matrix[
            (demand_matrix["u"] == x["id"]) | (demand_matrix["v"] == x["id"])
        ]["demand"]
    ),
    axis=1,
)
print(seed_points.info())
seed_points.head()

# %%
# TODO: Ask Anastassia: Is it correct to calculate it like this? Or am I counting stuff twice?
seed_points["demand_share"] = (
    seed_points["total_demand"] / seed_points["total_demand"].sum()
)
seed_points.head()

# %%
seed_points[["total_demand", "demand_share"]].describe()

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
seed_points.plot(ax=axes[0], column="total_demand", cmap="Blues", legend=True)
axes[0].set_axis_off()
axes[0].set_title("Total demand")
seed_points.plot(ax=axes[1], column="demand_share", cmap="Blues", legend=True)
axes[1].set_axis_off()
axes[1].set_title("Demand share")
fig.suptitle("Demand distribution per hexagon", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Initial Route Set Generation (IRSG)

# %% [markdown]
# ### Create Vicinity Node Set (VNS) for each seed point

# %%
matches = seed_points[["id", "geometry"]].sjoin(
    seed_points[["id", "geometry"]],
    how="left",
    predicate="touches",
    lsuffix="left",
    rsuffix="right",
)

# %%
seed_points = seed_points.copy()
seed_points["neighbours"] = matches.groupby(level=0)["id_right"].agg(
    lambda ids: ids.dropna().tolist()
)

# %%
seed_points.head()

# %%
seed_points = seed_points.copy()
seed_points["neighbours"] = seed_points.apply(
    lambda row: hex.vns_list_to_bearing_dict(row["id"], row["neighbours"]), axis=1
)

# %%
seed_points.head()

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
seed_points.boundary.plot(ax=ax, color="grey")
# seed_points["centroid"].plot(ax=ax, color="black", markersize=1)
selected_points = ["0,0", "-8,-16", "8,14", "4,-16", "-8,6"]
for i in selected_points:
    seed_points[seed_points["id"] == i].plot(ax=ax, color="orange", alpha=0.5)
    neighbour_dict = seed_points.loc[seed_points["id"] == i, "neighbours"].values[0]
    neighbour_df = seed_points[seed_points["id"].isin(neighbour_dict.values())]  # type: ignore
    neighbour_df.plot(ax=ax, color="blue", alpha=0.5)
    reverse_neighbour_dict = dict((v, k) for k, v in neighbour_dict.items())  # type: ignore
    for _, row in neighbour_df.iterrows():
        centroid = row["centroid"]
        neighbour_id = row["id"]
        ax.annotate(
            reverse_neighbour_dict[neighbour_id],
            (centroid.x, centroid.y),
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )
for _, row in seed_points[seed_points["id"].isin(selected_points)].iterrows():
    centroid = row["centroid"]
    ax.annotate(
        row["id"],
        (centroid.x, centroid.y),
        ha="center",
        va="center",
        fontsize=8,
        color="black",
    )
ax.set_axis_off()
ax.set_title("Vicinity Node Set (VNS)")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Save dataframes

# %%
print(demand_matrix.info())
demand_matrix.head(5)

# %%
demand_matrix = demand_matrix[
    ["pair", "u", "v", "demand", "demand_share", "geometry"]
].copy()
demand_matrix.to_file(f"{DATA_PATH}demand_matrix.geojson", driver="GeoJSON")

# %%
print(grid.info())
grid.head(5)

# %%
grid = grid[["cell_id", "geometry"]].copy()
grid.to_file(f"{DATA_PATH}grid.geojson", driver="GeoJSON")

# %%
print(seed_points.info())
seed_points.head(5)

# %%
seed_points = (
    seed_points[["id", "total_demand", "demand_share", "neighbours", "centroid"]]
    .set_geometry("centroid")
    .copy()
)
seed_points.to_file(f"{DATA_PATH}seed_points.geojson", driver="GeoJSON")
