from typing import List
from geopandas import GeoDataFrame as gdf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from shapely import LineString, Point


def create_matrix(pairs: gdf, demand: np.ndarray):
    demand_matrix = pd.concat([pairs, pd.Series(demand)], axis=1)
    demand_matrix.columns = ["pair", "u", "v", "geometry", "demand"]
    total_demand = float(demand_matrix["demand"].sum())
    demand_matrix["demand_share"] = demand_matrix["demand"] * 100 / total_demand
    return demand_matrix


def od_pairs(geo_df: gdf) -> gdf:
    seed_points = geo_df["cell_id"].values.tolist()
    geo: List[Point] = geo_df["centroid"].tolist()
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
    return gdf({"pair": ids, "u": origins, "v": dests, "geometry": lines})


def generate_uniform_demand(n: int) -> np.ndarray:
    seed: int = 8920299146682435931
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    return rng.uniform(low=0.0, high=1.0, size=n)


def generate_weird_demand(n: int) -> np.ndarray:
    return make_demand_weird(generate_uniform_demand(n))


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
def plot_demand_distribution(demand: np.ndarray) -> None:
    count, bins, _ = plt.hist(demand, 15, density=True)
    plt.plot(bins, np.ones_like(bins), linewidth=2, color="r")
    plt.show()
    return None


# Code from Anastassia
def plot_pairwise_demand(demand_matrix: gdf, grid: gdf) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    grid.boundary.plot(ax=ax, color="black", zorder=0, lw=0.5)
    demand_matrix.plot(ax=ax, lw=demand_matrix.demand_share * 20, alpha=0.2)
    ax.set_axis_off()
    ax.set_title("Seed point pairs")
    return None
