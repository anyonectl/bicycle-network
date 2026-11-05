from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps as mplcm
from geopandas import GeoDataFrame as gdf


def plot_population(
    population: gdf,
    seed_points: gdf,
    grid: gdf,
    demand_type: str,
    save=False,
    output_path="../results/plots/",
    title="",
):
    n_pop = population["network_id"].nunique()
    n_cols = 4
    n_rows = ceil(n_pop / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.ravel()
    cmap = mplcm.get_cmap("tab10")

    for i in range(n_pop):
        ax = axes[i]
        network = population[population["network_id"].values == i]
        demand_coverage = network.at[0, "demand_coverage"]
        path_ids = network["path_id"].unique()
        colors = cmap(np.linspace(0, 1, len(path_ids)))
        seed_points.boundary.plot(ax=ax, color="grey", lw=1)
        grid["centroid"].plot(ax=ax, color="black", markersize=1)

        for path_id in path_ids:
            network[network["path_id"] == path_id].iloc[[0]].plot(
                ax=ax, lw=5, alpha=0.5, label=path_id, color=colors[path_id]
            )

        # ax.legend()
        ax.set_title(f"Network {i}, demand coverage: {demand_coverage:.2f} %")
        ax.set_axis_off()

    fig.suptitle("Generation 0", fontsize=16)
    plt.tight_layout()

    filename = "start_population"
    if save:
        plt.savefig(
            f"{output_path}{filename}_{demand_type}_{title}.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()
