from math import ceil
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps as mplcm
from geopandas import GeoDataFrame as gdf
from shapely import LineString


def plot_population_gdf(
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
        grid.boundary.plot(ax=ax, color="grey", lw=1)
        seed_points["geometry"].plot(ax=ax, color="black", markersize=1)

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


def plot_population_dict(
    pop,
    seed_points: gdf,
    grid: gdf,
    demand_type: str,
    save=False,
    output_path="../results/plots/",
    filename="pop",
    title="",
    gen: int = 0,
):
    n_pop = len(pop)
    n_cols = 4
    n_rows = ceil(n_pop / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).ravel() if n_pop > 1 else np.array([axes])
    cmap = mplcm.get_cmap("tab10")
    id_to_centroid = seed_points.set_index("id")["geometry"].to_dict()
    xmin, ymin, xmax, ymax = grid.total_bounds

    for i, (nw_id, nw) in enumerate(pop.items()):
        ax = axes[i]
        demand_coverage = nw["demand_covered"]
        path_ids = list(nw["paths"].keys())
        colors = cmap(np.linspace(0, 1, len(path_ids)))
        grid.boundary.plot(ax=ax, color="grey", lw=1)
        seed_points["geometry"].plot(ax=ax, color="black", markersize=1)

        for j, (path_id, path) in enumerate(nw["paths"].items()):
            line = LineString([id_to_centroid[point_id] for point_id in path])
            xs, ys = line.xy
            ax.plot(xs, ys, color=colors[j], lw=5, alpha=0.5)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        # ax.legend()
        ax.set_title(f"Network {nw_id}, demand coverage: {demand_coverage:.2f} %")
        ax.set_axis_off()

    for ax in axes[n_pop:]:
        ax.set_axis_off()

    fig.suptitle(f"Gen. {gen}", fontsize=16)
    plt.tight_layout()

    if save:
        gen_str = f"gen{gen}"
        title_str = f"{title}_"
        plt.savefig(
            f"{output_path}{filename}_{title_str}{gen_str}.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()
