from math import ceil
from matplotlib.collections import LineCollection
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps as mplcm
from matplotlib.colors import Normalize
from geopandas import GeoDataFrame as gdf
from shapely import LineString
from datetime import datetime as dt
import networkx as nx


def current_time() -> str:
    year = str(dt.now().year)[2:]
    month = dt.now().month
    date = dt.now().day
    hour = dt.now().hour
    min = dt.now().minute
    return f"{year}{month}{date}{hour}{min}"


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
            coords = [id_to_centroid[point_id] for point_id in path]
            if len(coords) < 2:  # TODO Need to fix this in the algorithm somehow...
                print(f"Skipping path {path_id}: {coords}")
                continue
            line = LineString(coords)
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
        plt.savefig(
            f"{output_path}{filename}_{gen_str}_{current_time()}.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def _draw_network(ax, graph, seed_points: gdf, grid: gdf) -> None:
    """
    Draws one network on the given axis, one colour per connected component.
    """
    xmin, ymin, xmax, ymax = grid.total_bounds
    grid.boundary.plot(ax=ax, color="grey", lw=1)
    seed_points["geometry"].plot(ax=ax, color="black", markersize=1)
    id_to_centroid = seed_points.set_index("id")["geometry"].to_dict()
    components = list(nx.connected_components(graph))
    colors = mplcm["tab10"](np.linspace(0, 1, max(len(components), 1)))
    for color, component in zip(colors, components):
        segments = [
            [
                (id_to_centroid[u].x, id_to_centroid[u].y),
                (id_to_centroid[v].x, id_to_centroid[v].y),
            ]
            for u, v in graph.subgraph(component).edges
        ]
        if segments:
            ax.add_collection(
                LineCollection(segments, colors=[color], linewidths=5, alpha=0.5)
            )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def network(
    nw,
    seed_points: gdf,
    grid: gdf,
    save=False,
    output_path="../results/plots/",
    filename="network",
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    _draw_network(ax, nw["graph"], seed_points, grid)
    ax.set_title(
        f"Network: {nw['demand_covered']:.1f} % demand, "
        f"{nw['graph'].number_of_edges()} edges"
    )
    plt.tight_layout()
    if save:
        plt.savefig(
            f"{output_path}{filename}_{current_time()}.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()
    plt.close()


def population(
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
    axes = np.atleast_1d(axes).ravel()
    for ax, (nw_id, nw) in zip(axes, pop.items()):
        _draw_network(ax, nw["graph"], seed_points, grid)
        ax.set_title(
            f"Network {nw_id}: {nw['demand_covered']:.1f} % demand, "
            f"{nw['graph'].number_of_edges()} edges"
        )
    for ax in axes[n_pop:]:
        ax.set_axis_off()
    fig.suptitle(f"Gen. {gen}", fontsize=16)
    plt.tight_layout()
    if save:
        gen_str = f"gen{gen}"
        plt.savefig(
            f"{output_path}{filename}_{gen_str}_{current_time()}.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()
    plt.close()


def scatter(gens, budget=None, save=False, output_path="../results/plots/"):
    cmap = mplcm["viridis"]
    pop_ids = sorted(gens)
    norm = Normalize(vmin=pop_ids[0], vmax=pop_ids[-1])

    fig, ax = plt.subplots(figsize=(9, 7))
    for pop in pop_ids:
        networks = list(gens[pop].values())
        n_edges = [nw["graph"].number_of_edges() for nw in networks]
        coverage = [nw["demand_covered"] for nw in networks]
        ax.scatter(
            x=n_edges,
            y=coverage,
            color=cmap(norm(pop)),
            alpha=0.6,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

    if budget is not None:
        ax.axvline(
            budget,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Budget ({budget} edges)",
        )
        ax.legend(frameon=False, loc="best")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02).set_label("Generation")

    # xmin = 125
    # xmax = 140
    # ymin = 10
    # ymax = 30
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Number of edges")
    ax.set_ylabel("Demand coverage (%)")
    ax.set_title("Network performance by generation")
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(
            f"{output_path}scatter_{current_time()}.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()
    plt.close(fig)
