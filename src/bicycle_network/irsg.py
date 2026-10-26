import numpy as np
import geopandas as gpd
import pandas as pd
from shapely import LineString, Point
from bicycle_network.hex import direction_bias


# TODO: bias terms for each node in the VNS?
# TODO: needs some way to stop if no node in VNS is possible. boolean in irs dict?
def select_next_node(
    path_id: int,
    irs: dict,
    seed_points: gpd.GeoDataFrame,
    rng: np.random.Generator,
    demand_type: str,
):
    current_node = irs[path_id]["nodes"][-1]
    previous_node = (
        irs[path_id]["nodes"][-2] if len(irs[path_id]["nodes"]) > 1 else current_node
    )

    # Create Vicinity Node Set (VNS) for the current node, sorted by demand
    neighbour_dict = seed_points.loc[
        seed_points["id"] == current_node, ["neighbours"]
    ].values.tolist()[0][0]
    vns = (
        seed_points.loc[seed_points["id"].isin(neighbour_dict.values())]
        .sort_values(by=["total_demand"], ascending=False)
        .copy()
    )

    # Set direction bias for each node in VNS
    # TODO: Fix type warnings
    current_point: Point = seed_points.loc[
        seed_points["id"] == current_node, "centroid"
    ].iat[
        0
    ]  # type: ignore
    previous_point: Point = seed_points.loc[
        seed_points["id"] == previous_node, "centroid"
    ].iat[
        0
    ]  # type: ignore
    vns["direction_bias"] = vns["centroid"].apply(
        lambda x: direction_bias(previous_point, current_point, x)
    )

    # Calculate probabilities for each node in the VNS
    match demand_type:
        case "absolute":
            demand_series = vns["total_demand"]
        case "share":
            demand_series = vns["demand_share"]
        case "minmax":
            demand_series = vns["demand_minmax"]
        case _:
            raise ValueError(
                f"demand_type must be 'absolute', 'share' or 'minmax', was: {demand_type}"
            )

    vns["combined"] = vns["direction_bias"] * demand_series
    prob_sum = float(vns["combined"].sum())
    vns["probability"] = vns["combined"] / prob_sum

    # Select next node and add to IRS
    tries = vns.shape[0]
    next_node = ""
    for _ in range(tries):
        next_node = rng.choice(
            a=vns["id"], p=normalise_probabilities(vns["probability"])
        )
        if next_node not in irs[path_id]["nodes"]:
            irs[path_id]["nodes"].append(next_node)
            break
        else:
            vns = vns[(vns["id"] != next_node)]
    finished = vns.shape[0] == 0  # stop if no node in VNS is possible
    return (irs, finished)


def generate_network(
    k: int,
    max_paths: int,
    max_nodes: int,
    seed_points: gpd.GeoDataFrame,
    rng: np.random.Generator,
    demand_type: str,
):
    """
    Form Initial Node Set (INS) from the first k nodes. k must be larger than max_paths
    Form Initial Route Set (IRS) by selecting a first node from the INS for each path,
    and then adding nodes from the Vicinity Node Set (VNS) of the previous node until max_nodes is reached.
    """
    irs = {}
    ins = (
        seed_points.copy()
        .sort_values("total_demand", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )

    # Select first nodes
    for i in range(max_paths):
        # Calculate probabilities of selection a node j as first node
        demand_sum = float(ins["total_demand"].sum())
        ins["probability"] = ins["total_demand"] / demand_sum

        # Select first node
        first_node = rng.choice(
            a=ins["id"], p=normalise_probabilities(ins["probability"])
        )

        # Add to dict
        irs[i] = {"nodes": [first_node]}

        # Remove selected node from INS
        ins = ins[(ins["id"] != first_node)]
        # TODO: if ins falls below some threshold, put more in it and recalculate probabilities (see chakroborty paper)

        # Build paths
        for _ in range(max_nodes - 1):  # -1 because we already added first_node
            tmp_irs, finished = select_next_node(i, irs, seed_points, rng, demand_type)
            if finished:  # stop if no node in VNS is possible
                break
            irs = tmp_irs

    return irs


def generate_population(
    n: int,
    k: int,
    max_paths: int,
    max_nodes: int,
    seed_points: gpd.GeoDataFrame,
    rng: np.random.Generator,
    demand_type: str = "share",
) -> gpd.GeoDataFrame:
    """
    Create population of n networks (route sets).
    Each network will contain the starting nodes selected from the k nodes with highest demand.
    """
    pop_list = []
    for i in range(n):
        gdf = create_gdf(
            generate_network(k, max_paths, max_nodes, seed_points, rng, demand_type),
            seed_points,
        )
        gdf["network_id"] = i
        # TODO: save_plot(gdf)
        pop_list.append(gdf)

    pop_gdf = pd.concat(pop_list)
    # TODO: save as geojson here?
    return pop_gdf


def normalise_probabilities(df: pd.Series) -> np.ndarray:
    p = df.to_numpy(dtype=np.float64)
    return p / np.sum(p, dtype=np.float64)


def create_gdf(irs: dict, seed_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
