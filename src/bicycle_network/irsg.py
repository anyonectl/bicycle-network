from typing import Dict, List, Tuple

from tqdm import tqdm
import numpy as np
import geopandas as gpd
import pandas as pd
import networkx as nx
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
        case _:
            raise ValueError(
                f"demand_type must be 'absolute' or 'share', was: {demand_type}"
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
    pbar = tqdm(desc=f"Generate {n} connected networks. Loop iterations:", total=n * 15)
    i = 0
    while i < n:
        irs = generate_network(k, max_paths, max_nodes, seed_points, rng, demand_type)
        graph = create_graph(irs)
        if nx.is_connected(graph):
            gdf = create_gdf(irs, seed_points)
            gdf["network_id"] = i
            # TODO: save_plot(gdf)
            pop_list.append(gdf)
            i = i + 1
        pbar.update(1)
    pop_gdf = pd.concat(pop_list)
    # TODO: save as geojson here?
    return pop_gdf


def normalise_probabilities(df: pd.Series) -> np.ndarray:
    p = df.to_numpy(dtype=np.float64)
    sum = np.sum(p, dtype=np.float64)
    if sum == 0.0:
        raise ZeroDivisionError
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


def create_graph(irs: Dict) -> nx.Graph:
    graph = nx.Graph(get_edge_list(irs))
    return graph


def get_edge_list(irs: Dict) -> List[Tuple[str, str]]:
    path_list = []
    for i in range(len(irs)):
        path = irs[i]["nodes"]
        edge_list = get_edges_in_path(path)
        path_list.append(edge_list)
    return [edge for edge_list in path_list for edge in edge_list]


def get_edges_in_path(path: List[str]) -> List[Tuple[str, str]]:
    return list(zip(path[:], path[1:]))


def graph_connected(graph) -> bool:
    """
    Checks whether a graph/network is connected or not
    Uses DFS
    """
    return False


def connected_components(graph):
    """
    Returns the connected components of a graph (as list? set? sth else?)
    """


def connect_graph(components):
    """
    Find the shortest path on the hex grid from any vertex in one
    connected component to any vertex in another connected component

    Alternative: Randomly choose three vertices in each component
    and run single-source shortest path (BFS) from these

    The graph we're finding the SPs on is the complement of the produced network,
    i.e. the complete planar graph on the hex grid but with the edges and vertices
    already added in the produced network removed.

    V = all seed points in the grid - seed points already added to the produced network
    Each vertex has an edge to all of, its neighbours, i.e. |V| * 2 < |E| < |V| * 6

    Could use networkx k_edge_augmentation with k=1?
    """


# FOR TESTS
# >>> import networkx as nx
# >>> irs
# {0: {'nodes': [0, 1, 2, 3, 4]}, 1: {'nodes': [5, 6, 7, 8, 9]}, 2: {'nodes': [10, 11, 12, 13, 14]}}
# >>> def create_graph(irs: Dict) -> nx.Graph:
# ...     graph = nx.Graph(get_edge_list(irs))
# ...     return graph
# ...
# ...
# ... def get_edge_list(irs: Dict) -> List[Tuple[str, str]]:
# ...     path_list = []
# ...     for i in range(len(irs)):
# ...         path = irs[i]["nodes"]
# ...         edge_list = get_edges_in_path(path)
# ...         path_list.append(edge_list)
# ...     return [edge for edge_list in path_list for edge in edge_list]
# ...
# ...
# ... def get_edges_in_path(path: List[str]) -> List[Tuple[str, str]]:
# ...     return list(zip(path[:], path[1:]))
# ...
# >>> g = create_graph(irs)
# >>> type(g)
# <class 'networkx.classes.graph.Graph'>
# >>> nx.is_connected(g)
# False
# >>> nx.number_connected_components(g)
# 3
