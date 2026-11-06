# TODO SQL database with seed_points, demand_matrix, grid instead...
from typing import Dict, List, Tuple

from tqdm import tqdm
import numpy as np
import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely import LineString, Point
from bicycle_network.hex import direction_bias
from bicycle_network.demand import network_demand_coverage


# TODO: bias terms for each node in the VNS?
# TODO: needs some way to stop if no node in VNS is possible. boolean in irs dict?
def select_next_node(
    path_id: int,
    irs: Dict[int, List[str]],
    seed_points: gpd.GeoDataFrame,
    rng: np.random.Generator,
    demand_type: str,
):
    current_node = irs[path_id][-1]
    previous_node = irs[path_id][-2] if len(irs[path_id]) > 1 else current_node

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
        seed_points["id"] == current_node, "geometry"
    ].iat[
        0
    ]  # type: ignore
    previous_point: Point = seed_points.loc[
        seed_points["id"] == previous_node, "geometry"
    ].iat[
        0
    ]  # type: ignore
    vns["direction_bias"] = vns["geometry"].apply(
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
        if next_node not in irs[path_id]:
            irs[path_id].append(next_node)
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
    irs: Dict[int, List[str]] = {}
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

        # Add to list of lists
        irs[i] = [first_node]

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
    demand_matrix: gpd.GeoDataFrame,
    rng: np.random.Generator,
    demand_type: str = "share",
) -> Dict[int, dict]:
    """
    Create a population of {n} connected networks, where each network
    contains {max_paths} paths, and each path contains max {max_nodes}
    nodes.

    The starting nodes for each path in a network are randomly selected
    from the {k} nodes with highest demand from {seed_points}. Nodes
    with higher demand will have a higher probability of being chosen
    as a starting node.

    Parameters:
        n (int): Number of networks to create
        k (int): Number of nodes to use for the initial node set
        max_paths (int): Max number of paths in each network
        max_nodes (int): Max number of nodes in each path
        seed_points (gpd.GeoDataFrame):
        demand_matrix (gpd.GeoDataFrame):
        rng (numpy.random.Generator):
        demand_type (str): "share" or "absolute"

    Returns:
    A dictionary of dictionaries representing a population of networks.

    """
    population = {}
    pbar = tqdm(desc=f"Generate {n} connected networks. Loop iterations:", total=n * 15)
    i = 0
    while i < n:
        irs = generate_network(k, max_paths, max_nodes, seed_points, rng, demand_type)
        if is_connected(irs):
            nw = create_dict(irs, demand_matrix)
            population[i] = nw
            i = i + 1
        pbar.update(1)
    return population


def normalise_probabilities(df: pd.Series) -> np.ndarray:
    p = df.to_numpy(dtype=np.float64)
    sum = np.sum(p, dtype=np.float64)
    if sum == 0.0:
        raise ZeroDivisionError
    return p / np.sum(p, dtype=np.float64)


def create_dict(irs: Dict[int, List[str]], demand_matrix: gpd.GeoDataFrame) -> Dict:
    # Create dict for network
    # TODO: nw should be dict with demand_covered and paths as keys,
    # where paths is a dict with path_id as keys (integers) and ordered cell_ids as values
    node_list = get_nodes(irs)
    nw = {
        "demand_covered": network_demand_coverage(node_list, demand_matrix),
        "paths": irs,
    }
    return nw


def get_nodes(irs: Dict[int, List[str]]) -> list:
    graph = get_graph(irs)
    nodes = list(graph)
    return nodes


def get_geoms(nw: dict, seed_points: gpd.GeoDataFrame) -> List[LineString]:
    """
    Get the geometries matching seed point ids.

    Parameters:
        nw (dict):  Dictionary representing a single network with key "paths"
                    containing a dictionary that maps path_id to ordered
                    lists of cell_ids
        seed_points (geopandas.GeoDataFrame):   Geodataframe with seed points

    Returns:
    A list of shapely.Point that can be used as geometries for seed points.
    """
    id_to_centroid = seed_points.set_index("id")["geometry"].to_dict()
    geom = [
        LineString([id_to_centroid[point_id] for point_id in path])
        for path in nw["paths"]
    ]
    return geom


def get_graph(irs: Dict[int, List[str]]) -> nx.Graph:
    graph = nx.Graph(get_edge_list(irs))
    return graph


def is_connected(irs: Dict[int, List[str]]) -> bool:
    return nx.is_connected(get_graph(irs))


def get_edge_list(irs: Dict[int, List[str]]) -> List[Tuple[str, str]]:
    path_list = []
    for _, path in irs.items():
        edge_list = get_edges_in_path(path)
        path_list.append(edge_list)
    return [edge for edge_list in path_list for edge in edge_list]


def get_edges_in_path(path: List[str]) -> List[Tuple[str, str]]:
    return list(zip(path[:], path[1:]))


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
