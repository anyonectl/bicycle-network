from bicycle_network.genetic_algorithm import Network
from typing import Dict, List, Tuple
import networkx as nx


def is_connected(nw: Network) -> bool:
    return nx.is_connected(_get_graph(nw))


def network_len(nw: Network) -> float:
    # (possible temporary) proxy: number of edges.
    # regular hexagons means all edges will have same length
    # parallel edges are not allowed in networkx undirected graphs,
    # so overlapping edges won't be counted more than once, which is what we want
    return nx.number_of_edges(_get_graph(nw))


def _get_graph(nw: Network) -> nx.Graph:
    graph = nx.Graph(_get_edge_list(nw["paths"]))
    return graph


def _get_edge_list(paths: Dict[int, List[str]]) -> List[Tuple[str, str]]:
    path_list = []
    for path in paths.values():
        edge_list = _get_edges_in_path(path)
        path_list.append(edge_list)
    return [edge for edge_list in path_list for edge in edge_list]


def _get_edges_in_path(path: List[str]) -> List[Tuple[str, str]]:
    return list(zip(path[:], path[1:]))
