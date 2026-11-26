from typing import Dict, List, Sequence, Set, Tuple, TypedDict
import numpy as np
from geopandas import GeoDataFrame as gdf
import bicycle_network.demand as dmnd
import bicycle_network.graph
import networkx as nx
from networkx.classes.graph import Graph
import copy

type Population = Dict[int, Network]


class Network(TypedDict):
    fitness: float
    graph: Graph  # irs/ips


def run(
    start_pop: Population,
    iterations: int,
    budget: int,
    tournament_size: int,
    n_elites: int,
    mutation_probability: float,
    seed_points: gdf,
    grid: gdf,
    demand_matrix: gdf,
    seed: int = 3581054352,
) -> Dict[int, Population]:
    rng = np.random.default_rng(seed=seed)
    neighbours = _neighbour_lookup(seed_points)
    n = len(start_pop)
    pop = _evaluation(start_pop, demand_matrix)
    gens: dict[int, Population] = {}
    gens[0] = pop
    i = 1
    # assert id(pop) == id(start_pop)  # fails
    while i <= iterations:  # or until some other convergence criteria
        elites = _find_best_networks(pop, n_elites, budget)
        children = _reproduction(pop, n - n_elites, tournament_size, budget, rng)
        assert id(children) != id(pop)
        mutated = _mutation(children, neighbours, grid, rng, mutation_probability)
        assert id(mutated) != id(children)
        evaluated = _evaluation(
            mutated,
            demand_matrix,
        )
        assert id(evaluated) != id(mutated)
        pop = dict(enumerate(elites + list(evaluated.values())))
        gens[i] = pop
        assert id(pop) != id(start_pop)
        i = i + 1
    return gens


def _find_best_networks(pop: Population, n_elites: int, budget: int) -> List[Network]:
    """
    Find the `n_elites` best networks of a population and return them in a list.
    """
    ranked = sorted(pop.values(), key=lambda nw: _rank(nw, budget), reverse=True)
    return ranked[:n_elites]


def _reproduction(
    pop: Population,
    n_children: int,
    tournament_size: int,
    budget: int,
    rng: np.random.Generator,
) -> Population:
    """
    Creates `n_children` child networks by performing crossover of selected parents.
    Parents are selected using tournament selection.

    Parameters:
    pop (Population): Population of networks to select parent networks from.
    n_children (int): Number of child networks to create.
    budget (int): Max number of edges allowed in a network.
    rng (numpy.random.Generator): Random generator.

    Returns:
    A population of `n_children` networks.
    """
    children: List[Network] = []
    while len(children) < n_children:
        parent_a, parent_b = _select_parents(pop, tournament_size, budget, rng)
        child_pair = _crossover(parent_a, parent_b, rng)
        if child_pair is None:
            continue  # TODO maybe use parents as children instead? they may breed differently next round.
        children.extend(child_pair)
    return dict(enumerate(children[:n_children]))


def _select_parents(
    pop: Population, tournament_size: int, budget: int, rng: np.random.Generator
) -> Tuple[Network, Network]:
    """
    Selects two parent networks in two tournament selections.
    The same parent can't be chosen twice.
    """
    nw_ids = list(pop)
    parent_a = _tournament_selection(pop, nw_ids, tournament_size, budget, rng)
    nw_ids.remove(parent_a)
    parent_b = _tournament_selection(pop, nw_ids, tournament_size, budget, rng)
    return pop[parent_a], pop[parent_b]


def _tournament_selection(
    pop: Population,
    nw_ids: List[int],
    tournament_size: int,
    budget: int,
    rng: np.random.Generator,
) -> int:
    """
    Randomly choose a few parent networks from a population
    and return the id of the best network.

    Networks are compared by rank, which is a two-step process:
    1) A network within budget always wins over a network over budget.
    2a) A within-budget network with a higher fitness wins over within-budget networks with lower fitness.
    2b) Networks less over budget wins over networks more over budget.

    Params:
    pop: Population to
    nw_ids: Network IDs
    tournament_size (int): How many candidates to randomly pick.
    budget (int): User-defined budget constraint of max number of edges
    rng (numpy.random.Generator): Random geneator

    Returns:
    Network ID of chosen network
    """
    nw_candidates = [nw_ids[i] for i in rng.integers(len(nw_ids), size=tournament_size)]
    return max(nw_candidates, key=lambda nw_id: _rank(pop[nw_id], budget))


def _rank(nw: Network, budget: int) -> Tuple[int, float]:
    """
    Assigns a rank to a network, to be used for selection.

    Problem with this method is that it is unlikely that networks over budget
    will ever be chosen, which limits the search space for the GA.
    """
    cost = nw["graph"].number_of_edges()
    if cost > budget:
        return (0, -float(cost - budget))
    return (1, nw["fitness"])


def _crossover(
    parent_a: Network, parent_b: Network, rng: np.random.Generator
) -> Tuple[Network, Network] | None:
    """
    TODO: Random partition selection instead of similarity?

    Performs "inter-string" crossover by exchanging graph partitions
    between two parent networks to create two child networks. The
    partition node at which to cut is chosen randomly from a
    feasible set of cut vertices shared between both parent
    networks.

    Parameters:
    parent_a (Network): First parent network.
    parent_b (Network): Second parent network.
    rng (numpy.random.Generator): Random generator.

    Returns:
    A pair of child networks, or None if no shared cut node was found.
    """
    graph_a, graph_b = parent_a["graph"], parent_b["graph"]
    feasible_cuts = _get_shared_cut_nodes(graph_a, graph_b)
    if not feasible_cuts:
        return None
    cut_node = _random_choice(sorted(feasible_cuts), rng)
    components_a = _decompose_at(cut_node, graph_a)
    components_b = _decompose_at(cut_node, graph_b)
    comp_a, comp_b = _choose_similar_components(components_a, components_b)
    stripped_A = nx.compose_all([c for c in components_a if c is not comp_a])
    stripped_B = nx.compose_all([c for c in components_b if c is not comp_b])
    child_a = nx.compose(stripped_A, comp_b)
    child_b = nx.compose(stripped_B, comp_a)
    return _create_network(child_a), _create_network(child_b)


def _get_shared_cut_nodes(graph_a: Graph, graph_b: Graph) -> Set[str]:
    """
    Finds and returns the shared nodes between two networks that are
    also cut vertices in both networks.

    Parameters:
    graph_a (Graph): First network's graph.
    graph_b (Graph): Second network's graph.

    Returns:
    A set of cut vertices shared between the two networks.
    """
    shared = set(graph_a) & set(graph_b)
    return (
        shared
        & set(nx.articulation_points(graph_a))
        & set(nx.articulation_points(graph_b))
    )


def _decompose_at(cut_node: str, graph: Graph) -> List[Graph]:
    """
    Decompose a connected Networkx graph into several connected
    components by removing a cut vertex {v} and then
    re-introducing {v} to all subgraphs.

    Paramters:
    v (str): Cut vertex/articulation point in G.
    G (Graph): Connected Networkx graph.

    Returns:
    List of subgraphs that share node v.
    """
    disconnected = graph.copy()
    disconnected.remove_node(cut_node)
    components = sorted(nx.connected_components(disconnected), key=min)
    return [graph.subgraph(comp | {cut_node}).copy() for comp in components]


def _choose_similar_components(A: List[Graph], B: List[Graph]) -> Tuple[Graph, Graph]:
    """
    Uses Jaccard similarity to choose a graph from each input list.
    Most similar graph pair is returned.
    """
    chosen_pair = max(
        ((a, b) for a in A for b in B),
        key=_similarity_score,
    )
    return chosen_pair


def _similarity_score(pair: Tuple[Graph, Graph]) -> float:
    a = pair[0]
    b = pair[1]
    return len(set(a) & set(b)) / len(set(a) | set(b))


def _mutation(
    pop: Population,
    neighbours: dict[str, set[str]],
    grid: gdf,
    rng: np.random.Generator,
    mutation_probability=0.3,
) -> Population:
    """
    Randomly mutates a random selection of networks in the population
    by adding, removing or replacing edges.

    Parameters:
    pop (Population): The population of networks to randomly choose networks from

    Returns:
    A new population containing original and mutated networks.
    """
    mutate = rng.choice(
        a=[True, False],
        size=len(pop),
        replace=True,
        p=[mutation_probability, 1.0 - mutation_probability],
    ).tolist()
    mutated: Population = {}
    for nw_id, nw in pop.items():
        mutated[nw_id] = _mutate(nw, neighbours, grid, rng) if mutate[nw_id] else nw
    return mutated


def _evaluation(pop: Population, demand_matrix: gdf) -> Population:
    """
    Calculate the fitness for each network in the population.

    Returns a population of networks with recalculated fitness scores.
    """
    return {
        nw_id: {
            "fitness": _calculate_fitness(nw["graph"], demand_matrix),
            "graph": nw["graph"],
        }
        for nw_id, nw in pop.items()
    }


def _mutate(
    nw: Network, neighbours: dict[str, set[str]], grid: gdf, rng: np.random.Generator
) -> Network:
    """
    Mutate the given network either by adding, removing or
    replacing an edge. Whether to add, remove or replace is
    determined randomly.

    The original input network is not altered.

    Parameters:
    nw (Network): The network to mutate.

    Returns:
    A mutated version of the input network.
    """
    assert nx.is_connected(nw["graph"])
    action = _random_choice(["add", "remove", "replace"], rng)
    match action:
        case "add":
            new_nw = _add_edge(nw, neighbours, rng)
            assert id(nw) != id(new_nw)
        case "remove":
            new_nw = _remove_edge(nw, rng)
            # assert id(nw) != id(new_nw) # fails
        case "replace":
            new_nw = _replace_edge(nw, neighbours, rng)
            assert id(nw) != id(new_nw)
        case _:
            raise NotImplementedError
    assert nx.is_connected(new_nw["graph"])
    return new_nw


def _add_edge(
    nw: Network, neighbour_lookup: dict[str, set[str]], rng: np.random.Generator
) -> Network:
    """
    Adds an edge to the network that connects an existing node
    in the network with a neighbouring node not already in the network.
    The added edge is added as a new path, which may not be what we want.
    This method keeps connectivity in the graph and mutates a copy of
    the input network to avoid altering the original network.

    Parameters:
    nw (Network): The network to mutate.
    seed_points (geopandas.GeoDataFrame): Matrix containing all seed points and their neighbours

    Returns:
    A mutated version of the input network with one edge added.

    Idea, eager version:
    1. Get all nodes of network
    2. Get their neighbours
    3. Only keep neighbours that are not already in the network
    4. This will be the set of possible edge additions

    Alternatively, lazy version:
    1. Get all nodes of network (set)
    2. Randomly choose one node
    3. Get the neighbours of this node (either pass seed_points as param to this function,
        or create nw as a networkx graph)
    4. Check if any neighbours are not already in the network.
    5. If all neighbours are already in the network, go back to step 2.
    6. Randomly choose a node from the set of neighbours not already in the network.
    7. Add edge from chosen node in step 2 to chosen node in step 6.
    8. Should this be its own path, or should it be a part of the path with the
        original chosen node? I guess it shouldn't matter?
    """
    nodes = nw["graph"].nodes
    chosen_node = ""
    node_pool = {}
    while len(node_pool) == 0:
        chosen_node = str(rng.choice(nodes))  # type: ignore
        neighbours = neighbour_lookup[chosen_node]
        node_pool = neighbours - nodes  # set difference
    new_node = str(rng.choice(sorted(node_pool)))
    assert new_node not in nw["graph"]
    mutated_nw = copy_network(nw)
    assert id(nw) != id(mutated_nw)
    mutated_nw["graph"].add_edge(chosen_node, new_node)
    assert id(nw) != id(mutated_nw)
    return mutated_nw


def _remove_edge(nw: Network, rng: np.random.Generator) -> Network:
    """
    Removes an edge but no nodes.

    Removes an edge from one of the paths of the input network.
    The edge is only removed if it is not a bridge, i.e. removing
    the edge does not disconnect the network. If no edge exists
    that isn't a bridge in the network, the original network is
    returned. Nothing is done to the path that contained the
    removed edge, so the path itself will be disconnected.
    This method keeps connectivity in the graph and mutates a copy of
    the input network to avoid altering the original network.
    ISSUE: If there are parallel edges from different paths in the
    network dict structure, this method only removes one of them.
    May be fixed in the _find_edge function if we find all occurrences.

    Eager method:
    1. Gets all edges in network
    2. Finds the set of edges that aren't bridges
    3. Randomly chooses an edge from the non-bridge set
    4. Finds this edge in the network structure and removes it
    5. Returns the newly mutated network

    Parameters:
    nw (Network): The network to mutate.

    Returns:
    A mutated version of the input network with one edge removed.
    """
    graph = nw["graph"]
    bridges = {frozenset(edge) for edge in nx.bridges(graph)}
    feasible_edges = sorted(
        tuple(sorted(edge) for edge in graph.edges if frozenset(edge) not in bridges)
    )
    if not feasible_edges:
        return nw
    u, v = _random_choice(feasible_edges, rng)
    mutated_graph = graph.copy()
    assert id(graph) != id(mutated_graph)
    mutated_graph.remove_edge(u, v)
    assert nx.is_connected(mutated_graph)
    return _create_network(mutated_graph)


def _replace_edge(
    nw: Network, neighbours: dict[str, set[str]], rng: np.random.Generator
) -> Network:
    """ """
    new_nw = _remove_edge(nw, rng)
    return _add_edge(new_nw, neighbours, rng)


# EVALUATION HELPERS


def _calculate_fitness(graph: Graph, demand_matrix) -> float:
    return dmnd.network_demand_coverage(set(graph), demand_matrix)


# OTHER HELPERS


def _create_network(graph: Graph) -> Network:
    return {"fitness": 0.0, "graph": graph}


def _random_choice[T](items: Sequence[T], rng: np.random.Generator) -> T:
    """
    Returns a randomly chosen element from `items`.
    """
    i = int(rng.integers(len(items)))
    return items[i]


def _neighbour_lookup(seed_points: gdf) -> Dict[str, Set[str]]:
    """
    Returns a dict that maps every seed point ID to a set containing
    their neighbour seed IDs.
    """
    return {
        str(cell_id): {str(neighbour_id) for neighbour_id in neighbours.values()}
        for cell_id, neighbours in zip(seed_points["id"], seed_points["neighbours"])
    }


def copy_network(original: Network) -> Network:
    new = copy.deepcopy(original)
    assert id(new) != id(original)
    assert id(new["graph"]) != id(original["graph"])
    return new


def copy_pop(original: Population) -> Population:
    new = copy.deepcopy(original)
    assert id(new) != id(original)
    return new


def network_to_jsonable(nw: Network) -> dict:
    return {
        "demand_covered": float(nw["fitness"]),
        "graph": nx.node_link_data(nw["graph"], edges="edges"),
    }


def generations_to_jsonable(gens: Dict[int, Population]) -> dict:
    return {
        str(gen): {str(nw_id): network_to_jsonable(nw) for nw_id, nw in pop.items()}
        for gen, pop in gens.items()
    }


def jsonable_to_network(data: dict) -> Network:
    return {
        "fitness": float(data["demand_covered"]),
        "graph": nx.node_link_graph(data["graph"], edges="edges"),
    }


def jsonable_to_pop(data: dict) -> Population:
    pop: Population = {
        int(nw_id): jsonable_to_network(nw) for nw_id, nw in data.items()
    }
    return pop


def jsonable_to_gens(data: dict) -> Dict[int, Population]:
    return {int(pop_id): jsonable_to_pop(pop) for pop_id, pop in data.items()}


def convert_start_json_to_pop(data) -> Population:
    pop: Population = {
        int(nw_id): {
            "fitness": nw["fitness"],
            "graph": nx.Graph(bicycle_network.graph.get_edge_list(nw["paths"])),
        }
        for nw_id, nw in data.items()
    }
    return pop
