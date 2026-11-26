from math import fsum
from typing import Dict, List, Sequence, Set, Tuple, TypedDict
import numpy as np
from geopandas import GeoDataFrame as gdf
import bicycle_network.demand as dmnd
import bicycle_network.graph
import networkx as nx
from networkx.classes.graph import Graph
import copy

# Gene: a path consisting of several edges
# Chromosome: a candidate solution (i.e. a network) consisting of genes
# Population: a collection of candidate solutions
# Generation: the population in a given iteration of the genetic algorithm

# TYPE ALIASES
type Population = Dict[int, Network]


class Network(TypedDict):
    fitness: float
    graph: Graph  # irs/ips


# MAIN


def run(
    start_pop: Population,
    iterations: int,
    mutation_probability: float,
    m: int,
    demand_matrix: gdf,
    seed_points: gdf,
    grid: gdf,
    seed: int = 3581054352,
) -> Dict[int, Population]:
    rng = np.random.default_rng(seed=seed)
    neighbours = _neighbour_lookup(seed_points)
    return _genetic_algorithm(
        start_pop,
        iterations,
        mutation_probability,
        m,
        demand_matrix,
        neighbours,
        grid,
        rng,
    )


# ORCHESTRATOR


def _genetic_algorithm(
    start_pop: Population,
    iterations: int,
    mutation_probability: float,
    m: int,
    demand_matrix: gdf,
    neighbours: dict[str, set[str]],
    grid: gdf,
    rng: np.random.Generator,
) -> Dict[int, Population]:
    n = len(start_pop)
    gens = {}
    gens[0] = start_pop
    pop = start_pop
    i = 1
    assert id(pop) == id(start_pop)
    while i <= iterations:  # or until some other convergence criteria
        parents = _reproduction(n, m, pop, rng)
        assert id(parents) != id(pop)
        children = _crossover(n, parents, rng)
        assert id(children) != id(parents)
        mutated = _mutation(children, neighbours, grid, rng, mutation_probability)
        assert id(mutated) != id(children)
        evaluated = _evaluation(
            mutated,
            demand_matrix,
        )
        assert id(evaluated) != id(mutated)
        # selected = _selection(evaluated, n)
        # gens[i] = selected
        # pop = selected
        gens[i] = evaluated
        pop = evaluated
        assert id(pop) != id(start_pop)
        i = i + 1
    return gens


# HOW MANY PARENTS CHOSEN FOR PRODUCTION? CAN BE USER CHOICE
# MUTATION OR not can be decided by probability
# can also keep some parents
# everything should be randomised
# important to allow some of the not so fit route sets to survive to next generation
#   as they can help make better solutions later
# up to decicde if using 'elitism' or not

# KEY FUNCTIONS


def _reproduction(
    n: int, m: int, pop: Population, rng: np.random.Generator
) -> Population:
    """
    Selects the fittest networks from a population to breed the next generation.
    Networks with higher demand have a higher probability of being chosen.
    One network can be chosen several times.

    Parameters:
    n (int): The number of networks to choose for the mating pool.
    m (int): Scalar to multiply number of chosen networks.
    pop (Population): The population of networks to choose parent networks from.

    Returns:
    A 'mating pool' population of {n * m} networks.
    """
    selected_ids = _select_ids(n, m, pop, rng)
    mating_pool: Population = {
        new_id: copy_network(pop[old_id]) for new_id, old_id in enumerate(selected_ids)
    }
    assert id(mating_pool) != id(pop)
    return mating_pool


def _evolution(pop: Population) -> Population:
    """
    Create the next population.
    """
    return pop


def _crossover(n: int, mating_pool: Population, rng: np.random.Generator) -> Population:
    """
    Creates a child population by randomly choosing pairs of parent networks from
    the mating pool and performing crossover on them.

    Parameters:
    n (int): Number of child networks to create.
    mating_pool (Population): Population of selected parent networks to use for mating.

    Returns:
    A population of child networks.
    """
    child_pop: Population = {}
    parent_ids = list(mating_pool.keys())
    i = 0
    while i < n:
        # parent_ids = list(mating_pool.keys())
        id_A, id_B = _choose_parents(parent_ids, rng)
        parent_A = mating_pool[id_A]
        parent_B = mating_pool[id_B]
        feasible_cuts = _get_shared_cut_nodes(parent_A["graph"], parent_B["graph"])
        if not feasible_cuts:
            continue
            # return _create_network(graph_a), _create_network(graph_b)
        # remove_A, remove_B = mating_pool.pop(id_A), mating_pool.pop(id_B)
        # assert remove_A == parent_A, remove_B == parent_B
        childg_A, childg_B = _inter_crossover(
            parent_A["graph"], parent_B["graph"], feasible_cuts, rng
        )
        child_A: Network = {"fitness": 0.0, "graph": childg_A}
        child_B: Network = {"fitness": 0.0, "graph": childg_B}
        child_pop[i] = child_A
        child_pop[i + 1] = child_B
        i = i + 2
    return child_pop


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


def _selection(pop: Population, n: int, rng) -> Population:
    """
    Selects n networks for the next generation.
    Either this or _reproduction should choose completely randomly
    (not favour the best ones). But maybe we should keep the very
    best one?

    Parameters:
    pop (Population): The population to choose n networks from
    n (int): The number of networks to choose

    Returns:
    Population of n networks
    """
    nw_ids = [id for id in pop]
    selected_ids = rng.choice(nw_ids, size=n, replace=False).tolist()
    selected: Population = {
        new_id: pop[old_id] for new_id, old_id in enumerate(selected_ids)
    }
    return selected


# REPRODUCTION HELPERS


def _sum_demand_coverage(pop: Population) -> float:
    return fsum([nw.get("fitness") for nw in pop.values()])


def _get_probabilities(pop: Population) -> np.ndarray:
    demand_sum = _sum_demand_coverage(pop)
    demand_shares = [nw.get("fitness") * 100 / demand_sum for nw in pop.values()]
    return _normalise(demand_shares)


def _select_ids(
    n: int,
    m: int,
    pop: Population,
    rng: np.random.Generator,
) -> List[int]:
    demand_shares = _get_probabilities(pop)
    nw_ids = [id for id in pop]
    selected_ids = rng.choice(nw_ids, size=n * m, replace=True, p=demand_shares)
    return selected_ids.tolist()


def _normalise(ls: List[float]) -> np.ndarray:
    arr = np.array(ls, dtype=np.float64)
    sum = np.sum(arr, dtype=np.float64)
    if sum == 0.0:
        raise ZeroDivisionError
    return arr / sum


# CROSSOVER HELPERS


def _choose_parents(parent_ids: List[int], rng: np.random.Generator) -> Tuple[int, int]:
    """
    Randomly chooses two ids of parent networks from the mating pool to be
    used for crossover.

    Params:
    pop (Population): Mating pool to choose parent networks from.

    Returns:
    A pair of network ids pointing to the chosen parent networks in {pop}.
    """
    chosen_ids = rng.choice(parent_ids, size=2, replace=False)
    a = int(chosen_ids[0])
    b = int(chosen_ids[1])
    return a, b


def _choose_single_parent(pop: Population) -> Network | None:
    """
    Randomly chooses a single parent network from the mating pool to be
    used for intra-string crossover.

    Parameters:
    pop (Population): Mating pool to choose parent network from.

    Returns:
    A single parent network.
    """


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


def _inter_crossover(
    graph_a: Graph,
    graph_b: Graph,
    cut_nodes: Set[str],
    rng: np.random.Generator,
) -> Tuple[Graph, Graph]:
    """
    Performs inter-string crossover by exchanging graph partitions
    between two parent networks to create two child networks. The
    partition node at which to cut is chosen randomly from a
    feasible set of cut vertices shared between both parent
    networks.

    Parameters:
    parent A (Network): First parent network.
    parent B (Network): Second parent network.
    cut_nodes (set): Set of cut vertices shared between both parent networks.

    Returns:
    A pair of child networks.
    """
    cut_node = _random_choice(sorted(cut_nodes), rng)
    components_a = _decompose_at(cut_node, graph_a)
    components_b = _decompose_at(cut_node, graph_b)
    # i = rng.integers(0, len(components_A))
    # j = rng.integers(0, len(components_B))
    # stripped_A = nx.compose_all(
    #     components_A[:i] + components_A[i + 1 :]
    # )  # remove ith comp
    # stripped_B = nx.compose_all(
    #     components_B[:j] + components_B[j + 1 :]
    # )  # remove jth comp
    # childg_A = nx.compose(stripped_A, components_B[j])  # graph_A - ith comp + jth comp
    # childg_B = nx.compose(stripped_B, components_A[i])  # graph_B - jth comp + ith comp
    comp_a, comp_b = _choose_similar_components(components_a, components_b)
    stripped_A = nx.compose_all([c for c in components_a if c is not comp_a])
    stripped_B = nx.compose_all([c for c in components_b if c is not comp_b])
    childg_A = nx.compose(stripped_A, comp_b)
    childg_B = nx.compose(stripped_B, comp_a)
    return (childg_A, childg_B)


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


def _intra_crossover(parent: Network) -> Tuple[Network, Network]:
    """
    Performs intra-string crossover by exchanging path sections
    between two paths in the same parent network to create two child
    networks.

    Parameters:
    parent (Network): Parent network to exchange path sections in.

    Returns:
    A pair of child networks.
    """
    raise NotImplementedError


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


# MUTATION HELPERS


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
            assert id(nw) != id(new_nw)
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
