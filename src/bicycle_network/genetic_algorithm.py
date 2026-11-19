from math import fsum
from typing import Dict, List, Set, Tuple, TypedDict
import numpy as np
from geopandas import GeoDataFrame as gdf
import bicycle_network.demand as dmnd
import bicycle_network.graph

# Gene: a path consisting of several edges
# Chromosome: a candidate solution (i.e. a network) consisting of genes
# Population: a collection of candidate solutions
# Generation: the population in a given iteration of the genetic algorithm

SEED: int = 3581054352
RNG: np.random.Generator = np.random.default_rng(seed=SEED)

# TYPE ALIASES
type Population = Dict[int, Network]


# TODO Put in own module/class file with methods, i.e. get path, get nodes, get geoms etc
class Network(TypedDict):
    demand_covered: float
    paths: Dict[int, List[str]]  # irs/ips


def copy_network(original: Network) -> Network:
    copy: Network = {
        "demand_covered": original["demand_covered"],
        "paths": dict(original["paths"]),
    }
    assert id(copy) != id(original)
    assert id(copy["paths"]) != id(original["paths"])
    return copy


# MAIN


def run(
    start_pop: Population,
    iterations: int,
    mutation_probability: float,
    demand_matrix: gdf,
    seed_points: gdf,
    grid: gdf,
) -> Dict[int, Population]:
    return _genetic_algorithm(
        start_pop, iterations, mutation_probability, demand_matrix, seed_points, grid
    )


# ORCHESTRATOR


def _genetic_algorithm(
    start_pop: Population,
    iterations: int,
    mutation_probability: float,
    demand_matrix: gdf,
    seed_points: gdf,
    grid: gdf,
) -> Dict[int, Population]:
    pop = start_pop
    n = len(pop)
    i = 0
    gens = {}
    assert id(pop) == id(start_pop)
    while i < iterations:  # or until some other convergence criteria
        parents = _selection(n, pop)
        assert id(parents) != id(pop)
        # children = _crossover(n, parents)
        # assert id(children) != id(parents)
        # mutated = _mutation(children)
        # evaluated = _evaluation(children, demand_matrix)
        # assert id(evaluated) != id(children) # this fails, why? demand should be updated
        mutated = _mutation(parents, seed_points, mutation_probability)
        evaluated = _evaluation(mutated, demand_matrix)
        pop = evaluated
        assert id(pop) != id(start_pop)
        gens[i] = pop
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


def _selection(n: int, pop: Population) -> Population:
    """
    Selects the fittest networks from a population to breed the next generation.
    Networks with higher demand have a higher probability of being chosen.
    One network can be chosen several times.

    Parameters:
    n (int): The number of networks to choose for the mating pool.
    pop (Population): The population of networks to choose parent networks from.

    Returns:
    A 'mating pool' population of {n} networks.
    """
    selected_ids = _select_ids(n, pop)
    mating_pool: Population = {
        new_id: pop[old_id] for new_id, old_id in enumerate(selected_ids)
    }
    assert id(mating_pool) != id(pop)
    return mating_pool


def _evolution(pop: Population) -> Population:
    """
    Create the next population.
    """
    return pop


def _crossover(n: int, mating_pool: Population) -> Population:
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
        id_A, id_B = _choose_parents(parent_ids)
        parent_A = mating_pool[id_A]
        parent_B = mating_pool[id_B]
        shared_nodes = _get_shared_nodes(parent_A, parent_B)
        if shared_nodes is None:
            continue
        # remove_A, remove_B = mating_pool.pop(id_A), mating_pool.pop(id_B)
        # assert remove_A == parent_A, remove_B == parent_B
        child_A, child_B = _inter_intra_crossover(parent_A, parent_B, shared_nodes)
        assert id(child_A) != id(parent_A)
        assert id(child_B) != id(parent_B)
        child_pop[i] = child_A
        child_pop[i + 1] = child_B
        i = i + 2
    return child_pop


def _mutation(
    pop: Population, seed_points: gdf, mutation_probability=0.3
) -> Population:
    """
    Randomly mutates a random selection of networks in the population
    by adding, removing or replacing edges.

    Parameters:
    pop (Population): The population of networks to randomly choose networks from

    Returns:
    A new population containing original and mutated networks.
    """
    mutate = np.random.choice(
        a=[True, False],
        size=len(pop),
        replace=True,
        p=[mutation_probability, 1.0 - mutation_probability],
    ).tolist()
    mutated: Population = {}
    for nw_id, nw in pop.items():
        mutated[nw_id] = _mutate(nw, seed_points) if mutate[nw_id] else nw
    return mutated


def _evaluation(pop: Population, demand_matrix) -> Population:
    """
    1. Calculate the percentage of demand covered for each network in the population.
    2. Choose only the n fittest? (Not now - For now, let's just do the above)
    """
    return _recalculcate_fitness(pop, demand_matrix)


# SELECTION HELPERS


def _sum_demand_coverage(pop: Population) -> float:
    return fsum([nw.get("demand_covered") for nw in pop.values()])


def _get_probabilities(pop: Population) -> np.ndarray:
    demand_sum = _sum_demand_coverage(pop)
    demand_shares = [nw.get("demand_covered") * 100 / demand_sum for nw in pop.values()]
    return _normalise(demand_shares)


def _select_ids(n: int, pop: Population) -> List[int]:
    demand_shares = _get_probabilities(pop)
    nw_ids = [id for id in pop]
    selected_ids = RNG.choice(nw_ids, size=n, replace=True, p=demand_shares)
    return selected_ids.tolist()


def _normalise(ls: List[float]) -> np.ndarray:
    arr = np.array(ls, dtype=np.float64)
    sum = np.sum(arr, dtype=np.float64)
    if sum == 0.0:
        raise ZeroDivisionError
    return arr / sum


# CROSSOVER HELPERS


def _choose_parents(parent_ids: List[int]) -> Tuple[int, int]:
    """
    Randomly chooses two ids of parent networks from the mating pool to be
    used for crossover.

    Params:
    pop (Population): Mating pool to choose parent networks from.

    Returns:
    A pair of network ids pointing to the chosen parent networks in {pop}.
    """
    chosen_ids = RNG.choice(parent_ids, size=2, replace=False)
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


def _inter_crossover(
    parents: Tuple[Network, Network],
) -> Tuple[Network, Network] | None:
    """
    Performs inter-string crossover by exchanging two complete paths
    between two parent networks to create two child networks.

    Parameters:
    parents (Tuple[Network, Network]): Pair of parents to exchange paths between.

    Returns:
    A pair of child networks.
    """
    # CONNECTIVITY MUST BE ENFORCED


def _intra_crossover(parent: Network) -> Tuple[Network, Network] | None:
    """
    Performs intra-string crossover by exchanging path sections
    between two paths in the same parent network to create two child
    networks.

    Parameters:
    parent (Network): Parent network to exchange path sections in.

    Returns:
    A pair of child networks.
    """


def _inter_intra_crossover(
    parent_A: Network, parent_B: Network, shared_nodes: Set[str]
) -> Tuple[Network, Network]:
    """
    Exchanges path sections from two paths with at least one shared node
    from two parent networks to create two child networks.
    Assumes that {parent_A} and {parent_B} has at least one shared node.

    Parameters:
    parent A (Network): First parent network.
    parent B (Network): Second parent network.
    shared_nodes (set): Set of node ids shared by the two networks.

    Returns:
    A pair of child networks.

    For each parent pair (inspired by Mesbah 2012):
    1. Assume that each parent is a connected network
    2. Determine shared vertices (edges) between the parents
    3. Randomly choose 1 shared vertex*
    4. Swap all edges that precede the selected shared vertex
    5. This produces 2 child solutions that are both connected

    *if no shared vertex exists, this parent pair doesn't produce new child solutions
    """
    # Randomly choose 1 shared vertex
    cut_node = str(RNG.choice(list(shared_nodes)))

    # Find the paths that cut_node is in (naive method)
    cut_idx_A, path_id_A, path_A = _find_node(cut_node, parent_A)
    cut_idx_B, path_id_B, path_B = _find_node(cut_node, parent_B)
    assert parent_A["paths"][path_id_A] == path_A, (
        parent_B["paths"][path_id_A] == path_B
    )

    # Exchange
    new_A, new_B = _exchange_path_sections(cut_idx_A, cut_idx_B, path_A, path_B)
    assert id(new_A) != id(path_A), id(new_B) != id(path_B)

    # Put new paths into child networks
    child_A = copy_network(parent_A)
    child_B = copy_network(parent_B)
    child_A["paths"][path_id_A] = new_A
    child_B["paths"][path_id_B] = new_B
    assert id(parent_A["paths"][path_id_A]) != id(new_A), id(
        parent_B["paths"][path_id_A]
    ) != id(new_B)
    assert id(parent_A) != id(child_A), id(parent_B) != id(child_B)

    return (child_A, child_B)


def _get_node_set(nw: Network) -> Set[str]:
    return set().union(*nw["paths"].values())


def _get_shared_nodes(nw_A, nw_B) -> Set[str] | None:
    """
    Finds and returns the shared nodes between two networks.

    Parameters:
    nw_A (Network): First network.
    nw_B (Network): Second network.

    Returns:
    A set of shared nodes between the two networks if any exist, otherwise None.
    """
    # use set intersection
    set_A = _get_node_set(nw_A)
    set_B = _get_node_set(nw_B)
    shared = set_A & set_B
    return shared or None


def _find_node(node: str, nw: Network) -> Tuple[int, int, List[str]]:
    """
    Finds {node} and its path in {nw}.
    Only gets the first occurrence of the node.
    Naive method.

    Parameters:
    node (str): The node to search for.
    nw (Network): The network to search in.

    Returns:
    A tuple containing 1) the node's index in its path, 1) the path id, 2) the path itself.
    """
    node_index = -1
    path_id = -1
    path_list = []
    for i, path in nw["paths"].items():
        if node in path:
            node_index = path.index(node)
            path_id = i
            path_list = path
            break
    return (node_index, path_id, path_list)


def _find_edge(edge: Tuple[str, str], nw: Network) -> Tuple[int, int, int]:
    """
    Finds {edge} and its path in {nw}. Naive method.

    Parameters:
    node (str): The node to search for.
    nw (Network): The network to search in.

    Returns:
    A tuple containing 1) the node's index in its path, 1) the path id, 2) the path itself.
    """
    u, v = edge
    for path_id, path in nw["paths"].items():
        if u in path and v in path:
            u_idx = path.index(u)
            v_idx = path.index(v)
            diff = v_idx - u_idx
            if diff == 1:
                return (path_id, u_idx, v_idx)
            if diff == -1:
                return (path_id, v_idx, u_idx)
    raise KeyError


def _exchange_path_sections(
    index_A: int, index_B: int, path_A: List[str], path_B: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Creates two new paths from two paths by exchanging sections at index.

    Parameters:
    index_A (int): The index to cut in {path_A}
    index_B (int): The index to cut in {path_B}
    path_A (List[str]): First path to exchange sections
    path_B (List[str]): Second path to exchange sections

    Returns:
    A tuple containing two new paths.
    """
    new_A = path_A[:index_A] + path_B[index_B:]
    new_B = path_B[:index_B] + path_A[index_A:]
    return (new_A, new_B)


# MUTATION HELPERS


def _mutate(nw: Network, seed_points: gdf) -> Network:
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
    action = str(RNG.choice(["add", "remove", "replace"], size=1))
    match action:
        case "add":
            nw = _add_edge(nw, seed_points)
        case "remove":
            nw = _remove_edge(nw)
        case "replace":
            nw = _replace_edge(nw, seed_points)
    return nw


def _add_edge(nw: Network, seed_points: gdf) -> Network:
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
    nodes_in_network = _get_node_set(nw)
    chosen_node = ""
    node_pool = {}
    while len(node_pool) == 0:
        chosen_node = str(RNG.choice(list(nodes_in_network)))
        neighbours = set(
            seed_points.loc[seed_points["id"] == chosen_node, "neighbours"]
            .values[0]
            .values()  # type: ignore
        )
        node_pool = neighbours - nodes_in_network  # set difference
    new_node = str(RNG.choice(list(node_pool)))
    mutated_nw = copy_network(nw)
    n_paths = len(mutated_nw["paths"])
    mutated_nw["paths"][n_paths] = [chosen_node, new_node]
    assert id(nw) != id(mutated_nw)
    return mutated_nw


def _remove_edge(nw: Network) -> Network:
    """
    Removes an edge from one of the paths of the input network.
    The edge is only removed if it is not a bridge, i.e. removing
    the edge does not disconnect the network. If no edge exists
    that isn't a bridge in the network, the original network is
    returned. Nothing is done to the path that contained the
    removed edge, so the path itself will be disconnected.
    This method keeps connectivity in the graph and mutates a copy of
    the input network to avoid altering the original network.

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
    edges = bicycle_network.graph.get_edge_set(nw)
    bridges = bicycle_network.graph.get_bridges(nw)
    edge_pool = edges - bridges  # set difference
    if len(edge_pool) == 0:
        return nw
    fst, snd = tuple(RNG.choice(list(edge_pool)))
    chosen_edge = (str(fst), str(snd))
    path_id, u_idx, v_idx = _find_edge(chosen_edge, nw)
    mutated_nw = copy_network(nw)
    removed_u = mutated_nw["paths"][path_id].pop(u_idx)
    removed_v = mutated_nw["paths"][path_id].pop(v_idx)
    same_edge = ((removed_u, removed_v) == chosen_edge) or (
        (removed_v, removed_u) == chosen_edge
    )
    assert same_edge
    assert id(nw) != id(mutated_nw)
    return mutated_nw


def _replace_edge(nw: Network, seed_points: gdf) -> Network:
    """ """
    nw = _remove_edge(nw)
    return _add_edge(nw, seed_points)


# EVALUATION HELPERS


def _recalculcate_fitness(pop: Population, demand_matrix) -> Population:
    for nw in pop.values():
        nw["demand_covered"] = dmnd.network_demand_coverage(
            _get_node_set(nw), demand_matrix
        )
    return pop


# OTHER HELPERS


def json_to_pop(data) -> Population:
    pop: Population = {
        int(nw_id): {
            "demand_covered": nw["demand_covered"],
            "paths": {int(path_id): path for path_id, path in nw["paths"].items()},
        }
        for nw_id, nw in data.items()
    }
    return pop


def json_to_gens(data) -> Dict[int, Population]:
    gens: Dict[int, Population] = {
        int(pop_id): json_to_pop(pop) for pop_id, pop in data.items()
    }
    return gens


# def json_to_gens(data) -> Dict[int, Population]:
#     gens: Dict[int, Population] = {
#         int(pop_id): {
#             int(nw_id): {
#                 "demand_covered": nw["demand_covered"],
#                 "paths": {int(path_id): path for path_id, path in nw["paths"].items()},
#             }
#             for nw_id, nw in pop.items()
#         }
#         for pop_id, pop in data.items()
#     }
#     return gens


# VISUALISATION


def plot_network():
    """
    Visualises a chromosome as a network, including the percentage of demand covered.
    """


def plot_generation():
    """
    Creates a scatter plot of a generation, where each blob is a chromosome (network),
    the x axis is the total length of the network, and the y axis is the percentage of
    total demand covered.
    """
