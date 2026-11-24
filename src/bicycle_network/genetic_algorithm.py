from math import fsum
from typing import Dict, List, Set, Tuple, TypedDict
import numpy as np
from geopandas import GeoDataFrame as gdf
import bicycle_network.demand as dmnd
import bicycle_network.plot as bcnwplt
import bicycle_network.graph
import networkx as nx
from networkx.classes.graph import Graph
import copy

# Gene: a path consisting of several edges
# Chromosome: a candidate solution (i.e. a network) consisting of genes
# Population: a collection of candidate solutions
# Generation: the population in a given iteration of the genetic algorithm

SEED: int = 3581054352
RNG: np.random.Generator = np.random.default_rng(seed=SEED)

# TYPE ALIASES
type Population = Dict[int, Network]


class Network(TypedDict):
    demand_covered: float
    graph: Graph  # irs/ips


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
        "demand_covered": float(nw["demand_covered"]),
        "graph": nx.node_link_data(nw["graph"], edges="edges"),
    }


def generations_to_jsonable(gens: Dict[int, Population]) -> dict:
    return {
        str(gen): {str(nw_id): network_to_jsonable(nw) for nw_id, nw in pop.items()}
        for gen, pop in gens.items()
    }


def jsonable_to_network(data: dict) -> Network:
    nw: Network = {
        "demand_covered": float(data["demand_covered"]),
        "graph": nx.node_link_graph(data["graph"]),
    }
    return nw


def jsonable_to_pop(data: dict) -> Population:
    pop: Population = {
        int(nw_id): jsonable_to_network(nw) for nw_id, nw in data.items()
    }
    return pop


def jsonable_to_gens(data: dict) -> Dict[int, Population]:
    return {int(pop_id): jsonable_to_pop(pop) for pop_id, pop in data.items()}


# MAIN


def run(
    start_pop: Population,
    iterations: int,
    mutation_probability: float,
    m: int,
    budget: int,
    max_edges: int,
    demand_matrix: gdf,
    seed_points: gdf,
    grid: gdf,
    penalty: float = 0.2,
) -> Dict[int, Population]:
    return _genetic_algorithm(
        start_pop,
        iterations,
        mutation_probability,
        m,
        demand_matrix,
        budget,
        max_edges,
        penalty,
        seed_points,
        grid,
    )


# ORCHESTRATOR


def _genetic_algorithm(
    start_pop: Population,
    iterations: int,
    mutation_probability: float,
    m: int,
    demand_matrix: gdf,
    budget: int,
    max_edges: int,
    penalty: float,
    seed_points: gdf,
    grid: gdf,
) -> Dict[int, Population]:
    pop = start_pop
    n = len(pop)
    i = 0
    gens = {}
    assert id(pop) == id(start_pop)
    while i < iterations:  # or until some other convergence criteria
        parents = _reproduction(n, m, pop)
        assert id(parents) != id(pop)
        children = _crossover(n, parents)
        assert id(children) != id(parents)
        mutated = _mutation(children, seed_points, grid, mutation_probability)
        assert id(mutated) != id(children)
        evaluated = _evaluation(mutated, demand_matrix, budget, max_edges, penalty)
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


def _reproduction(n: int, m: int, pop: Population) -> Population:
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
    selected_ids = _select_ids(n, m, pop)
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
        feasible_nodes = _get_shared_cut_nodes(parent_A["graph"], parent_B["graph"])
        if not feasible_nodes:
            continue
        # remove_A, remove_B = mating_pool.pop(id_A), mating_pool.pop(id_B)
        # assert remove_A == parent_A, remove_B == parent_B
        childg_A, childg_B = _inter_crossover(
            parent_A["graph"], parent_B["graph"], feasible_nodes
        )
        child_A: Network = {"demand_covered": 0.0, "graph": childg_A}
        child_B: Network = {"demand_covered": 0.0, "graph": childg_B}
        child_pop[i] = child_A
        child_pop[i + 1] = child_B
        i = i + 2
    return child_pop


def _mutation(
    pop: Population, seed_points: gdf, grid: gdf, mutation_probability=0.3
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
        mutated[nw_id] = _mutate(nw, seed_points, grid) if mutate[nw_id] else nw
        if not nx.is_connected(mutated[nw_id]["graph"]):
            print(f"see network id: {nw_id}")
            print("first plot: pop before mutation")
            bcnwplt.plot_population_dict(
                pop,
                seed_points,
                grid,
                demand_type="absolute",
                save=True,
                output_path="../results/plots/",
                filename=f"connectivity_BEFORE_mutation_nw{nw_id}",
                gen=-99,
            )
            print(f"see network id: {nw_id}")
            print("second plot: pop after mutation")
            bcnwplt.plot_population_dict(
                mutated,
                seed_points,
                grid,
                demand_type="absolute",
                save=True,
                output_path="../results/plots/",
                filename=f"connectivity_AFTER_mutation_nw{nw_id}",
                gen=-100,
            )
    return mutated


def _evaluation(
    pop: Population, demand_matrix, budget, max_edges, penalty
) -> Population:
    """
    1. Calculate the percentage of demand covered for each network in the population.
    2. Choose only the n fittest? (Not now - For now, let's just do the above)
    """
    return _recalculcate_fitness(pop, demand_matrix, budget, max_edges, penalty)


def _selection(pop: Population, n: int) -> Population:
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
    selected_ids = RNG.choice(nw_ids, size=n, replace=False).tolist()
    selected: Population = {
        new_id: pop[old_id] for new_id, old_id in enumerate(selected_ids)
    }
    return selected


# REPRODUCTION HELPERS


def _sum_demand_coverage(pop: Population) -> float:
    return fsum([nw.get("demand_covered") for nw in pop.values()])


def _get_probabilities(pop: Population) -> np.ndarray:
    demand_sum = _sum_demand_coverage(pop)
    demand_shares = [nw.get("demand_covered") * 100 / demand_sum for nw in pop.values()]
    return _normalise(demand_shares)


def _select_ids(n: int, m: int, pop: Population) -> List[int]:
    demand_shares = _get_probabilities(pop)
    nw_ids = [id for id in pop]
    selected_ids = RNG.choice(nw_ids, size=n * m, replace=True, p=demand_shares)
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


def _decompose_at(v: str, G: Graph) -> List[Graph]:
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
    disconnected = G.copy()
    disconnected.remove_node(v)
    connected_components = nx.connected_components(disconnected)
    return [G.subgraph(comp | {v}).copy() for comp in connected_components]
    # shouldn't use copy() for subgraph views - create them with a function instead...
    # see networkx docs for subgraph


def _inter_crossover(
    graph_A: Graph, graph_B: Graph, cut_nodes: Set[str]
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
    cut_node = str(RNG.choice(sorted(cut_nodes)))
    components_A = _decompose_at(cut_node, graph_A)
    components_B = _decompose_at(cut_node, graph_B)
    # i = RNG.integers(0, len(components_A))
    # j = RNG.integers(0, len(components_B))
    # stripped_A = nx.compose_all(
    #     components_A[:i] + components_A[i + 1 :]
    # )  # remove ith comp
    # stripped_B = nx.compose_all(
    #     components_B[:j] + components_B[j + 1 :]
    # )  # remove jth comp
    # childg_A = nx.compose(stripped_A, components_B[j])  # graph_A - ith comp + jth comp
    # childg_B = nx.compose(stripped_B, components_A[i])  # graph_B - jth comp + ith comp
    comp_a, comp_b = _choose_similar_components(components_A, components_B)
    stripped_A = nx.compose_all([c for c in components_A if c is not comp_a])
    stripped_B = nx.compose_all([c for c in components_B if c is not comp_b])
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
    cut_node = str(RNG.choice(sorted(shared_nodes)))

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


def _get_shared_cut_nodes(graph_A: Graph, graph_B: Graph) -> Set[str]:
    shared_nodes = _get_shared_nodes(graph_A, graph_B)
    cut_nodes_A = set(nx.articulation_points(graph_A))
    cut_nodes_B = set(nx.articulation_points(graph_B))
    return shared_nodes & cut_nodes_A & cut_nodes_B  # set intersection


def _get_shared_nodes(graph_A: Graph, graph_B: Graph) -> Set[str]:
    """
    Finds and returns the shared nodes between two networks.

    Parameters:
    nw_A (Graph): First network's graph.
    nw_B (Graph): Second network's graph.

    Returns:
    A set of shared nodes between the two networks if any exist, otherwise None.
    """
    # use set intersection
    return set(graph_A.nodes) & set(graph_B.nodes)


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


def _find_edge(edge: Tuple[str, str], nw: Network) -> Tuple[int, int]:
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
                return (path_id, u_idx)
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


def _mutate(nw: Network, seed_points: gdf, grid: gdf) -> Network:
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
    action = str(RNG.choice(["add", "remove", "replace"], size=1)[0])
    match action:
        case "add":
            new_nw = _add_edge(nw, seed_points)
            assert id(nw) != id(new_nw)
        case "remove":
            new_nw = _remove_edge(nw, seed_points, grid)
            assert id(nw) != id(new_nw)
        case "replace":
            new_nw = _replace_edge(nw, seed_points, grid)
            assert id(nw) != id(new_nw)
        case _:
            raise NotImplementedError
    assert nx.is_connected(new_nw["graph"])
    return new_nw


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
    nodes = nw["graph"].nodes
    chosen_node = ""
    node_pool = {}
    while len(node_pool) == 0:
        chosen_node = str(RNG.choice(nodes))  # type: ignore
        neighbours = set(
            seed_points.loc[seed_points["id"] == chosen_node, "neighbours"]
            .values[0]
            .values()  # type: ignore
        )
        node_pool = neighbours - nodes  # set difference
    new_node = str(RNG.choice(sorted(node_pool)))
    assert new_node not in nw["graph"]
    mutated_nw = copy_network(nw)
    assert id(nw) != id(mutated_nw)
    mutated_nw["graph"].add_edge(chosen_node, new_node)
    assert id(nw) != id(mutated_nw)
    return mutated_nw


def _remove_edge(nw: Network, seed_points, grid) -> Network:
    """
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
    edges = nw["graph"].edges
    bridges = set(nx.bridges(nw["graph"]))
    edge_pool = edges - bridges  # set difference
    mutated_nw = copy_network(nw)
    if len(edge_pool) == 0:
        return mutated_nw
    fst, snd = tuple(RNG.choice(sorted(edge_pool)))  # type: ignore
    u, v = str(fst), str(snd)
    assert id(nw) != id(mutated_nw)
    mutated_nw["graph"].remove_edge(u, v)
    assert nx.is_connected(mutated_nw["graph"])
    if not nx.is_connected(mutated_nw["graph"]):
        print(f"nw: {nw}")
        print(f"mutated_nw: {mutated_nw}")
        bcnwplt.plot_network(nw, seed_points, grid)
        bcnwplt.plot_network(mutated_nw, seed_points, grid)
    return mutated_nw


def _replace_edge(nw: Network, seed_points: gdf, grid) -> Network:
    """ """
    new_nw = _remove_edge(nw, seed_points, grid)
    return _add_edge(new_nw, seed_points)


# EVALUATION HELPERS


# TODO "demand_covered" in the data structure should be called "fitness" instead
def _recalculcate_fitness(
    pop: Population,
    demand_matrix: gdf,
    budget: int,
    max_edges: int,
    penalty_weight: float,
) -> Population:
    """
    Calculates the fitness function for every network in a population.
    Fitness function is demand_coverage, but if a network is over the budget
    of max number of edges, then the score is penalised. Solutions that cross
    the budget more are also penalised harder.
    """
    max_overspend = max_edges - budget
    new_pop = copy_pop(pop)
    for nw in new_pop.values():
        demand_covered = dmnd.network_demand_coverage(nw["graph"].nodes, demand_matrix)
        cost = nw["graph"].number_of_edges()
        if cost > budget:
            overspend_pct = ((cost - budget) / max_overspend) * 100
            nw["demand_covered"] = max(
                0.0, demand_covered - (penalty_weight * overspend_pct)
            )
        else:
            nw["demand_covered"] = demand_covered
    return new_pop


# OTHER HELPERS


def json_to_pop(data) -> Population:
    pop: Population = {
        int(nw_id): {
            "demand_covered": nw["demand_covered"],
            "graph": nx.Graph(bicycle_network.graph.get_edge_list(nw["paths"])),
        }
        for nw_id, nw in data.items()
    }
    return pop


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
