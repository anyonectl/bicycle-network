from math import fsum
from typing import Dict, List, Set, Tuple, TypedDict
import numpy as np

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


# MAIN


def run(start_pop: Population, iterations: int) -> Population:
    return _genetic_algorithm(start_pop, iterations)
    # should the last population be returned to the user, or something else?


# ORCHESTRATOR


def _genetic_algorithm(start_pop: Population, iterations: int) -> Population:
    pop = start_pop
    n = len(pop)
    i = 0
    while i < iterations:  # or until some other convergence criteria
        parents = _selection(n, pop)
        children = _crossover(n, parents)
        mutated = _mutation(children)
        evaluated = _evaluation(mutated)
        pop = evaluated
        i = i + 1
    return pop


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
    parent_ids = list(mating_pool.keys())
    pop: Population = {}
    i = 0
    while i < n:
        id_A, id_B = _choose_parents(parent_ids)
        parent_A = mating_pool[id_A]
        parent_B = mating_pool[id_B]
        shared_nodes = _get_shared_nodes(parent_A, parent_B)
        if shared_nodes is None:
            continue
        # remove parents from mating_pool? maybe not?
        child_A, child_B = _inter_intra_crossover(parent_A, parent_B, shared_nodes)
        pop[i] = child_A
        pop[i + 1] = child_B
        i = i + 2
    return pop


def _mutation(pop: Population) -> Population:
    """
    Randomly add, remove or replace an edge

    For each ??? (inspired by Mesbah 2012):
    1. Determine the possible edge additions and removals that ensure connectedness
    2. Randomly choose 1 action from [add|remove|replace]
    3. Randomly choose 1 edge (from possible choices in step 1) to [add|remove|replace]
    """
    return pop


def _evaluation(pop: Population) -> Population:
    """
    Calculates the percentage of demand covered for each network in the population.
    """
    return pop


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
    return (chosen_ids[0], chosen_ids[1])


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

    # Exchange
    new_A, new_B = _exchange_path_sections(cut_idx_A, cut_idx_B, path_A, path_B)

    # Put new paths into networks
    parent_A["paths"][path_id_A] = new_A
    parent_B["paths"][path_id_B] = new_B

    return (parent_A, parent_B)


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


def _mutate():
    """
    Randomly add, remove or replace an edge
    """
    return


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
