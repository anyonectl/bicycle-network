from math import fsum
from typing import Dict, List, Tuple, TypedDict
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
        children = _crossover(parents)
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
    demand_sum = fsum([nw.get("demand_covered") for nw in pop.values()])
    demand_shares = [nw.get("demand_covered") * 100 / demand_sum for nw in pop.values()]
    nw_ids = [id for id in pop]
    selected_ids = RNG.choice(nw_ids, size=n, replace=True, p=demand_shares)
    mating_pool: Population = {id: pop[id] for id in selected_ids}
    return mating_pool


def _evolution(pop: Population) -> Population:
    """
    Create the next population.
    """
    return pop


def _crossover(pop: Population) -> Population:
    """
    Swap genes (sequences of edges) between parent chromosomes to create two child chromosomes.

    For each parent pair (inspired by Mesbah 2012):
    1. Assume that each parent is a connected network
    2. Determine shared vertices (edges) between the parents
    3. Randomly choose 1 shared vertex*
    4. Swap all edges that precede the selected shared vertex
    5. This produces 2 child solutions that are both connected

    *if no shared vertex exists, this parent pair doesn't produce new child solutions
    """
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
    for c in pop:
        pop[c] = (fitness(N, c, demand) * 100) / total_demand
    return pop


# HELPERS


def _choose_parent_pair(pop: Population) -> Tuple[Network, Network]:
    """
    Randomly chooses two parent networks from the mating pool to be
    used for inter-string crossover.

    Params:
    pop (Population): Mating pool to choose parent networks from.

    Returns:
    A pair of parent networks.
    """


def _choose_single_parent(pop: Population) -> Network:
    """
    Randomly chooses a single parent network from the mating pool to be
    used for intra-string crossover.

    Params:
    pop (Population): Mating pool to choose parent network from.

    Returns:
    A single parent network.
    """


def _inter_crossover(parents: Tuple[Network, Network]) -> Tuple[Network, Network]:
    """
    Performs inter-string crossover by exchanging two complete paths
    between two parent networks to create two child networks.

    Params:
    parents (Tuple[Network, Network]): Pair of parents to exchange paths between.

    Returns:
    A pair of child networks.
    """
    # CONNECTIVITY MUST BE ENFORCED


def _intra_crossover(parent: Network) -> Tuple[Network, Network]:
    """
    Performs intra-string crossover by exchanging path sections
    between two paths in the same parent network to create two child
    networks.

    Params:
    parent (Network): Parent network to exchange path sections in.

    Returns:
    A pair of child networks.
    """


def fitness(N: int, nw: Network, demand: List[float]) -> float:
    acc_demand = 0.0
    for i in range(N):
        if edge_included(i, nw):
            acc_demand += demand[i]
    return acc_demand


def edge_included(pos: int, num: int) -> bool:
    mask = 1 << pos
    return (num & mask) != 0


def _mutate():
    """
    Randomly add, remove or replace an edge
    """
    return


def _reproduce():
    return


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
