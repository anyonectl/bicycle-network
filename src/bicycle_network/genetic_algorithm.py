from typing import Dict, List, TypedDict

# Gene: a path consisting of several edges
# Chromosome: a candidate solution (i.e. a network) consisting of genes
# Population: a collection of candidate solutions
# Generation: the population in a given iteration of the genetic algorithm

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
    i = 0
    while i < iterations:  # or until some other convergence criteria
        parents = _selection(pop)
        children = _evolution(parents)
        evaluated = _evaluation(children)
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


def _selection(pop: Population) -> Population:
    """
    Select parent networks to breed the next generation, based on some survival probability.
    """
    return pop


def _evolution(pop: Population) -> Population:
    """
    Create the next population.
    """
    return pop


def _crossover():
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


def _mutation():
    """
    Randomly add, remove or replace an edge

    For each ??? (inspired by Mesbah 2012):
    1. Determine the possible edge additions and removals that ensure connectedness
    2. Randomly choose 1 action from [add|remove|replace]
    3. Randomly choose 1 edge (from possible choices in step 1) to [add|remove|replace]
    """


def _evaluation(pop: Population) -> Population:
    """
    Calculates the percentage of demand covered for each network in the population.
    """
    for c in pop:
        pop[c] = (fitness(N, c, demand) * 100) / total_demand
    return pop


# HELPERS


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
