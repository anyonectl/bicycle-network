from typing import Dict, List, Tuple

# Gene: a path consisting of several edges
# Chromosome: a candidate solution (i.e. a network) consisting of genes
# Population: a collection of candidate solutions
# Generation: the population in a given iteration of the genetic algorithm

### TYPE ALIASES ###

type EdgeList = list[Tuple[int, int]]
type Gene = list[int] # the ints map to bit indexes in a chromosome
type Chromosome = int # each bit is an edge, identified by its bit index in the int
type Population = dict[Chromosome, float] # float for fitness function

### MAIN ###

def run(edges: EdgeList, demand: List[float], iterations: int) -> EdgeList:
    N = len(edges)
    total_demand = sum(demand)
    return _genetic_algorithm(iterations, N, demand, total_demand)
    # should the last population be returned to the user, or something else?

### VISUALISATION ##

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

### ORCHESTRATOR ###

def _genetic_algorithm(iterations: int, N: int, demand: List[float], total_demand: float) -> Population:
    population = _initialise(N)
    for i in range(iterations): # or until some other convergence criteria
        evaluated = _evaluation(N, population, demand, total_demand)
        parents = _selection(evaluated)
        children = _evolution(parents)
        plot_generation(children)
        #_log_generation(children)
        population = children
    return population

### KEY FUNCTIONS ###

def _initialise(N: int) -> Population:
    """
    Generate a random selection of chromosomes for the first generation.
    """

def _evaluation(N: int, pop: Population, demand: List[float], total_demand: float) -> Population:
    """
    Calculates the percentage of demand covered for each chromosome in the population.
    """
    for c in pop:
        pop[c] = (fitness(N, c, demand) * 100) / total_demand
    return pop

def _selection():
    """
    Select parent chromosomes to breed the next generation, based on some survival probability.
    """

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

def _evolution():
    """
    Create the next population.
    """

### HELPERS ###

def fitness(N: int, chromosome: Chromosome, demand: List[float]) -> float:
    acc_demand = 0.0
    for i in range(N):
        if edge_included(i, chromosome):
            acc_demand += demand[i]
    return acc_demand

def edge_included(pos: int, num: int) -> bool:
    mask = 1 << pos
    return (num & mask) != 0

def _mutate():
    """
    Randomly add, remove or replace an edge 
    """

def _reproduce():

#def _log_generation():