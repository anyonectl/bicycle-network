# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: bicycle-network:default (3.14.8)
#     language: python
#     name: python3
# ---

# %%
import geohexgrid as ghg
import numpy as np
from shapely import Polygon, LineString, LinearRing, Point, MultiPoint, GeometryCollection
import geopandas as gpd
import pandas as pd
from math import comb
import numpy as np
import matplotlib.pyplot as plt
import collections
from typing import List, Tuple

# %% [markdown]
# ## Get the shape of some city

# %%
d = {'id': ['1'], 'geometry': [Polygon([(0., 0.), (0., 100.), (100., 100.), (100., 0.)])]}
base = gpd.GeoDataFrame(data=d, crs=None)
background = base.plot(figsize=(10, 10))
print(base)

# %% [markdown]
# ## Create hex grid

# %%
R = 4
grid = ghg.make_grid_from_bounds(0., 0., 100., 100., R)
grid.plot(figsize=(10, 10))

# %%
type(grid)

# %%
hex_example = grid.iat[0,-1]
hex_example
print(hex_example)

# %%
print(f"area: {hex_example.area}")
print(f"boundary: {hex_example.boundary}")
print(f"bounds: {hex_example.bounds}")
print(f"centroid: {hex_example.centroid}")

# %% [markdown]
# ## Extract seed points

# %%
grid['centroid'] = grid['geometry'].centroid
grid.head()

# %%
centroids = grid["centroid"]
centroids.plot(figsize=(10, 10))


# %% [markdown]
# ## Create demand matrix

# %% [markdown]
# ### Compute seed point pairs

# %%
def od_pairs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    seed_points = gdf['cell_id'].values.tolist()
    geo = gdf['centroid'].values.tolist()
    n = len(seed_points)
    ids = []
    origins = []
    dests = []
    lines = []
    for i in range(n):
        for j in range(i+1, n):
            # print(f'seed_points[i]: {seed_points[i]}, seed_points[j]: {seed_points[j]}')
            # print(f'geo[i]: {geo[i]}, geo[j]: {geo[j]}')
            ids.append((seed_points[i], seed_points[j]))
            origins.append(seed_points[i])
            dests.append(seed_points[j])
            lines.append(LineString([geo[i], geo[j]]))
    return gpd.GeoDataFrame({'pair': ids, 'u': origins, 'v': dests, 'geometry': lines})

test_pairs = od_pairs(grid)
print(f'no. of seed points: {grid.shape[0]}')
print(f'no. of pairs is actually: {test_pairs.shape[0]}')
print(f'no. of pairs should be: {comb(grid.shape[0], 2)}')
test_pairs.head()
test_pairs.plot(figsize=(10, 10))


# %% [markdown]
# ### Generate random demand

# %%
def generate_normal_demand(mean: float, std: float, n: int) -> np.ndarray:
    seed: int = 8920299146682435931
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    return rng.normal(mean, std, size=n)

# Code from NumPy docs
def plot_normal_demand_distribution(mean: float, std: float, demand: np.ndarray):
    count, bins, _ = plt.hist(demand, 30, density=True)
    plt.plot(bins, 1/(std * np.sqrt(2 * np.pi)) *
                   np.exp( - (bins - mean)**2 / (2 * std**2) ),
             linewidth=2, color='r')
    plt.show()

def generate_uniform_demand(n: int) -> np.ndarray:
    seed: int = 8920299146682435931
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    return rng.uniform(low=0.0, high=1.0, size=n)

def make_demand_weird(demand: np.ndarray) -> np.ndarray:
    seed: int = 8920299146682435931
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    n = len(demand)
    indices = [x for x in range(n)] 
    doubles = rng.choice(a=indices, size=int(n/4), replace=False)
    halves = rng.choice(a=indices, size=int(n/4), replace=False)
    for i in doubles:
        demand[i] = demand[i] * 2
    for i in halves:
        demand[i] = demand[i] / 1.5
    return demand

# Code from NumPy docs
def plot_demand_distribution(demand: np.ndarray):
    count, bins, _ = plt.hist(demand, 15, density=True)
    plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    plt.show()

#TODO: MAKE THE DEMAND DISTRIBUTION WEIRD
#for report: demand could also be made proportional to popylation density. but for now this is our proxy 


# %%
P = test_pairs.shape[0] # number of seed point pairs
mean: float = 1.0
std: float = 0.2
test_demand = make_demand_weird(generate_uniform_demand(P))
print(type(test_demand))
print(len(test_demand))
print(len(test_demand))
test_demand

# %%
plot_demand_distribution(test_demand)

# %%
demand_matrix = pd.concat([test_pairs, pd.Series(test_demand)], axis=1)
demand_matrix.columns = ['pair', 'u', 'v', 'geometry', 'demand']
demand_matrix.head()
demand_matrix.info()

# %%
ordered_demand = demand_matrix.sort_values(by=['demand', 'pair'], ascending=False)
ordered_demand.head()

# %% [markdown]
# ## Calculate total demand

# %%
total_demand = float(demand_matrix['demand'].sum())
total_demand

# %%
demand_matrix["demand_share"] = demand_matrix["demand"] * 100 / total_demand

# %%
# With Anastassia's plot:

fig, ax = plt.subplots(1,1, figsize = (10,10))
grid.boundary.plot(ax=ax, color = "black", zorder = 0, lw = 0.5)
#grid.boundary.plot(column="centroid", ax=ax, alpha=0.5, color = "orange", label="seed points", markersize = 2, zorder = 1)
#test_pairs.plot(column='u', ax=ax, alpha=0.5, color = "blue", label = "origins", markersize = 2, zorder = 1)
#test_pairs.plot(column='v', ax=ax, alpha=0.5, color = "orange", label = "destinations", markersize = 2, zorder = 2)
demand_matrix.plot(ax=ax, lw = demand_matrix.demand_share * 20, alpha = 0.2)
ax.set_axis_off()
ax.set_title("Seed point pairs")


# %%
seed_points = grid
seed_points.columns = ['id', 'geometry', 'centroid']
seed_points.info()

# %%
seed_points["total_demand"] = seed_points.apply(
    lambda x:
        sum(demand_matrix[(demand_matrix.u == x.id) | (demand_matrix.v == x.id)].demand),
    axis = 1
)
seed_points.sort_values('total_demand', ascending=False).head()

# %%

fig, ax = plt.subplots(1,1)
seed_points.plot(
    ax=ax,
    column="total_demand",
    cmap = "Blues",
    legend=True
)
ax.set_axis_off()
ax.set_title("Total demand per hexagon")

# %% [markdown]
# ## Genetic Algorithm

# %% [markdown]
# We currently have two dataframes we are working with: `seed_points` and `demand_matrix`

# %%
N_ITE: int = 100            # iterations
N_POP: int = 20
MAX_PATHS: int = 10
MAX_NODES: int = 15
SEED: int = 3581054352
RNG: np.random.Generator = np.random.default_rng(seed=SEED)


# %% [markdown]
# ### Initial Route Set Generation

# %%
def numeric_id(s: str) -> Tuple[int, int]:
    x, y = s.split(",")
    return int(x), int(y)

def str_id(t: Tuple[int, int]) -> str:
    return f'{t[0]},{t[1]}'


# %% [markdown]
# #### Neighbour nodes and VNS

# %%
# Inspired by Red Blob Games

hex_directions = [
    (1, 0), (1, -1), (0, -1), 
    (-1, 0), (-1, 1), (0, 1)
]

def hex_direction(direction):
    return hex_directions[direction]

def hex_add(a, b):
    return a[0] + b[0], a[1] + b[1]

def hex_neighbour(hex, direction):
    return hex_add(hex, hex_direction(direction))

def axial_to_cube(q, r):
    s = -q-r
    return (q, r, s)

def cube_to_axial(q, r, s):
    assert not (round(q + r + s) != 0), "q + r + s must be 0"
    return (q, r)

def double_to_axial(node: Tuple[int, int]) -> Tuple[int, int]:
    u, v = node
    x, y = ghg.double_to_cartesian(u, v, R)
    return ghg.cartesian_to_axial(x, y, R)

# TODO: should only add neighbours to vns if they are in the dataset, i.e. within the bounding box
def vns(node: Tuple[int, int]) -> List[Tuple[int, int]]:
    q, r = double_to_axial(node)
    vns = []
    for i in range(6):
        a, b = hex_neighbour((q, r), i)
        neighbour = ghg.axial_to_double(a, b)
        vns.append(str_id(neighbour))
    return vns

# SPATIAL INDEXING IN GEOPANDAS. STH WITH TREES


# %%
# Place neighbour nodes in Vicinity Node Set
seed_points["vns"] = seed_points["id"].apply(lambda s: vns(numeric_id(s)))
seed_points.head()

# %%
# Anastassia's code for VNS plot
fig, ax = plt.subplots(1,1, figsize = (4,4))
seed_points.boundary.plot(ax=ax, color = "grey")
for i in ['7,11', '4,4', '1,9']:
    seed_points[seed_points.id==i].plot(ax=ax, color = "orange", alpha = 0.5)
    seed_points[seed_points.id.isin(seed_points.loc[seed_points.id==i, "vns"].values[0])].plot(ax=ax, color = "blue", alpha = 0.5)
ax.set_axis_off()
ax.set_title("Each node has a vicinity set")
plt.show()


# %% [markdown]
# #### Create starting network (Initial Route Set)

# %%
def normalise_probabilities(df):
    p = df.to_numpy(dtype=np.float64)
    return p / np.sum(p, dtype=np.float64)

# TODO: bias terms for each node in the VNS?
# TODO: needs some way to stop if no node in VNS is possible. boolean in irs dict?
def next_node(p: int, previous_node: str, irs: dict):
    # Calculate probabilities for each node in the VNS
    filter = seed_points.loc[seed_points['id'] == previous_node, ['vns']].values.tolist()[0][0]
    vns = seed_points.loc[seed_points['id'].isin(filter)].sort_values(by=['total_demand'], ascending=False)
    activity_sum = float(vns['total_demand'].sum()) 
    vns['probability'] = vns['total_demand'] / activity_sum
    #print(vns['probability'].sum())
    #vns['probability'] = vns['probability'] / np.sum(vns['probability'], dtype=np.float64) # normalise, see: https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
    #print(vns['probability'].sum())

    # Select next node and add to dict
    tries = vns.shape[0]
    next_node = ''
    for _ in range(tries):
        next_node = RNG.choice(a=vns['id'], p=normalise_probabilities(vns['probability']))
        if next_node not in irs[p]['nodes']:
            #print(type(irs[p]['nodes']))
            #print(irs[p]['nodes']) 
            irs[p]['nodes'].append(next_node)
            break
        else:
            vns = vns[(vns['id'] != next_node)]
    finished = vns.shape[0] == 0 # stop if no node in VNS is possible
    return (next_node, irs, finished)

def generate_route(k: int):
    """
    Form Initial Node Set (INS) from the first k nodes. k must be larger than MAX_PATHS
    """
    irs = {}
    ins = seed_points.copy().sort_values("total_demand", ascending=False).head(k).reset_index(drop=True)

    # Select first nodes
    for i in range(MAX_PATHS):
        # Calculate probabilities of selection a node j as first node
        activity_sum = float(ins['total_demand'].sum())
        ins['probability'] = ins['total_demand'] / activity_sum
        #ins['probability'] /= ins['probability'].sum() # normalise, see: https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
    
        # Select first node
        first_node = RNG.choice(a=ins['id'], p=normalise_probabilities(ins['probability']))
    
        # Add to dict
        #u, v = numeric_id(first_node)
        irs[i] = {'nodes': [first_node]}
    
        # Remove selected node from INS
        ins = ins[(ins['id'] != first_node)]

        # Build paths
        previous_node = first_node
        for _ in range(MAX_NODES-1): # -1 because we already added first_node
            tmp_next, tmp_irs, finished = next_node(i, previous_node, irs)
            if finished: # stop if no node in VNS is possible
                break
            previous_node = tmp_next
            irs = tmp_irs

    return irs


# %%
k = int(len(seed_points)/2) # how many start nodes do we initialize our routes with
irs = generate_route(k)
irs

# %%
id_to_centroid = seed_points.set_index("id")["centroid"]
id_to_centroid

# %%
# Create dict for network
d = {
    'path_id': [path_id for path_id in irs],
    'start_id': [path.get('nodes')[:][0] for _, path in irs.items()],
    'end_id': [path.get('nodes')[:][-1] for _, path in irs.items()],
    'path': [path.get('nodes') for _, path in irs.items()],
    #'demand_covered': [],
    #'geometry': []
}

# Flatten list of paths to list of seed points
points = [
    point
    for path in d['path']
    for point in path
]

# TODO: Calculate covered demand


# Create geometries
id_to_centroid = seed_points.set_index("id")["centroid"].to_dict()
geom = [
    LineString([id_to_centroid[point_id] for point_id in path])
    for path in d['path']
]

# Create dataframe
initial_network = gpd.GeoDataFrame(d, geometry=geom, crs=None)

initial_network

# %%
import matplotlib as mpl

cmap = mpl.colormaps.get_cmap('tab10')
colors = cmap(np.linspace(0, 1, len(initial_network)))

fig, ax = plt.subplots(1,1, figsize=(10,10))
#start_nodes.plot(ax=ax, alpha = .1)
seed_points.boundary.plot(ax=ax, color = "grey", lw = 1)
centroids.plot(ax=ax, color = "black", markersize=1)
for path_id in initial_network['path_id']:
    initial_network[initial_network['path_id']==path_id].plot(
        ax=ax,
        lw=5,
        alpha = 0.6,
        label=path_id,
        color=colors[path_id])
ax.legend()
ax.set_title("wip")
ax.set_axis_off()
plt.show()

# %% [markdown]
# # Create initial population of 

# %%

# %%
print(seed_points.info())
seed_points.head()

# %%
print(demand_matrix.info())
demand_matrix.head()

# %%
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
