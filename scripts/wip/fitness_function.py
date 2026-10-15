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

# %% [markdown]
# # Create a fitness function
#
# For all nodes that are on the created network, sum the demand of the nodes (the total demand for a node $i$ is the sum of all demand pairs where $i$ is in the pair).
#
# The the goal of the objective function $Z$ is to maximise the sum of the demand of the included nodes in the network:
#
# $$
# \max Z = \sum_{i=0}^{N}{d_{i}\phi_{i}}
# $$
#
# where $N$ is the number of seed points, $d_i$ is the total demand of seed point $i$, and $\phi_{i}$ is $1$ if seed point $i$ is included in the network, $0$ otherwise.
#
# (Later we will add shortest path distance to the fitness function / as optimisation criterion)
#
# But this probably won't work, since the total demand of a seed point would include demand from edges that are not included in the network... so we could do it with edges (seed point pairs) instead:
#
# The goal of the objective function $Z$ is to maximise the sum of the demand of the edges in the network:
#
# $$
# \max Z = \sum_{i=0}^{N}{d_{i}\phi_{i}}
# $$
#
# where $N$ is the total number of edges (seed point pairs), $d_i$ is the total demand of edge $i$, and $\phi_{i}$ is $1$ if edge $i$ is included in the network, $0$ otherwise.
#

# %%
from typing import List


# %%
def fitness_function(nw: List[int], demand: np.ndarray) -> float:
    """
    Evaluates the fitness of a network.

    Args:
        nw: The network list with indices representing seed point pairs (edges),
            and a binary value representing whether that edge is included
            in the network or not.
        demand: A list of the same length as nw, containing demand per edge.
    Returns:
        A float representing the fitness of the input network.

    """
    d: np.ndarray = normalised_demand(demand)
    total_demand: float = 0.0
    N: int = len(nw)
    for i in range(
        N
    ):  # can maybeoptimise by replacing loop with bit string and doing bit hacks?
        if nw[i] == 1:
            total_demand += d[i]
    return total_demand
