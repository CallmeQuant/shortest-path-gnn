import numpy as np
import networkx as nx
from typing import List, Tuple


def yen_top_k_shortest_paths(adjacency_matrix: np.ndarray, source: int, target: int, K: int) -> list:
    """
    Finds the top K shortest paths from source to target using Yen's algorithm.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Adjacency matrix with edge weights. Shape [num_nodes, num_nodes].
    source : int
        Source node index.
    target : int
        Target node index.
    K : int
        Number of shortest paths to find.

    Returns
    -------
    list
        List of tuples (total_cost, path) where `total_cost` is the cost of the path, and `path` is a list of node indices representing the shortest path.
        If no path is found, returns an empty list.
    """
    # Create a graph from the adjacency matrix using NetworkX
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph())
    for u, v, data in G.edges(data=True):
        data['weight'] = adjacency_matrix[u, v]  # Assign edge weights from adjacency matrix

    try:
        # Find simple shortest paths using Dijkstra's or Yen's algorithm
        paths = nx.shortest_simple_paths(G, source, target, weight='weight')
    except nx.NetworkXNoPath:
        return []  # If no path exists, return an empty list

    top_k_paths = []
    for path in paths:
        # Calculate the total cost of the current path
        total_cost = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        # Append the path and its cost to the list of top K paths
        top_k_paths.append((total_cost, path))
        if len(top_k_paths) == K:
            break  # Stop once we have found K paths

    return top_k_paths
