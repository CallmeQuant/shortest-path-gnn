import torch
import torch.nn.functional as F
import numpy as np
import random
import networkx as nx
import folium
from torch import Tensor
from typing import Dict, Tuple, List, Optional
import ml_collections
import yaml
import os 
import warnings

def compute_normalized_laplacian(adj_matrix: torch.Tensor) -> torch.sparse.FloatTensor:
    """
    Computes the normalized Laplacian matrix.

    The normalized Laplacian is defined as:
        L = I - D^{-0.5} * A * D^{-0.5}
    where:
        - I is the identity matrix.
        - A is the adjacency matrix.
        - D is the degree matrix.

    Parameters
    ----------
    adj_matrix : torch.Tensor
        Adjacency matrix of shape [num_nodes, num_nodes].

    Returns
    -------
    torch.sparse.FloatTensor
        Normalized Laplacian matrix of shape [num_nodes, num_nodes].

    Raises
    ------
    ValueError
        If the adjacency matrix is not square.
    """
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    # Degree matrix
    degree = torch.sparse.sum(adj_matrix, dim=1).to_dense()  # Shape: [num_nodes]
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

    # D^{-0.5} as a diagonal matrix
    D_inv_sqrt = torch.diag(degree_inv_sqrt)  # Shape: [num_nodes, num_nodes]

    # Convert sparse adjacency to dense for multiplication
    adj_dense = adj_matrix.to_dense()  # Shape: [num_nodes, num_nodes]

    # Normalized Laplacian: I - D^{-0.5} * A * D^{-0.5}
    num_nodes = adj_matrix.shape[0]
    identity = torch.eye(num_nodes, device=adj_matrix.device)
    normalized_laplacian = identity - D_inv_sqrt @ adj_dense @ D_inv_sqrt

    # Convert back to sparse
    return normalized_laplacian.to_sparse()


def check_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Checks if a NumPy array is symmetric within a tolerance.

    Parameters
    ----------
    a : np.ndarray
        Input array to check for symmetry.
    rtol : float, optional
        Relative tolerance parameter (default is 1e-05).
    atol : float, optional
        Absolute tolerance parameter (default is 1e-08).

    Returns
    -------
    bool
        `True` if the array is symmetric within the specified tolerances, `False` otherwise.
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def find_node_index(node_id_place_name: Dict[int, str],
                    node_to_index: Dict[int, int],
                    value: str) -> Optional[int]:
    """
    Finds the node index corresponding to a given place name.

    Parameters
    ----------
    node_id_place_name : Dict[int, str]
        Mapping from node IDs to place names.
    node_to_index : Dict[int, int]
        Mapping from node IDs to node indices.
    value : str
        The place name to search for.

    Returns
    -------
    Optional[int]
        The node index if found, otherwise `None`.
    """
    for node_id, place_name in node_id_place_name.items():
        if place_name == value:
            return node_to_index.get(node_id)
    return None


def visualize_k_shortest_paths(
    k_shortest_paths: List[Tuple[float, List[int]]],
    index_to_node_id: Dict[int, int],
    node_id_to_coords: Dict[int, Tuple[float, float]],
    node_id_to_name: Dict[int, str],
    map_html: str = 'k_shortest_paths_map.html'
) -> None:
    """
    Visualizes the K-shortest paths on an interactive Folium map.

    Parameters
    ----------
    k_shortest_paths : List[Tuple[float, List[int]]]
        List of tuples where each tuple contains the cost and the list of node indices representing a path.
    index_to_node_id : Dict[int, int]
        Mapping from node indices to node IDs.
    node_id_to_coords : Dict[int, Tuple[float, float]]
        Mapping from node IDs to their (latitude, longitude) coordinates.
    node_id_to_name : Dict[int, str]
        Mapping from node IDs to place names.
    map_html : str, optional
        Output HTML file name for the map (default is 'k_shortest_paths_map.html').

    Returns
    -------
    None
        Saves the interactive map to the specified HTML file.
    """
    print("Visualizing K-shortest paths...")

    if not k_shortest_paths:
        print("No paths to visualize.")
        return

    # Initialize Folium map centered around the first node in the first path
    first_index = k_shortest_paths[0][1][0]
    first_node_id = index_to_node_id.get(first_index)
    first_coords = node_id_to_coords.get(first_node_id, (0.0, 0.0))
    m = folium.Map(location=first_coords, zoom_start=15)

    # Define colors for different paths
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for idx, (cost, path_indices) in enumerate(k_shortest_paths):
        # Convert node indices to node IDs
        node_ids = [index_to_node_id.get(index) for index in path_indices]

        # Extract latitude and longitude for each node in the path
        path_coords = [node_id_to_coords.get(node_id, (0.0, 0.0)) for node_id in node_ids]

        # Add polyline to the map
        folium.PolyLine(
            locations=path_coords,
            color=colors[idx % len(colors)],
            weight=5,
            opacity=0.7,
            popup=f'Path {idx+1}: Cost {cost:.2f}'
        ).add_to(m)

    # Optionally, add markers for all start and end points
    for idx, (cost, path_indices) in enumerate(k_shortest_paths):
        node_ids = [index_to_node_id.get(index) for index in path_indices]
        start_node_id = node_ids[0]
        end_node_id = node_ids[-1]
        start_coords = node_id_to_coords.get(start_node_id, (0.0, 0.0))
        end_coords = node_id_to_coords.get(end_node_id, (0.0, 0.0))

        # Start point
        folium.Marker(
            location=start_coords,
            popup=f"Start: {node_id_to_name.get(start_node_id, 'Start Node')}",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)

        # End point
        folium.Marker(
            location=end_coords,
            popup=f"End: {node_id_to_name.get(end_node_id, 'End Node')}",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)

    # Save the map to an HTML file
    m.save(map_html)
    print(f"K-shortest paths map saved as '{map_html}'. Open it in a web browser to view.")


def generate_random_graph(
    num_nodes: int,
    directed: bool = False,
    edge_prob: float = 0.3,
    weight_range: Tuple[float, float] = (1, 10)
) -> np.ndarray:
    """
    Generates a random graph represented as an adjacency matrix.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    directed : bool, optional
        Whether the graph is directed (default is `False`).
    edge_prob : float, optional
        Probability of edge creation between nodes (default is 0.3).
    weight_range : Tuple[float, float], optional
        Range of edge weights (default is (1, 10)).

    Returns
    -------
    np.ndarray
        Adjacency matrix of shape [num_nodes, num_nodes].

    Raises
    ------
    ValueError
        If `num_nodes` is not a positive integer or `edge_prob` is not in [0, 1].
    """
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be a positive integer.")
    if not (0 <= edge_prob <= 1):
        raise ValueError("Edge probability must be between 0 and 1.")

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue  # No self-loops
            if random.random() < edge_prob:
                weight = random.uniform(*weight_range)
                adj_matrix[i][j] = weight
                if not directed:
                    adj_matrix[j][i] = weight
    return adj_matrix


def generate_connected_random_graph(
    num_nodes: int,
    directed: bool = False,
    edge_prob: float = 0.3,
    weight_range: Tuple[float, float] = (1, 10)
) -> np.ndarray:
    """
    Generates a connected random graph represented as an adjacency matrix.

    Ensures that the generated graph is connected by first creating a random spanning tree,
    then adding additional random edges based on the specified probability.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    directed : bool, optional
        Whether the graph is directed (default is `False`).
    edge_prob : float, optional
        Probability of additional edge creation between nodes (default is 0.3).
    weight_range : Tuple[float, float], optional
        Range of edge weights (default is (1, 10)).

    Returns
    -------
    np.ndarray
        Adjacency matrix of shape [num_nodes, num_nodes].

    Raises
    ------
    ValueError
        If `num_nodes` is less than 1 or `edge_prob` is not in [0, 1].
    """
    if num_nodes < 1:
        raise ValueError("Number of nodes must be at least 1.")
    if not (0 <= edge_prob <= 1):
        raise ValueError("Edge probability must be between 0 and 1.")

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Ensure connectivity by creating a random spanning tree
    for i in range(1, num_nodes):
        j = random.randint(0, i - 1)
        weight = random.uniform(*weight_range)
        adj_matrix[i][j] = weight
        if not directed:
            adj_matrix[j][i] = weight

    # Add additional random edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                weight = random.uniform(*weight_range)
                adj_matrix[i][j] = weight
                if not directed:
                    adj_matrix[j][i] = weight

    return adj_matrix


def set_random_seeds(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Seed value (default is 42).

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed}.")


def path_based_loss(
    outputs: torch.Tensor, 
    true_distances: torch.Tensor, 
    adj_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Computes the path-based loss combining MSE loss and path validity loss.

    Parameters
    ----------
    outputs : torch.Tensor
        Predicted distances of shape [num_nodes, 1].
    true_distances : torch.Tensor
        Ground truth distances of shape [num_nodes, 1].
    adj_matrix : torch.Tensor
        Adjacency matrix of shape [num_nodes, num_nodes].

    Returns
    -------
    torch.Tensor
        Combined loss value.
    """
    outputs = outputs.squeeze()
    true_distances = true_distances.squeeze()

    mse_loss = F.mse_loss(outputs, true_distances)

    # Path validity loss
    adj_indices = adj_matrix.indices()
    adj_values = adj_matrix.values()
    i = adj_indices[0]
    j = adj_indices[1]
    edge_weight = adj_values

    edge_losses = F.relu(outputs[i] - outputs[j] - edge_weight)
    path_loss = edge_losses.sum()

    return mse_loss + 0.1 * path_loss


def dijkstra_shortest_path_lengths(
    adj_matrix: torch.Tensor, 
    target: int
) -> torch.Tensor:
    """
    Computes the shortest path lengths from all nodes to the target node using Dijkstra's algorithm with NetworkX.

    Parameters
    ----------
    adj_matrix : torch.Tensor
        Adjacency matrix of shape [num_nodes, num_nodes] with edge weights.
    target : int
        The index of the target node.

    Returns
    -------
    torch.Tensor
        Tensor of shape [num_nodes] containing shortest path lengths.
    """
    G = nx.from_numpy_array(adj_matrix.cpu().numpy())

    path_lengths = nx.single_target_shortest_path_length(G, target)

    num_nodes = adj_matrix.shape[0]
    distances = torch.full((num_nodes,), float('inf'))

    for node, length in path_lengths:
        distances[node] = length

    return distances


def load_config(file_dir: str) -> ml_collections.ConfigDict:
    """
    Loads the configuration using ml_collections.ConfigDict from a YAML file.

    Parameters
    ----------
    file_dir : str
        Directory path to the configuration YAML file.

    Returns
    -------
    ml_collections.ConfigDict
        Configuration dictionary loaded from YAML.
    """
    with open(file_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    return config

def load_matrices(graph_name: str, feature_matrix_name: str):
    """
    Loads adjacency and feature matrices from given file paths.

    Parameters
    ----------
    graph_name : str
        Name of the graph (used to determine adjacency matrix file).
    feature_matrix_name : str
        Name of the feature matrix (used to determine feature matrix file).

    Returns
    -------
    tuple
        A tuple containing the adjacency matrix and feature matrix as numpy arrays.
    """
    # Construct file names from graph name and feature matrix name located in the "data" folder
    base_dir = "data"
    adjacency_path = os.path.join(base_dir, f"{graph_name}_adjacency_matrix.npy")
    feature_matrix_path = os.path.join(base_dir, f"{feature_matrix_name}_feature_matrix.npy")

    # Load the matrices
    adjacency_matrix = np.load(adjacency_path)
    feature_matrix = np.load(feature_matrix_path)

    return adjacency_matrix, feature_matrix
