from layers.attention_layer import * 
from layers.chebyshev_layer import * 
from layers.gcn_layer import * 
from models.GAT import * 
from models.GCN import * 
from models.GNN import * 
from models.Yen import * 
from utils import * 

import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import networkx as nx
import heapq
from typing import Type, Tuple, List, Any, Optional

def find_top_k_paths_unified(
    model_class: Type[nn.Module],
    adj_matrix: torch.Tensor,
    feature_matrix: torch.Tensor,
    source: int,
    target: int,
    K: int,
    hidden_dim: int = 16,
    output_dim: int = 1,
    dropout: float = 0.1,
    normalization: str = "sym",
    num_layers: int = 2,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    device: Union[str, torch.device] = 'cpu',
    chebyshev_k: Optional[int] = None,
    residual: bool = True,
    **model_kwargs: Any
) -> Tuple[List[Tuple[float, List[int]]], float]:
    """
    Unified function to find top K shortest paths using different models (GCN, Chebyshev GCN, GNN, GAT).

    Parameters
    ----------
    model_class : Type[nn.Module]
        The class of the model to be used (e.g., GCN, Chebyshev GCN, GNN, GAT).
    adj_matrix : torch.Tensor
        Adjacency matrix of shape [num_nodes, num_nodes].
    feature_matrix : torch.Tensor
        Feature matrix of shape [num_nodes, input_dim].
    source : int
        Source node index.
    target : int
        Target node index.
    K : int
        Number of shortest paths to find.
    hidden_dim : int, optional
        Dimensionality of hidden layers, by default 16.
    output_dim : int, optional
        Dimensionality of output features, by default 1.
    dropout : float, optional
        Dropout rate, by default 0.1.
    normalization : str, optional
        Type of normalization ('sym' for symmetric, 'rw' for random walk), by default "sym".
    num_layers : int, optional
        Number of GCN layers, must be at least 2, by default 2.
    num_epochs : int, optional
        Number of training epochs, by default 100.
    learning_rate : float, optional
        Learning rate for the optimizer, by default 0.01.
    device : Union[str, torch.device], optional
        Device to run the computations on ('cpu' or 'cuda'), by default 'cpu'.
    chebyshev_k : Optional[int], optional
        Chebyshev polynomial order for Chebyshev GCN, by default None.
    residual : bool, optional
        Whether to use residual connections, by default True.
    **model_kwargs : Any
        Additional keyword arguments for the model initialization.

    Returns
    -------
    Tuple[List[Tuple[float, List[int]]], float]
        A tuple containing the list of top K paths with their total costs and the execution time in seconds.
    """
    start_time = time.time()

    # Move tensors to device
    adj_matrix = adj_matrix.to(device)
    feature_matrix = feature_matrix.to(device)

    # Ensure adjacency matrix is sparse
    if not adj_matrix.is_sparse:
        adj_matrix = adj_matrix.to_sparse()

    num_nodes, input_dim = feature_matrix.shape

    # Determine if Chebyshev GCN is used
    use_chebyshev = chebyshev_k is not None

    # Compute normalized Laplacian if using Chebyshev GCN
    if use_chebyshev:
        laplacian = compute_normalized_laplacian(adj_matrix)
    else:
        laplacian = None  # Not used for standard GCN or GNN

    # Initialize the model
    if use_chebyshev:
        model = model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_hidden_layers=num_layers - 2 if num_layers > 2 else 0,
            dropout=dropout,
            residual=residual,
            k=chebyshev_k,
            **model_kwargs
        ).to(device)
    else:
        model = model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            residual=residual,
            # normalization=normalization, # add this to kwargs since GAT
            num_layers=num_layers,
            **model_kwargs
        ).to(device)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Compute ground truth distances using Dijkstra's algorithm
    distances = dijkstra_shortest_path_lengths(adj_matrix.to_dense().cpu(), target).to(device).unsqueeze(1)

    # Replace 'inf' with a large number for training purposes
    max_distance = distances[distances != float('inf')].max().item()
    distances[distances == float('inf')] = max_distance * 2

    # Training Loop
    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        if use_chebyshev:
            outputs = model(feature_matrix, laplacian)
        else:
            outputs = model(adj_matrix, feature_matrix)
        loss = path_based_loss(outputs, distances, adj_matrix)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if epoch % max(1, num_epochs // 10) == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    # Find top K paths using the trained model
    top_k_paths = find_top_k_paths(model, adj_matrix,
                                             feature_matrix, source,
                                             target, K, device, use_chebyshev,
                                             laplacian=laplacian)

    execution_time = time.time() - start_time
    return top_k_paths, execution_time


def find_top_k_paths(
    model: nn.Module,
    adjacency_matrix: torch.Tensor,
    feature_matrix: torch.Tensor,
    source_node: int,
    target_node: int,
    K: int,
    device: Union[str, torch.device] = 'cpu',
    use_chebyshev: bool = False,
    laplacian: Optional[torch.sparse_coo_tensor] = None
) -> List[Tuple[float, List[int]]]:
    """
    Finds the top K shortest paths from source_node to target_node using the trained model.

    Parameters
    ----------
    model : nn.Module
        The trained model used to compute node scores.
    adjacency_matrix : torch.Tensor
        Adjacency matrix of shape [num_nodes, num_nodes].
    feature_matrix : torch.Tensor
        Feature matrix of shape [num_nodes, input_dim].
    source_node : int
        Source node index.
    target_node : int
        Target node index.
    K : int
        Number of shortest paths to find.
    device : Union[str, torch.device], optional
        Device to run the computations on ('cpu' or 'cuda'), by default 'cpu'.
    use_chebyshev : bool, optional
        Whether Chebyshev GCN is used, by default False.
    laplacian : Optional[torch.sparse_coo_tensor], optional
        Scaled Laplacian matrix, required if use_chebyshev is True, by default None.

    Returns
    -------
    List[Tuple[float, List[int]]]
        List of tuples where each tuple contains the total cost and the corresponding path as a list of node indices.
    """
    model.eval()
    with torch.no_grad():
        if use_chebyshev:
            node_scores = model(feature_matrix.to(device), laplacian.to(device)).cpu().numpy().flatten()
        else:
            node_scores = model(adjacency_matrix.to(device), feature_matrix.to(device)).cpu().numpy().flatten()

    num_nodes = adjacency_matrix.size(0)
    adjacency_dense = adjacency_matrix.to_dense().cpu().numpy()

    # Combine edge weights with model's node scores to adjust edge weights
    adjusted_edge_weights = np.zeros_like(adjacency_dense)
    for u in range(num_nodes):
        for v in range(num_nodes):
            if adjacency_dense[u, v] != 0:
                # Adjust edge weight using node_scores and original edge weight
                adjusted_edge_weights[u, v] = adjacency_dense[u, v] / max(node_scores[v], 0.001)

    # find the top K shortest paths based on adjusted edge weights
    top_k_paths = find_k_shortest_paths_custom(adjusted_edge_weights, source_node, target_node, K)

    return top_k_paths

def find_k_shortest_paths_custom(
    edge_weights: np.ndarray, 
    source: int, 
    target: int, 
    K: int
) -> List[Tuple[float, List[int]]]:
    """
    Custom implementation to find K shortest paths in a weighted graph.

    Parameters
    ----------
    edge_weights : np.ndarray
        Weighted adjacency matrix of shape [num_nodes, num_nodes].
    source : int
        Source node index.
    target : int
        Target node index.
    K : int
        Number of shortest paths to find.

    Returns
    -------
    List[Tuple[float, List[int]]]
        List of tuples where each tuple contains the total cost and the corresponding path as a list of node indices.
    """
    num_nodes = edge_weights.shape[0]
    paths = []
    queue = []
    heapq.heappush(queue, (0, [source]))

    visited_paths = set()

    while queue and len(paths) < K:
        cumulative_cost, path = heapq.heappop(queue)
        current_node = path[-1]

        if current_node == target:
            path_tuple = tuple(path)
            if path_tuple not in visited_paths:
                paths.append((cumulative_cost, path))
                visited_paths.add(path_tuple)
            continue

        for neighbor in np.where(edge_weights[current_node] > 0)[0]:
            if neighbor not in path:
                edge_weight = edge_weights[current_node, neighbor]
                new_cumulative_cost = cumulative_cost + edge_weight
                new_path = path + [neighbor]
                heapq.heappush(queue, (new_cumulative_cost, new_path))

    return paths



