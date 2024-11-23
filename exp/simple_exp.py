import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from find_path import find_top_k_paths_unified
from models.GAT import GAT 
from models.GCN import StackGCN, ChebNetGCN
from models.GNN import StackGNN
from models.Yen import yen_top_k_shortest_paths
from utils import *
import argparse

def run_simple_experiment(config: ml_collections.ConfigDict) -> tuple:
    """
    Runs the experiment based on the provided configuration.

    Parameters
    ----------
    config : ml_collections.ConfigDict
        Configuration dictionary containing hyperparameters and settings.

    Returns
    -------
    tuple
        A tuple containing two lists of top K paths - one from the model and one using Yen's algorithm.
    """
    # Set random seeds
    seed = config.get('seed', 123)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load matrices based on the graph and feature matrix specified in the config
    adjacency_matrix_np, feature_matrix_np = load_matrices(config.graph_name, config.feature_matrix)

    # Convert matrices to appropriate formats
    adjacency_matrix = torch.tensor(adjacency_matrix_np, dtype=torch.float32).to_sparse()
    feature_matrix = torch.tensor(feature_matrix_np, dtype=torch.float32)

    # Extract parameters from config
    source_node = config.source_node
    target_node = config.target_node
    K = config.K

    hidden_dim = config.hidden_dim
    output_dim = config.output_dim
    dropout = config.dropout
    normalization = config.get('normalization', None)
    num_layers = config.num_layers
    n_heads = config.get('n_heads', 4)
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    device = config.device
    chebyshev_k = config.get('chebyshev_k', None)
    residual = config.get('residual', True)
    concat = config.get('concat', True)

    # Choose the model type based on config
    model_type = config.model_type
    print(f"Running experiment with {model_type}...\n")

    # Map model names to model classes
    model_classes = {
        'Chebyshev GCN': ChebNetGCN,
        'Standard GCN': StackGCN,
        'GAT': GAT,
        'GNN': StackGNN,
    }
    model_class = model_classes[model_type]

    # Prepare additional model kwargs
    model_kwargs = {}
    if model_type == 'GAT':
        model_kwargs.update({'n_heads': n_heads, 'concat': concat, 'residual': residual})

    # Find Top K Paths using the unified function
    top_paths, _ = find_top_k_paths_unified(
        model_class=model_class,
        adj_matrix=adjacency_matrix,
        feature_matrix=feature_matrix,
        source=source_node,
        target=target_node,
        K=K,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        normalization=normalization,
        num_layers=num_layers,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        chebyshev_k=chebyshev_k,
        **model_kwargs
    )

    # Display the Top K Paths
    print("\nTop K Paths (Model):")
    for idx, (cost, path) in enumerate(top_paths, 1):
        print(f"Path {idx}: Cost = {cost:.4f}, Path = {path}")

    # Compute Top K Paths using Yen's algorithm for comparison
    top_paths_yen = yen_top_k_shortest_paths(adjacency_matrix_np, source_node, target_node, K)
    print("\nTop K Paths (Yen's Algorithm):")
    for idx, (cost, path) in enumerate(top_paths_yen, 1):
        print(f"Path {idx}: Cost = {cost:.4f}, Path = {path}")

    # Return results
    return top_paths, top_paths_yen


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a graph experiment with a configurable dataset.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--graph_name", type=str, required=False, help="Name of the graph to use (e.g., 'graph1', 'graph2', 'graph3').")
    parser.add_argument("--feature_matrix", type=str, required=False, help="Name of the feature matrix to use (e.g., 'default').")
    args = parser.parse_args()

    # Load configuration file
    config = load_config(args.config_path)

    # Override graph and feature matrix paths if provided
    if args.graph_name:
        config.graph_name = args.graph_name
    if args.feature_matrix:
        config.feature_matrix = args.feature_matrix

    # Run the experiment
    run_simple_experiment(config)