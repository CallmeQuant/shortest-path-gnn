import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from find_path import find_top_k_paths_unified
from models.GAT import GAT 
from models.GCN import StackGCN, ChebNetGCN
from models.GNN import StackGNN
from models.Yen import yen_top_k_shortest_paths
from utils import * 


import time
import random
import torch
import numpy as np
import pandas as pd
import ml_collections
import yaml
import argparse
import os

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

def run_full_experiment(config: ml_collections.ConfigDict, output_csv_path: str = None):
    """
    Runs the experiment comparing different models on randomly generated graphs.
    Metrics: Accuracy and Execution Time.

    Parameters
    ----------
    config : ml_collections.ConfigDict
        Configuration dictionary containing hyperparameters and settings.
    output_csv_path : str, optional
        Path to save the experiment results as a CSV file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the experiment results.
    """
    # Set random seeds
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    graph_sizes = config.graph_sizes
    graph_types = config.graph_types
    K = config.K

    # Shared parameters for the models
    shared_params = config.shared_params
    results = []

    for graph_type in graph_types:
        for size in graph_sizes:
            print(f"\nGenerating {graph_type} graph with {size} nodes...")
            directed = graph_type == 'directed'

            # Generate a connected random graph
            adj_matrix_np = generate_connected_random_graph(
                num_nodes=size,
                directed=directed,
                edge_prob=0.3,
                weight_range=(1, 10)
            )

            # Convert adjacency matrix to torch sparse tensor
            adj_matrix = torch.from_numpy(adj_matrix_np).float().to_sparse()

            # Feature matrix: random features + degree
            degrees = adj_matrix_np.sum(axis=1)
            feature_matrix_np = np.hstack([
                np.random.randn(size, 2).astype(np.float32),  # Random features
                degrees[:, np.newaxis].astype(np.float32)     # Degree as an additional feature
            ])
            feature_matrix = torch.from_numpy(feature_matrix_np).float()

            # Select random source and target nodes
            source, target = random.sample(range(size), 2)
            print(f"Source node: {source}, Target node: {target}")

            # Generate true top K paths using Yen's Algorithm
            print("Generating true top K paths using Yen's Algorithm...")
            start_time = time.time()
            true_paths_with_scores = yen_top_k_shortest_paths(adj_matrix_np.copy(), source, target, K)
            yen_exec_time = time.time() - start_time

            true_paths = [path for _, path in true_paths_with_scores if path is not None]
            if len(true_paths) < K:
                print(f"Only found {len(true_paths)} paths. Adjusting K to {len(true_paths)}.")
                current_K = len(true_paths)
            else:
                current_K = K
            print(f"True paths: {true_paths}")

            # Define models to compare
            models = {
                'GCN': StackGCN,
                'Chebyshev GCN': ChebNetGCN,
                'GNN': StackGNN,
                'GAT': GAT
            }

            for model_name, model_class in models.items():
                print(f"\nRunning {model_name}...")
                chebyshev_k = 3 if model_name == 'Chebyshev GCN' else None
                
                model_kwargs = {}
                if model_name == 'GAT':
                    model_kwargs.update({'n_heads': 8, 'concat': True})

                # Find top K paths using the unified function
                top_paths, exec_time = find_top_k_paths_unified(
                    model_class=model_class,
                    adj_matrix=adj_matrix,
                    feature_matrix=feature_matrix,
                    source=source,
                    target=target,
                    K=current_K,
                    chebyshev_k=chebyshev_k,
                    **shared_params,
                    **model_kwargs
                )

                # Process top_paths to ensure all paths are lists of integers
                processed_top_paths = []
                for cost, path in top_paths:
                    path = [int(node) for node in path]
                    processed_top_paths.append(path)

                # Convert to set of tuples for comparison
                found_path_set = set(tuple(p) for p in processed_top_paths)
                true_path_set = set(tuple(p) for p in true_paths)

                # Compute accuracy
                correct = len(true_path_set.intersection(found_path_set))
                accuracy = correct / current_K if current_K > 0 else 0.0

                # Store results
                results.append({
                    'Graph Type': graph_type,
                    'Graph Size': size,
                    'Model': model_name,
                    'Accuracy (%)': accuracy * 100,
                    'Execution Time (s)': exec_time
                })

                print(f"{model_name} - Accuracy: {accuracy*100:.2f}%, Execution Time: {exec_time:.4f} seconds")
                print(f"Found paths: {processed_top_paths}")

    results_df = pd.DataFrame(results)

    # Save results to CSV if requested
    if output_csv_path:
        results_df.to_csv(output_csv_path, index=False)
        print(f"\nExperiment results saved to {output_csv_path}")

    # Display the results in the terminal
    if not output_csv_path:
        print("\n\n=== Experiment Results ===")
        print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the full experiment on various graph sizes and models.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--output_csv_path", type=str, required=False, help="Path to save the results as a CSV file.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)

    # Run the experiment
    run_full_experiment(config, output_csv_path=args.output_csv_path)
