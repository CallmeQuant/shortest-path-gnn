# Shortest Path Prediction Using GNNs

This project explores various Graph Neural Network (GNN)-based models to solve shortest path prediction problems. It includes implementations of multiple GNN models such as Graph Convolutional Networks (GCN), Chebyshev GCN, Graph Attention Networks (GAT), and a standard GNN. The focus is on comparing the models in terms of accuracy and execution time when predicting shortest paths on randomly generated graphs.

## GNN Models for Shortest Path Prediction
- **Graph Convolutional Networks (GCN)**: A GNN that uses spectral graph theory to apply convolution operations on nodes ([GCN paper](https://arxiv.org/abs/1609.02907)).
- **Chebyshev GCN**: Uses Chebyshev polynomials for more efficient spectral convolutions.
- **Graph Attention Network (GAT)**: Applies attention mechanisms on graph nodes, allowing for more nuanced message passing between nodes ([GAT paper](https://arxiv.org/abs/1710.10903)).
- **Stacked GNN**: A simple stacked implementation of GNN layers for comparison.

## Project Structure
The project is organized as follows:

```plaintext
.
├── configs/
├── data/
│   ├── graph1_adjacency_matrix.npy
│   ├── graph2_adjacency_matrix.npy
│   ├── graph3_adjacency_matrix.npy
│   ├── default_feature_matrix.npy
├── exp/
│   ├── __init__.py
│   ├── full_exp.py
│   ├── simple_exp.py
├── figs/
├── layers/
│   ├── __init__.py
│   ├── attention_layer.py
│   ├── chebyshev_layer.py
│   ├── gcn_layer.py
├── models/
│   ├── __init__.py
│   ├── GAT.py
│   ├── GCN.py
│   ├── GNN.py
│   ├── Yen.py
├── utils.py
├── requirements.txt
├── README.md
```

### Directory Breakdown
- **configs/**: Configuration files for setting up different experiment parameters.
- **data/**: Stores `.npy` files that contain the adjacency matrices and feature matrices used for the graph data.
- **exp/**: Experiment scripts, including [full_exp.py](exp/full_exp.py) for running full experiments comparing models and [simple_exp.py](exp/simple_exp.py) for smaller, isolated runs.
- **figs/**: Stores generated figures like accuracy and runtime plots.
- **layers/**: Contains code defining different GNN layers.
  - **attention_layer.py**: Implements the attention mechanism used in Graph Attention Networks.
  - **chebyshev_layer.py**: Defines the layer using Chebyshev polynomials for efficient spectral convolutions.
  - **gcn_layer.py**: Implements the basic GCN layer for spectral convolution operations.
- **models/**: Houses the implementations of various GNN models.
  - **GAT.py**: Contains the implementation of the Graph Attention Network.
  - **GCN.py**: Implements the Graph Convolutional Network.
  - **GNN.py**: Contains the general GNN model used for comparison.
  - **Yen.py**: Implementation of Yen's K-Shortest Paths algorithm for path-finding baselines.
- **utils.py**: Utility functions shared across scripts.
- **requirements.txt**: List of required Python packages for running the project.

## Running the Project
A typical process for installing the package dependencies involves creating a new Python virtual environment. Below are the instructions to execute the experiments from the command line.

### Step 1: Clone the Repository
Clone this repository to your local machine using:

```sh
git clone https://github.com/yourusername/shortest-path-gnn.git
cd shortest-path-gnn
```

### Step 2: Set Up the Environment
Install the required dependencies using the provided `requirements.txt`:

```sh
pip install -r requirements.txt
```

### Step 3: Run the Experiments
You can run the experiments by executing either the full or simple experiment scripts from the **exp/** folder.

#### Run Full Experiment
The [full_exp.py](exp/full_exp.py) script runs full experiments comparing multiple models and graph sizes. You can run it using:

```sh
python -m exp.full_exp --config_path "configs/full_experiment_config.yaml" --output_csv_path "results/experiment_results.csv"
```

#### Run Simple Experiment
To quickly test a specific configuration, you can use the [simple_exp.py](exp/simple_exp.py) script:

```sh
python -m exp.simple_exp --config_path "configs/simple_exp_config.yaml"
```

### Modifying Experiment Parameters
The experiment parameters are stored in the YAML files located in the **configs/** folder. You can modify these files to adjust various aspects of the experiments, such as:

- **Graph Size and Types**: Control the number of nodes, edge probabilities, and graph types (directed/undirected).
- **Model Parameters**: Adjust hyperparameters such as:
  - **hidden_dim**: Number of hidden dimensions in each layer.
  - **num_layers**: Number of GNN layers.
  - **learning_rate**: Learning rate for training.
  - **dropout**: Dropout rate used in the GNN layers.

Example of a configuration snippet:
```yaml
seed: 42
graph_sizes: [10, 30, 50, 100]
graph_types: ['undirected', 'directed']
K: 3
shared_params:
  hidden_dim: 32
  output_dim: 1
  dropout: 0.1
  normalization: "sym"
  num_layers: 4
  num_epochs: 100
  learning_rate: 0.01
  device: "cpu"
```

To change parameters, simply modify the values in the respective YAML file before running the experiments.

### Output and Results
- **Figures**: The generated figures will be saved in the **figs/** directory.
- **CSV Results**: Experiment results, if specified, will be saved in a CSV file for easy analysis.

#### Note on Usage
This repository is designed for academic and experimental purposes. The graphs used are randomly generated, and the code compares GNN-based shortest path predictions to traditional path-finding algorithms.
