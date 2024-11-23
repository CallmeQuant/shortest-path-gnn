import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import warnings
from layers.gcn_layer import ResidualGCNLayer, GCNLayer

class StackGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        normalization: str = "sym",
        num_layers: int = 2,
        residual: bool = True
    ) -> None:
        """
        Initializes the StackGNN model.

        Parameters
        ----------
        input_dim : int
            Number of input features per node.
        hidden_dim : int
            Number of hidden units.
        output_dim : int
            Number of output features per node.
        dropout : float, optional
            Dropout rate, by default 0.1.
        normalization : str, optional
            Type of normalization ('sym' for symmetric, 'rw' for random walk), by default "sym".
        num_layers : int, optional
            Number of GCN layers, must be at least 2. Defaults to 2.
        residual : bool, optional
            Whether to use residual connections, by default True.
        """
        super(StackGNN, self).__init__()

        if num_layers < 2:
            warnings.warn("num_layers should be at least 2")
            num_layers = 2  # At least input and output layers

        self.residual = residual
        layers = [GCNLayer(input_dim, hidden_dim, normalization=normalization)]
        for _ in range(num_layers - 2):
            if residual:
                layers.append(ResidualGCNLayer(hidden_dim, hidden_dim, normalization=normalization))
            else:
                layers.append(GCNLayer(hidden_dim, hidden_dim, normalization=normalization))
        layers.append(GCNLayer(hidden_dim, output_dim, normalization=normalization))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes weights of the GCN layers using Xavier uniform initialization.
        """
        for layer in self.layers:
            if isinstance(layer, GCNLayer):
                nn.init.xavier_uniform_(layer.linear.weight)
            elif isinstance(layer, ResidualGCNLayer):
                nn.init.xavier_uniform_(layer.gcn.linear.weight)
                if isinstance(layer.residual, nn.Linear):
                    nn.init.xavier_uniform_(layer.residual.weight)

    def forward(
        self, 
        adj_matrix: torch.sparse.FloatTensor, 
        feature_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the StackGNN model.

        Parameters
        ----------
        adj_matrix : torch.sparse.FloatTensor
            Sparse adjacency matrix of shape [num_nodes, num_nodes].
        feature_matrix : torch.Tensor
            Feature matrix of shape [num_nodes, input_dim].

        Returns
        -------
        torch.Tensor
            Output feature matrix of shape [num_nodes, output_dim].
        """
        x = self.dropout(feature_matrix)
        for i, layer in enumerate(self.layers):
            x = layer(adj_matrix, x)
            if i < len(self.layers) - 1:
                if not self.residual or not isinstance(layer, ResidualGCNLayer):
                    x = self.relu(x)
                x = self.dropout(x)
        return x