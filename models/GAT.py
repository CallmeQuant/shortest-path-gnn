import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List
import warnings
from layers.attention_layer import GraphAttentionLayer

class GAT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        n_heads: int = 1,
        num_layers: int = 2,
        concat: bool = True,
        residual: bool = False
        ) -> None:
        """
        Initializes the Graph Attention Network (GAT) model.

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
        n_heads : int, optional
            Number of attention heads, by default 1.
        num_layers : int, optional
            Number of GAT layers, by default 2.
        concat : bool, optional
            Whether to concatenate the attention heads' outputs in hidden layers, by default True.
        residual : bool, optional
            Whether to use residual connections, by default False.
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.concat = concat
        self.n_heads = n_heads
        self.residual = residual

        # Input layer
        self.layers.append(GraphAttentionLayer(input_dim, hidden_dim, dropout, concat=concat, n_heads=n_heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphAttentionLayer(hidden_dim * n_heads if concat else hidden_dim,
                                                   hidden_dim, dropout, concat=concat, n_heads=n_heads))

        # Output layer
        self.layers.append(GraphAttentionLayer(hidden_dim * n_heads if concat else hidden_dim,
                                               output_dim, dropout, concat=False, n_heads=1))

        if self.residual:
            self.residuals = nn.ModuleList()
            in_dims = [input_dim] + [hidden_dim * n_heads if concat else hidden_dim] * (num_layers - 1)
            out_dims = [hidden_dim * n_heads if concat else hidden_dim] * (num_layers - 1) + [output_dim]
            for in_dim, out_dim in zip(in_dims, out_dims):
                self.residuals.append(nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity())

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes weights of the GraphAttentionLayer layers using Xavier uniform initialization.
        """
        for layer in self.layers:
            if isinstance(layer, GraphAttentionLayer):
                nn.init.xavier_uniform_(layer.W)
                nn.init.xavier_uniform_(layer.a)
        if self.residual:
            for res in self.residuals:
                if isinstance(res, nn.Linear):
                    nn.init.xavier_uniform_(res.weight)

    def forward(
        self, 
        adj: torch.sparse.FloatTensor, 
        x: torch.Tensor
        ) -> torch.Tensor:
        """
        Forward pass of the GAT model.

        Parameters
        ----------
        adj : torch.sparse.FloatTensor
            Sparse adjacency matrix of shape [num_nodes, num_nodes].
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].

        Returns
        -------
        torch.Tensor
            Output feature matrix of shape [num_nodes, output_dim].
        """
        for i, layer in enumerate(self.layers):
            x_in = x
            x = layer(x, adj)
            if self.residual:
                x = x + self.residuals[i](x_in)  # Add residual connection
            if i != len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x