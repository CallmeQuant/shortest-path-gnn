import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Union
from layers.gcn_layer import ResidualGCNLayer, GCNLayer
from layers.chebyshev_layer import ChebNetConv


class StackGCN(nn.Module):
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
        Initializes the StackGCN model.

        Parameters
        ----------
        input_dim : int
            Dimensionality of input features.
        hidden_dim : int
            Dimensionality of hidden layers.
        output_dim : int
            Dimensionality of the output layer.
        dropout : float, optional
            Dropout rate, by default 0.1.
        normalization : str, optional
            Type of normalization ('sym' for symmetric, 'rw' for random walk), by default "sym".
        num_layers : int, optional
            Number of GCN layers, by default 2. Must be at least 2.
        residual : bool, optional
            Whether to use residual connections, by default True.
        """
        super(StackGCN, self).__init__()

        self.residual = residual
        layers = []
        if num_layers < 2:
            num_layers = 2  # At least input and output layers

        # Input layer
        layers.append(GCNLayer(input_dim, hidden_dim, normalization=normalization))
        # Hidden layers
        for _ in range(num_layers - 2):
            if residual:
                layers.append(ResidualGCNLayer(hidden_dim, hidden_dim, normalization=normalization))
            else:
                layers.append(GCNLayer(hidden_dim, hidden_dim, normalization=normalization))
        # Output layer
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

    def forward(self, adj_matrix: torch.sparse.FloatTensor, feature_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the StackGCN model.

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

class ChebNetGCN(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        num_hidden_layers: int = 0, 
        dropout: float = 0.1, 
        residual: bool = True, 
        k: int = 2
    ) -> None:
        """
        Initializes the ChebNetGCN model.

        Parameters
        ----------
        input_dim : int
            Number of input features per node.
        hidden_dim : int
            Number of hidden units.
        output_dim : int
            Number of output features per node.
        num_hidden_layers : int, optional
            Number of hidden ChebNetConv layers, by default 0.
        dropout : float, optional
            Dropout rate, by default 0.1.
        residual : bool, optional
            Whether to use residual connections, by default True.
        k : int, optional
            Chebyshev polynomial order, by default 2.
        """
        super(ChebNetGCN, self).__init__()

        self.dropout = dropout
        self.residual = residual
        self.k = k

        # Input layer
        self.input_conv = ChebNetConv(input_dim, hidden_dim, k)

        # Hidden layers
        self.hidden_convs = nn.ModuleList([
            ChebNetConv(hidden_dim, hidden_dim, k) for _ in range(num_hidden_layers)
        ])

        # Output layer
        self.output_conv = ChebNetConv(hidden_dim, output_dim, k)

        # Activation
        self.relu = nn.ReLU()

        # self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes weights of the ChebNetConv layers using Xavier uniform initialization.
        """
        for layer in self.hidden_convs:
            nn.init.xavier_uniform_(layer.linear.weight)
        nn.init.xavier_uniform_(self.input_conv.linear.weight)
        nn.init.xavier_uniform_(self.output_conv.linear.weight)

    def forward(
        self, 
        x: torch.Tensor, 
        laplacian: torch.sparse_coo_tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the ChebNetGCN model.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        laplacian : torch.sparse_coo_tensor
            Scaled Laplacian matrix.
        labels : torch.Tensor, optional
            Labels for nodes, by default None.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If labels are provided, returns (output, loss).
            Otherwise, returns output.
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input_conv(x, laplacian)  # Shape: [num_nodes, hidden_dim]
        x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.hidden_convs:
            if self.residual:
                x_residual = x
                x = conv(x, laplacian)
                x = self.relu(x + x_residual)
            else:
                x = conv(x, laplacian)
                x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.output_conv(x, laplacian)  # Shape: [num_nodes, output_dim]

        if labels is not None:
            # For regression tasks, use MSELoss
            loss_fn = nn.MSELoss()
            loss = loss_fn(x, labels)
            return x, loss

        return x