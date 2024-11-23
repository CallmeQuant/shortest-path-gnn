import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network (GCN) Layer.

    This layer performs graph convolution operations, including normalization of the adjacency matrix
    and linear transformation of the input features.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    output_dim : int, optional
        Dimensionality of output features. Defaults to 1.
    normalization : str, optional
        Type of normalization to apply to the adjacency matrix. 
        - 'sym' for symmetric normalization (D^{-0.5} * A * D^{-0.5}).
        - 'rw' for random walk normalization (D^{-1} * A).
        Defaults to "sym".

    Attributes
    ----------
    linear : nn.Linear
        Linear transformation layer without bias.
    input_dim : int
        Dimensionality of input features.
    output_dim : int
        Dimensionality of output features.
    normalization : str
        Type of normalization applied to the adjacency matrix.

    Raises
    ------
    ValueError
        If an invalid normalization type is provided.

    Examples
    --------
    >>> gcn_layer = GCNLayer(input_dim=16, output_dim=32, normalization='sym')
    >>> adj = torch.sparse.FloatTensor(indices, values, size)
    >>> features = torch.randn(num_nodes, 16)
    >>> output = gcn_layer(adj, features)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        normalization: str = "sym"
    ) -> None:
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalization = normalization
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(
        self,
        adj_matrix: torch.sparse.FloatTensor,
        feature_matrix: Tensor
    ) -> Tensor:
        """
        Forward pass of the GCNLayer.

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

        Raises
        ------
        ValueError
            If an invalid normalization type is specified.
        """
        # Ensure adjacency matrix is sparse and coalesced
        if not adj_matrix.is_sparse:
            adj_matrix = adj_matrix.to_sparse()
        else:
            adj_matrix = adj_matrix.coalesce()

        # Apply normalization
        if self.normalization == "sym":
            adj_normalized = self.symmetric_normalization(adj_matrix)
        elif self.normalization == "rw":
            adj_normalized = self.row_normalization(adj_matrix)
        else:
            raise ValueError("Invalid normalization type. Choose 'sym' or 'rw'.")

        # Perform the GCN operation: A_hat * X * W
        x = torch.sparse.mm(adj_normalized, feature_matrix)  # Shape: [num_nodes, input_dim]
        x = self.linear(x)  # Shape: [num_nodes, output_dim]
        return x

    def symmetric_normalization(self, adj: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
        """
        Applies symmetric normalization to the adjacency matrix: D^{-0.5} * A * D^{-0.5}.

        Parameters
        ----------
        adj : torch.sparse.FloatTensor
            Sparse adjacency matrix.

        Returns
        -------
        torch.sparse.FloatTensor
            Symmetrically normalized adjacency matrix.
        """
        degree = torch.sparse.sum(adj, dim=1).to_dense()  # Shape: [num_nodes]
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0  # Handle division by zero

        D_inv_sqrt = torch.diag(degree_inv_sqrt)  # Shape: [num_nodes, num_nodes]
        adj_dense = adj.to_dense()  # Shape: [num_nodes, num_nodes]

        adj_normalized = D_inv_sqrt @ adj_dense @ D_inv_sqrt  # Shape: [num_nodes, num_nodes]
        return adj_normalized.to_sparse()

    def row_normalization(self, adj: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
        """
        Applies random walk normalization to the adjacency matrix: D^{-1} * A.

        Parameters
        ----------
        adj : torch.sparse.FloatTensor
            Sparse adjacency matrix.

        Returns
        -------
        torch.sparse.FloatTensor
            Row-normalized adjacency matrix.
        """
        degree = torch.sparse.sum(adj, dim=1).to_dense()  # Shape: [num_nodes]
        degree_inv = torch.pow(degree, -1)
        degree_inv[torch.isinf(degree_inv)] = 0.0  # Handle division by zero

        D_inv = torch.diag(degree_inv)  # Shape: [num_nodes, num_nodes]
        adj_dense = adj.to_dense()  # Shape: [num_nodes, num_nodes]

        adj_normalized = D_inv @ adj_dense  # Shape: [num_nodes, num_nodes]
        return adj_normalized.to_sparse()


class ResidualGCNLayer(nn.Module):
    """
    Residual Graph Convolutional Network (GCN) Layer.

    This layer incorporates residual connections into the GCNLayer to facilitate better gradient flow
    and model performance.

    Parameters
    ----------
    in_features : int
        Dimensionality of input features.
    out_features : int
        Dimensionality of output features.
    normalization : str, optional
        Type of normalization to apply to the adjacency matrix. 
        - 'sym' for symmetric normalization (D^{-0.5} * A * D^{-0.5}).
        - 'rw' for random walk normalization (D^{-1} * A).
        Defaults to "sym".

    Attributes
    ----------
    gcn : GCNLayer
        Graph Convolutional Network layer.
    relu : nn.ReLU
        ReLU activation function.
    residual : nn.Module
        Residual connection module, either a linear layer or identity.

    Examples
    --------
    >>> residual_gcn = ResidualGCNLayer(in_features=16, out_features=32, normalization='rw')
    >>> adj = torch.sparse.FloatTensor(indices, values, size)
    >>> features = torch.randn(num_nodes, 16)
    >>> output = residual_gcn(adj, features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        normalization: str = "sym"
    ) -> None:
        super(ResidualGCNLayer, self).__init__()
        self.gcn = GCNLayer(in_features, out_features, normalization)
        self.relu = nn.ReLU()
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = nn.Identity()

    def forward(
        self,
        adj_matrix: torch.sparse.FloatTensor,
        feature_matrix: Tensor
    ) -> Tensor:
        """
        Forward pass of the ResidualGCNLayer.

        Parameters
        ----------
        adj_matrix : torch.sparse.FloatTensor
            Sparse adjacency matrix of shape [num_nodes, num_nodes].
        feature_matrix : torch.Tensor
            Feature matrix of shape [num_nodes, in_features].

        Returns
        -------
        torch.Tensor
            Output feature matrix of shape [num_nodes, out_features].

        Raises
        ------
        ValueError
            If the dimensions of input and output features do not align.
        """
        identity = self.residual(feature_matrix)
        out = self.gcn(adj_matrix, feature_matrix)
        return self.relu(out + identity)
