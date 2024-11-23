import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

class ChebNetConv(nn.Module):
    """
    Chebyshev Convolution Layer (ChebNetConv).

    This layer implements the Chebyshev spectral graph convolution, which approximates graph convolutions
    using Chebyshev polynomials. It efficiently captures localized graph structure up to the K-th order
    neighborhood.

    Parameters
    ----------
    in_features : int
        Number of input features per node.
    out_features : int
        Number of output features per node.
    k : int
        Order of the Chebyshev polynomials (K).

    Attributes
    ----------
    K : int
        Chebyshev polynomial order.
    linear : nn.Linear
        Linear transformation layer that maps concatenated Chebyshev features to output features.

    Examples
    --------
    >>> cheb_conv = ChebNetConv(in_features=16, out_features=32, k=3)
    >>> x = torch.randn(num_nodes, 16)
    >>> laplacian = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes))
    >>> output = cheb_conv(x, laplacian)
    """

    def __init__(self, in_features: int, out_features: int, k: int) -> None:
        super(ChebNetConv, self).__init__()
        self.K: int = k
        self.linear: nn.Linear = nn.Linear(in_features * k, out_features)

    def forward(self, x: Tensor, laplacian: torch.sparse.FloatTensor) -> Tensor:
        """
        Forward pass of the ChebNetConv layer.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, in_features].
        laplacian : torch.sparse.FloatTensor
            Scaled Laplacian matrix of shape [num_nodes, num_nodes].

        Returns
        -------
        torch.Tensor
            Output feature matrix of shape [num_nodes, out_features].

        Raises
        ------
        RuntimeError
            If the Laplacian matrix is not a sparse tensor.
        """
        if not laplacian.is_sparse:
            laplacian = laplacian.to_sparse()

        cheb_x: Tensor = self.__transform_to_chebyshev(x, laplacian)  # Shape: [num_nodes, in_features * K]
        out: Tensor = self.linear(cheb_x)  # Shape: [num_nodes, out_features]
        return out

    def __transform_to_chebyshev(self, x: Tensor, laplacian: torch.sparse.FloatTensor) -> Tensor:
        """
        Transforms the input features using Chebyshev polynomials.

        Computes the Chebyshev polynomials up to the K-th order and concatenates them for each node's features.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, in_features].
        laplacian : torch.sparse.FloatTensor
            Scaled Laplacian matrix of shape [num_nodes, num_nodes].

        Returns
        -------
        torch.Tensor
            Transformed feature matrix of shape [num_nodes, in_features * K].

        Raises
        ------
        ValueError
            If the Chebyshev order K is less than 1.
        """
        if self.K < 1:
            raise ValueError("Chebyshev order K must be at least 1.")

        # Initialize Chebyshev polynomials
        cheb_x = x.unsqueeze(2)  # Shape: [num_nodes, in_features, 1]
        x0 = x  # T0(x)
        cheb_x = x0.unsqueeze(2)  # Shape: [num_nodes, in_features, 1]

        if self.K > 1:
            x1 = torch.sparse.mm(laplacian, x0)  # T1(x) = L * T0(x)
            cheb_x = torch.cat((cheb_x, x1.unsqueeze(2)), dim=2)  # Shape: [num_nodes, in_features, 2]

            for _ in range(2, self.K):
                x2 = 2 * torch.sparse.mm(laplacian, x1) - x0  # T_k(x) = 2 * L * T_{k-1}(x) - T_{k-2}(x)
                cheb_x = torch.cat((cheb_x, x2.unsqueeze(2)), dim=2)  # Append T_k(x)
                x0, x1 = x1, x2  # Update T_{k-2} and T_{k-1}

        cheb_x = cheb_x.view(x.size(0), -1)  # Flatten to [num_nodes, in_features * K]
        return cheb_x
