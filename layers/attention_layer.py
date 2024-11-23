import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
        n_heads: int = 1
    ) -> None:
        """
        Initializes the Graph Attention Layer.

        Parameters
        ----------
        in_features : int
            Number of input features per node.
        out_features : int
            Number of output features per node.
        dropout : float, optional
            Dropout rate, by default 0.1.
        alpha : float, optional
            Negative slope for LeakyReLU, by default 0.2.
        concat : bool, optional
            Whether to concatenate the attention heads' outputs, by default True.
        n_heads : int, optional
            Number of attention heads, by default 1.
        """
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.n_heads = n_heads

        # Learnable parameters
        self.W = nn.Parameter(torch.empty(size=(in_features, n_heads * out_features)))
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * out_features, 1)))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initializes learnable parameters using Xavier normal initialization.
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W, gain=gain)
        nn.init.xavier_normal_(self.a, gain=gain)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W, gain=gain)
        nn.init.xavier_normal_(self.a, gain=gain)

    def forward(
        self, 
        x: torch.Tensor, 
        adj: torch.sparse.FloatTensor
        ) -> torch.Tensor:
        """
        Forward pass of the GraphAttentionLayer.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, in_features].
        adj : torch.sparse.FloatTensor
            Sparse adjacency matrix of shape [num_nodes, num_nodes].

        Returns
        -------
        torch.Tensor
            Output feature matrix of shape [num_nodes, out_features * n_heads] if concat=True,
            else [num_nodes, out_features].
        """
        N = x.size(0)

        if adj.is_sparse:
            edge_index = adj._indices()  # [2, E]
        else:
            edge_index = adj.nonzero().t()  # [2, E]

        # Linear transformation
        h = torch.mm(x, self.W)  # [N, n_heads * out_features]
        h = h.view(N, self.n_heads, self.out_features)  # [N, n_heads, out_features]

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[:,None,:,:].expand(-1,N,-1,-1), h[None,:,:,:].expand(N,-1,-1,-1)), dim=-1)  # [N, N, n_heads, 2*out_features]
        edge_e = self.leakyrelu(torch.einsum("ijhf,hfa->ijha", edge_h, self.a)).squeeze(-1)  # [N, N, n_heads]

        # Attention weights
        attention = F.softmax(edge_e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)  # [N, N, n_heads]

        # Apply attention
        h_prime = torch.einsum("ijh,jhf->ihf", attention, h)  # [N, n_heads, out_features]

        if self.concat:
            # Concatenate heads
            h_prime = h_prime.reshape(N, self.n_heads * self.out_features)
        else:
            # Average heads
            h_prime = h_prime.mean(dim=1)

        return h_prime