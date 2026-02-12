"""Graph convolution layers for CoRe-GD model."""

import torch
from torch_geometric.nn import MessagePassing


class GRUEdgeConv(MessagePassing):
    """
    GRU-based edge convolution with edge-conditioned message passing.
    
    Args:
        emb_dim: Embedding dimension
        mlp_edge: MLP for processing edge features
        aggr: Aggregation method ('add', 'mean', 'max')
    """
    def __init__(self, emb_dim, mlp_edge, aggr):
        super(GRUEdgeConv, self).__init__(aggr=aggr)
        self.rnn = torch.nn.GRUCell(emb_dim, emb_dim)
        self.mlp_edge = mlp_edge

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with GRU update.
        
        Args:
            x: Node features
            edge_index: Edge indices  
            edge_attr: Optional edge attributes (e.g., UMAP weights)
        """
        out = self.rnn(self.propagate(edge_index, x=x, edge_attr=edge_attr), x)
        return out

    def message(self, x_j, x_i, edge_attr=None):
        """
        Compute messages from source to target nodes.
        
        Edge attributes are concatenated with node features as MLP input,
        allowing the network to learn how to use edge weights.
        
        Args:
            x_j: Source node features
            x_i: Target node features
            edge_attr: Edge attributes (concatenated if provided)
        """
        if edge_attr is not None:
            # Concatenate: (source_features, target_features, edge_weight)
            concatted = torch.cat((x_j, x_i, edge_attr.view(-1, 1)), dim=1)
        else:
            # Backward compatibility: no edge attributes
            concatted = torch.cat((x_j, x_i), dim=1)
        
        return self.mlp_edge(concatted)

class GINEdgeConv(MessagePassing):
    """
    Graph Isomorphism Network (GIN) edge convolution.
    
    Args:
        mlp: MLP for node feature transformation
        mlp_edge: MLP for processing edge features
        aggr: Aggregation method ('add', 'mean', 'max')
    """
    def __init__(self, mlp, mlp_edge, aggr):
        super(GINEdgeConv, self).__init__(aggr=aggr)
        self.mlp = mlp
        self.mlp_edge = mlp_edge
        self.eps = torch.nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with GIN aggregation.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Optional edge attributes (e.g., UMAP weights)
        """
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out
    
    def message(self, x_j, x_i, edge_attr=None):
        """
        Compute messages from source to target nodes.
        
        Edge attributes are concatenated with node features as MLP input,
        allowing the network to learn how to use edge weights.
        
        Args:
            x_j: Source node features
            x_i: Target node features
            edge_attr: Edge attributes (concatenated if provided)
        """
        if edge_attr is not None:
            # Concatenate: (source_features, target_features, edge_weight)
            concatted = torch.cat((x_j, x_i, edge_attr.view(-1, 1)), dim=1)
        else:
            # Backward compatibility: no edge attributes
            concatted = torch.cat((x_j, x_i), dim=1)
        
        return self.mlp_edge(concatted)
