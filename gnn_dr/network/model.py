import torch
import torch.nn.functional as F

from torch_geometric.utils import to_undirected, batched_negative_sampling
import copy
from torch_cluster import knn
from .convolutions import GRUEdgeConv, GINEdgeConv

from torch_geometric.nn import GATv2Conv

def get_model(config):
    """
    Create and configure a CoRe-GD model based on the provided configuration.
    
    Args:
        config: Configuration object containing model hyperparameters
        
    Returns:
        Configured CoReGD model instance
    """
    if config.normalization == 'LayerNorm':
        normalization_function = torch.nn.LayerNorm
    elif config.normalization == 'BatchNorm':
        normalization_function = torch.nn.BatchNorm1d
    elif config.normalization == 'None':
        normalization_function = torch.nn.Identity
    else:
        print('Unrecognized normalization function: ' + config.normalization)
        exit(1)

    # Calculate input channels: random + beacons + spectral + random_projection + pca + clip_embeddings
    random_projection_dim = getattr(config, 'random_projection_dim', 0)
    pca_dim = getattr(config, 'pca_dim', 0)
    in_channels = config.random_in_channels + config.laplace_eigvec + random_projection_dim + pca_dim + config.clip_in_channels
    if config.use_beacons:
        in_channels += config.num_beacons * config.encoding_size_per_beacon

    # Get neg_sample parameters if available (for negative sampling rewiring)
    neg_sample_multiplier = getattr(config, 'neg_sample_multiplier', 1.0)
    neg_sample_force_undirected = getattr(config, 'neg_sample_force_undirected', True)
    
    # Check if edge attributes (e.g., UMAP weights) should be used
    use_edge_attr = getattr(config, 'use_edge_attr', True)  # Default True for new models
    
    model = CoReGD(in_channels, config.hidden_dimension, config.out_dim, config.hidden_state_factor, config.dropout,mlp_depth=config.mlp_depth, conv=config.conv,
                    skip_prev=config.skip_previous, skip_input=config.skip_input, aggregation=config.aggregation,
                    normalization=normalization_function, overlay=config.rewiring, overlay_freq=config.alt_freq, knn_k=config.knn_k,
                    neg_sample_multiplier=neg_sample_multiplier, neg_sample_force_undirected=neg_sample_force_undirected,
                    use_edge_attr=use_edge_attr)

    return model

class CoReGD(torch.nn.Module):
    """
    CoRe-GD: Competitive Graph Drawing with Graph Neural Networks.
    
    This model uses message passing neural networks with dynamic graph rewiring
    to learn graph layouts by minimizing stress functions.
    
    Args:
        in_channels: Number of input features per node
        hidden_channels: Number of hidden features
        out_channels: Number of output dimensions (typically 2 for 2D layouts)
        hidden_state_factor: Multiplier for hidden layer sizes in MLPs
        dropout: Dropout probability
        mlp_depth: Number of hidden layers in MLPs
        conv: Type of convolution ('gin', 'gru', 'gat')
        skip_input: Whether to use skip connections from input
        skip_prev: Whether to use skip connections from previous layer
        aggregation: Aggregation method for message passing ('add', 'mean', 'max')
        normalization: Normalization layer class
        overlay: Graph rewiring strategy ('knn', 'neg_sample', None)
        overlay_freq: How often to apply rewiring (every N convolution layers)
        knn_k: Number of nearest neighbors for KNN rewiring
    """
    def __init__(self, in_channels, hidden_channels, out_channels, hidden_state_factor, dropout, mlp_depth=2, conv='gin', skip_input=False, skip_prev=False, aggregation='add', normalization=torch.nn.LayerNorm, overlay='knn', overlay_freq='1', knn_k='4', neg_sample_multiplier=1.0, neg_sample_force_undirected=True, use_edge_attr=True):
        super(CoReGD, self).__init__()
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        self.encoder = self.get_mlp(in_channels, hidden_state_factor*hidden_channels, mlp_depth , hidden_channels, normalization, last_relu=True)
        self.overlay = overlay
        self.overlay_freq = overlay_freq
        self.knn_k = knn_k
        self.neg_sample_multiplier = neg_sample_multiplier
        self.neg_sample_force_undirected = neg_sample_force_undirected
       
        # Edge MLP input dimension: 2*hidden + 1 if using edge_attr, else 2*hidden
        edge_mlp_input_dim = 2*hidden_channels + (1 if use_edge_attr else 0)
        
        if conv == 'gin':
            main_conv = GINEdgeConv(self.get_mlp(hidden_channels, hidden_state_factor*hidden_channels, mlp_depth, hidden_channels, normalization,last_relu=True),
             self.get_mlp(edge_mlp_input_dim, hidden_state_factor*2*hidden_channels, mlp_depth, hidden_channels, normalization, last_relu=True), aggr=aggregation)
        elif conv == 'gru' or conv == 'gru-mlp':
            main_conv = GRUEdgeConv(hidden_channels, self.get_mlp(edge_mlp_input_dim, hidden_state_factor*hidden_channels, mlp_depth, hidden_channels, normalization), aggr=aggregation)
        elif conv =='gat':
            main_conv = GATv2Conv(hidden_channels, hidden_channels)
        else:
            raise Exception('Unrecognized option: ' + conv)

        self.convs = torch.nn.ModuleList([copy.deepcopy(main_conv) for i in range(overlay_freq)])

        # Rewiring conv: always without edge_attr (rewired edges don't have attributes)
        # Build separately with 2*hidden_channels input (no +1 for edge_attr)
        edge_mlp_input_dim_rewire = 2*hidden_channels
        
        if conv == 'gin':
            self.conv_alt = GINEdgeConv(self.get_mlp(hidden_channels, hidden_state_factor*hidden_channels, mlp_depth, hidden_channels, normalization,last_relu=True),
             self.get_mlp(edge_mlp_input_dim_rewire, hidden_state_factor*2*hidden_channels, mlp_depth, hidden_channels, normalization, last_relu=True), aggr=aggregation)
        elif conv == 'gru' or conv == 'gru-mlp':
            self.conv_alt = GRUEdgeConv(hidden_channels, self.get_mlp(edge_mlp_input_dim_rewire, hidden_state_factor*hidden_channels, mlp_depth, hidden_channels, normalization), aggr=aggregation)
        elif conv =='gat':
            self.conv_alt = GATv2Conv(hidden_channels, hidden_channels)
        else:
            raise Exception('Unrecognized option: ' + conv)

        self.decoder = self.get_mlp(hidden_channels, hidden_state_factor * hidden_channels, mlp_depth, out_channels, normalization, last_relu = False)
        
        self.skip_input = self.get_mlp(hidden_channels + in_channels, hidden_state_factor * hidden_channels, mlp_depth, hidden_channels, normalization) if skip_input else None
        self.skip_previous = self.get_mlp(2*hidden_channels, hidden_state_factor*2*hidden_channels, mlp_depth, hidden_channels, normalization) if skip_prev else None


    def get_mlp(self, input_dim, hidden_dim, mlp_depth, output_dim, normalization, last_relu=True):
        """
        Construct a multi-layer perceptron with normalization and dropout.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            mlp_depth: Number of hidden layers
            output_dim: Output dimension
            normalization: Normalization layer class
            last_relu: Whether to apply ReLU and normalization to output
            
        Returns:
            Sequential MLP module
        """
        relu_layer = torch.nn.ReLU()
        modules = [torch.nn.Linear(input_dim, int(hidden_dim)), normalization(int(hidden_dim)), relu_layer, torch.nn.Dropout(self.dropout)]
        for i in range(0, int(mlp_depth)):
            modules = modules + [torch.nn.Linear(int(hidden_dim), int(hidden_dim)), normalization(int(hidden_dim)), relu_layer, torch.nn.Dropout(self.dropout)]
        modules = modules + [torch.nn.Linear(int(hidden_dim), output_dim)]
        
        if last_relu:
            modules.append(normalization(output_dim))
            modules.append(relu_layer)

        return torch.nn.Sequential(*modules)

    def encode(self, batched_data):
        """Encode initial node features."""
        return self.encoder(batched_data.x)

    def compute_rewiring(self, pos, batched_data):
        """
        Compute dynamic graph rewiring based on current node positions.
        
        If precomputed_rewiring_edge_index exists in batched_data, use it directly
        (for neg_sample only). Otherwise, compute dynamically.
        
        Args:
            pos: Current node positions
            batched_data: Batch of graph data
            
        Returns:
            New edge index or None if no rewiring
        """
        # Check for precomputed rewiring edges (only for neg_sample)
        if self.overlay == 'neg_sample':
            precomputed = getattr(batched_data, 'precomputed_rewiring_edge_index', None)
            if precomputed is not None:
                # Use precomputed edges - no computation needed
                return precomputed
        
        if self.overlay == 'knn':
            new_edges = knn(x=pos, y=pos, k=self.knn_k, batch_x=batched_data.batch, batch_y=batched_data.batch)
            new_edges = torch.flip(new_edges, dims=[0,1])
            return new_edges
        if self.overlay == 'neg_sample':
            # Use negative sampling for repulsive edges (dynamic computation)
            num_neg_samples = max(1, int(batched_data.edge_index.shape[1] * self.neg_sample_multiplier))
            neg_edges = batched_negative_sampling(
                edge_index=batched_data.edge_index,
                batch=batched_data.batch,
                num_neg_samples=num_neg_samples,
                method='sparse'
            )
            if self.neg_sample_force_undirected:
                neg_edges = to_undirected(neg_edges)
            return neg_edges
        return None


    def forward(self, batched_data, iterations, return_layers=False, encode=True, transform_to_undirected=False):
        """
        Forward pass through the model.
        
        Args:
            batched_data: Batch of graph data
            iterations: Number of message passing iterations
            return_layers: Whether to return intermediate layer states
            encode: Whether to encode input features
            transform_to_undirected: Whether to make edges undirected
            
        Returns:
            Node positions (and optionally intermediate layer states)
        """
        x_orig, x, edge_index = batched_data.x_orig, batched_data.x, batched_data.edge_index

        # Extract edge attributes if available and config allows (e.g., UMAP fuzzy weights)
        edge_attr = getattr(batched_data, 'edge_attr', None) if self.use_edge_attr else None

        if transform_to_undirected:
            edge_index = to_undirected(edge_index)

        layers = []

        if encode:
            batched_data.x_orig = x
            x_orig = x
            x = self.encoder(x)
        else:
            #pos = torch.sigmoid(self.decoder(x)).detach()
            pos = self.decoder(x).detach()
            if self.skip_input is not None:
                x = self.skip_input(torch.cat([x, x_orig], dim=1))
            new_edges = self.compute_rewiring(pos, batched_data)
            if new_edges is not None:
                # Rewired edges don't have edge attributes
                x = self.conv_alt(x, new_edges, edge_attr=None)

        previous = x

        for i in range(iterations-1):
            for conv in self.convs:
                # Pass edge attributes to convolutions
                x = conv(x, edge_index, edge_attr=edge_attr)

            #pos = torch.sigmoid(self.decoder(x)).detach()
            pos = self.decoder(x).detach()
            if return_layers:
                layers.append(x)
            
            if self.skip_input is not None:
                x = self.skip_input(torch.cat([x, x_orig], dim=1))

            new_edges = self.compute_rewiring(pos, batched_data)
            if new_edges is not None:
                # Rewired edges don't have edge attributes
                x = self.conv_alt(x, new_edges, edge_attr=None)

            if self.skip_previous is not None:
                x = self.skip_previous(torch.cat([x, x + previous], dim=1))
            
            previous = x
        
        for conv in self.convs:
            # Final convolutions with edge attributes
            x = conv(x, edge_index, edge_attr=edge_attr)
        if return_layers:
            layers.append(x)

        x = self.decoder(x)
        #x = torch.sigmoid(x)
        
        if return_layers:
            return x, layers
        else:
            return x
