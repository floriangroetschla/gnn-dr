from tqdm import tqdm
import torch
from torch_geometric.transforms.add_positional_encoding import AddLaplacianEigenvectorPE
import math
import random
from torch_geometric.nn import MessagePassing
import torch_scatter
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    to_undirected,
    is_undirected,
)
from torch_geometric.data import Data
from typing import Any, Optional
import numpy as np
#import cupy as cp
try:
    import cupy as cp
except ImportError:
    pass

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
#torch.set_default_device(device)


class BFSConv(MessagePassing):
    def __init__(self, aggr = "min"):
        super().__init__(aggr=aggr)

    def forward(self, distances, edge_index):
        msg = self.propagate(edge_index, x=distances)
        return torch.minimum(msg, distances)

    def message(self, x_j):
        return x_j + 1

class BFS(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = BFSConv()
    
    def forward(self, data, distances, max_iterations):
        edge_index = data.edge_index

        iteration = 0
        while float('Inf') in distances and iteration < max_iterations:
            distances = self.conv(distances, edge_index)
            iteration += 1
        
        if iteration == max_iterations:
            print('Warning: Check if the graph is connected!')

        return distances

def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

class AddLaplacian(BaseTransform):
    r"""Adds the Laplacian eigenvector positional encoding from the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper to the given graph
    (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of eigenvectors. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    def __init__(
        self,
        k: int,
        attr_name: Optional[str] = 'laplacian_eigenvector_pe',
        use_cupy=False,
        **kwargs,
    ):
        self.k = k
        self.attr_name = attr_name
        self.kwargs = kwargs
        self.use_cupy = use_cupy

    def forward(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization='sym',
            num_nodes=num_nodes,
        )
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        L_np = L.toarray()
        
        #L_cp = cp.array(L_np)
        #eig_vals, eig_vecs = cp.linalg.eigh(L_cp)
        #eig_vecs = cp.real(eig_vecs[:, eig_vals.argsort()])
        #pe = torch.from_numpy(cp.asnumpy(eig_vecs[:, 1:self.k + 1])).to(device)

        if device == 'cpu' or not self.use_cupy:
            eig_vals, eig_vecs = np.linalg.eigh(L_np)
            eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
            pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])

        else:
            L_cp = cp.array(L_np)
            eig_vals, eig_vecs = cp.linalg.eigh(L_cp)
            eig_vecs = cp.real(eig_vecs[:, eig_vals.argsort()])
            pe = torch.from_numpy(cp.asnumpy(eig_vecs[:, 1:self.k + 1])).to(device)


        #eig_vals,eig_vecs = np.linalg.eigh(L_np)

        #eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        #pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])
        sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
        sign = sign.to(pe.device)
        pe *= sign

        data = add_node_attr(data, pe.to(data.x.device), attr_name=self.attr_name)
        return data


class AddPCAProjection(BaseTransform):
    r"""Adds PCA (Principal Component Analysis) projection as positional encoding features.
    
    Computes PCA on each graph's CLIP embeddings independently and returns the top k
    principal component scores. This provides a low-dimensional representation that
    captures the directions of maximum variance in the data.
    
    Unlike global PCA, this computes PCA per-graph, making each graph's features
    relative to its own data distribution.
    
    Args:
        n_components (int): The number of principal components to keep.
        attr_name (str, optional): The attribute name of the data object to add
            projections to. If set to :obj:`None`, will be concatenated to :obj:`data.x`.
            (default: :obj:`"pca_pe"`)
    """
    def __init__(
        self,
        n_components: int,
        attr_name: Optional[str] = 'pca_pe',
        **kwargs,
    ):
        self.n_components = n_components
        self.attr_name = attr_name
        self.kwargs = kwargs

    def forward(self, data: Data) -> Data:
        # Get CLIP embeddings - either from clip_embedding attribute or x
        if hasattr(data, 'clip_embedding') and data.clip_embedding is not None:
            embeddings = data.clip_embedding
        else:
            embeddings = data.x
        
        device = embeddings.device
        n_samples, d_in = embeddings.shape
        
        # Limit components to min(n_samples, d_in, n_components)
        k = min(self.n_components, n_samples, d_in)
        
        # Center the data
        mean = embeddings.mean(dim=0, keepdim=True)
        centered = embeddings - mean
        
        # Compute PCA using SVD: X = U @ S @ V^T
        # The columns of U @ S are the principal component scores
        # Use torch.linalg.svd for GPU support
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        
        # Get principal component scores: U[:, :k] @ diag(S[:k])
        # This is equivalent to centered @ Vh.T[:, :k]
        projected = U[:, :k] * S[:k].unsqueeze(0)  # (N, k)
        
        # Pad with zeros if we have fewer components than requested
        if k < self.n_components:
            padding = torch.zeros(n_samples, self.n_components - k, device=device, dtype=projected.dtype)
            projected = torch.cat([projected, padding], dim=1)
        
        # Add to data object
        data = add_node_attr(data, projected, attr_name=self.attr_name)
        return data


class AddGaussianRandomProjection(BaseTransform):
    r"""Adds Gaussian Random Projection coordinates as positional encoding features.
    
    Uses Johnson-Lindenstrauss random projection to project high-dimensional CLIP embeddings
    to a lower-dimensional space. This is a fast, distance-preserving dimensionality reduction
    technique that can be computed entirely on GPU.
    
    The projection matrix R has entries drawn from N(0, 1/n_components), which guarantees
    approximate preservation of pairwise distances with high probability.
    
    Args:
        n_components (int): The number of dimensions to project to.
        attr_name (str, optional): The attribute name of the data object to add
            projections to. If set to :obj:`None`, will be concatenated to :obj:`data.x`.
            (default: :obj:`"random_projection_pe"`)
        random_state (int, optional): Random seed for reproducibility. If None, uses
            a random projection matrix each time. (default: :obj:`None`)
    """
    def __init__(
        self,
        n_components: int,
        attr_name: Optional[str] = 'random_projection_pe',
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self.n_components = n_components
        self.attr_name = attr_name
        self.random_state = random_state
        self.kwargs = kwargs

    def forward(self, data: Data) -> Data:
        # Get CLIP embeddings - either from clip_embedding attribute or x
        if hasattr(data, 'clip_embedding') and data.clip_embedding is not None:
            embeddings = data.clip_embedding
        else:
            embeddings = data.x
        
        device = embeddings.device
        d_in = embeddings.shape[1]
        
        # Gaussian Random Projection: R ~ N(0, 1/k) where k = n_components
        # This scaling ensures E[||Rx||^2] = ||x||^2
        if self.random_state is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.random_state)
            R = torch.randn(d_in, self.n_components, device=device, generator=generator) / math.sqrt(self.n_components)
        else:
            R = torch.randn(d_in, self.n_components, device=device) / math.sqrt(self.n_components)
        
        # Project: X_proj = X @ R
        projected = embeddings @ R
        
        # Add to data object
        data = add_node_attr(data, projected, attr_name=self.attr_name)
        return data


def compute_positional_encodings(dataset, num_beacons, encoding_size_per_beacon):
    bfs = BFS()
    for graph in dataset:
        starting_nodes = random.sample(range(graph.num_nodes), num_beacons)
        distances = torch.empty(graph.num_nodes, num_beacons, device = graph.x.device).fill_(float('Inf'))
        for i in range(num_beacons):
            distances[starting_nodes[i], i] = 0
        distance_encodings = torch.zeros((graph.num_nodes, num_beacons * encoding_size_per_beacon), dtype=torch.float)
        bfs_distances = bfs(graph, distances, graph.num_nodes)
    
        div_term = torch.exp(torch.arange(0, encoding_size_per_beacon, 2) * (-math.log(10000.0) / encoding_size_per_beacon)).to(bfs_distances.device)
        pes = []
        for beacon_index in range(num_beacons):
            pe = torch.zeros(graph.num_nodes, encoding_size_per_beacon, device=bfs_distances.device)
            pe[:, 0::2] = torch.sin(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
            pe[:, 1::2] = torch.cos(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
            pes.append(pe)
        graph.pe = torch.cat(pes,1)
    
    return dataset

def compute_positional_encodings_batch(batch, num_beacons, encoding_size_per_beacon):
    bfs = BFS()
    graph_sizes = torch_scatter.scatter(torch.ones(batch.batch.shape[0]), batch.batch.cpu()).tolist()
    starting_nodes_per_graph = [random.sample(range(int(num_nodes)), num_beacons) for num_nodes in graph_sizes]
    
    graph_size_acc = 0

    distances = torch.empty(batch.x.shape[0], num_beacons, device = batch.x.device).fill_(float('Inf'))
    for i in range(0, len(graph_sizes)):
        for j in range(num_beacons):
            distances[starting_nodes_per_graph[i][j] + int(graph_size_acc), j] = 0
        graph_size_acc += graph_sizes[i]

    distance_encodings = torch.zeros((batch.x.shape[0], num_beacons * encoding_size_per_beacon), dtype=torch.float)
    bfs_distances = bfs(batch, distances, max(graph_sizes))
    
    div_term = torch.exp(torch.arange(0, encoding_size_per_beacon, 2) * (-math.log(10000.0) / encoding_size_per_beacon)).to(bfs_distances.device)
    pes = []
    for beacon_index in range(num_beacons):
        pe = torch.zeros(batch.x.shape[0], encoding_size_per_beacon, device=bfs_distances.device)
        pe[:, 0::2] = torch.sin(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
        pes.append(pe)
    pes_tensor = torch.cat(pes,1)
    
    return pes_tensor


def reset_randomized_features_batch(batch, config):
    rand_features = torch.rand(batch.x.shape[0], config.random_in_channels, dtype=torch.float, device=batch.x.device)
    batch.x[:,:config.random_in_channels] = rand_features
    if config.use_beacons:
        pes = compute_positional_encodings_batch(batch, config.num_beacons, config.encoding_size_per_beacon)
        batch.x[:,config.random_in_channels:config.random_in_channels+pes.size(dim=1)] = pes
        batch.pe = pes
    batch.x_orig = torch.clone(batch.x)

    return batch


def reset_eigvecs(datalist, config):
    pe_transform = AddLaplacian(k=config.laplace_eigvec, attr_name="laplace_ev", is_undirected=True, use_cupy=config.use_cupy)
    for idx in range(len(datalist)):
        datalist[idx] = pe_transform(datalist[idx])
        spectral_features = datalist[idx].laplace_ev
        datalist[idx].x[:,-config.laplace_eigvec:] = spectral_features
        datalist[idx].x_orig[:,-config.laplace_eigvec:] = spectral_features
    return datalist


def preprocess_dataset_dr(datalist, config):
    """
    Preprocess DR dataset by adding random/beacon/spectral features to CLIP embeddings.
    
    Handles both static (list) and dynamic (generator) datasets.
    Final input features: [random_features] + [beacon_features] + [spectral_features] + [clip_embeddings]
    
    Args:
        datalist: List of Data objects or dynamic dataset with CLIP embeddings in x field
        config: Configuration object
        
    Returns:
        Preprocessed datalist (for static) or wrapper (for dynamic)
    """
    # Check if it's a dynamic dataset (has __getitem__ but not subscriptable list)
    if hasattr(datalist, '__getitem__') and not isinstance(datalist, list):
        # Return a preprocessing wrapper for dynamic datasets
        return PreprocessingWrapperDR(datalist, config)
    
    # Static dataset - preprocess in place
    spectrals = []
    if config.use_beacons:
        datalist = compute_positional_encodings(datalist, config.num_beacons, config.encoding_size_per_beacon)
    
    for idx in range(len(datalist)):
        # Store original CLIP embeddings
        clip_embeddings = datalist[idx].clip_embedding.clone()
        #datalist[idx].clip_embedding = clip_embeddings
        
        eigenvecs = config.laplace_eigvec
        beacons = torch.zeros(datalist[idx].num_nodes, 0, dtype=torch.float, device=datalist[idx].x.device)
        if config.use_beacons:
            beacons = datalist[idx].pe
        spectral_features = torch.zeros(datalist[idx].num_nodes, 0, dtype=torch.float, device=datalist[idx].x.device)
        if eigenvecs > 0:
            pe_transform = AddLaplacian(k=eigenvecs, attr_name="laplace_ev", is_undirected=True, use_cupy=config.use_cupy)
            datalist[idx] = pe_transform(datalist[idx])
            spectral_features = datalist[idx].laplace_ev
        
        # Compute Gaussian Random Projection features if enabled
        random_projection_dim = getattr(config, 'random_projection_dim', 0)
        random_projection_features = torch.zeros(datalist[idx].num_nodes, 0, dtype=torch.float, device=datalist[idx].x.device)
        if random_projection_dim > 0:
            rp_transform = AddGaussianRandomProjection(n_components=random_projection_dim, attr_name="random_projection_pe")
            datalist[idx] = rp_transform(datalist[idx])
            random_projection_features = datalist[idx].random_projection_pe
        
        # Compute PCA features if enabled
        pca_dim = getattr(config, 'pca_dim', 0)
        pca_features = torch.zeros(datalist[idx].num_nodes, 0, dtype=torch.float, device=datalist[idx].x.device)
        if pca_dim > 0:
            pca_transform = AddPCAProjection(n_components=pca_dim, attr_name="pca_pe")
            datalist[idx] = pca_transform(datalist[idx])
            pca_features = datalist[idx].pca_pe
        
        dim = datalist[idx].x.size(dim=0)
        
        # Build GNN input features
        features = []
        
        # Generate random features
        if config.random_in_channels > 0:
            random_features = torch.rand(dim, config.random_in_channels, dtype=torch.float, device=datalist[idx].x.device)
            features.append(random_features)
        
        # Add beacon features
        if beacons.shape[1] > 0:
            features.append(beacons)
        
        # Add spectral features
        if spectral_features.shape[1] > 0:
            features.append(spectral_features)
        
        # Add random projection features
        if random_projection_features.shape[1] > 0:
            features.append(random_projection_features)
        
        # Add PCA features
        if pca_features.shape[1] > 0:
            features.append(pca_features)
        
        # Add CLIP embeddings (sliced to clip_in_channels dimensions for GNN input)
        if config.clip_in_channels > 0:
            features.append(clip_embeddings[:, :config.clip_in_channels])
        
        # Concatenate: [random] + [beacons] + [spectral] + [random_projection] + [clip_embeddings[:clip_in_channels]]
        datalist[idx].x = torch.cat(features, dim=1)
        datalist[idx].x_orig = torch.clone(datalist[idx].x)
        datalist[idx].num_nodes = dim
        
        # Precompute rewiring edges if enabled
        rewiring_precompute = getattr(config, 'rewiring_precompute', False)
        if rewiring_precompute:
            precompute_rewiring_edges(datalist[idx], config)
    
    return datalist


class PreprocessingWrapperDR:
    """Wrapper that applies full preprocessing (random/beacon/spectral) to dynamic DR datasets on-the-fly."""
    
    def __init__(self, dataset, config):
        """
        Initialize preprocessing wrapper for DR datasets.
        
        Args:
            dataset: Dynamic dataset that generates Data objects with CLIP embeddings in x
            config: Configuration object with feature settings
        """
        self.dataset = dataset
        self.config = config
        self.bfs = BFS()
    
    def __len__(self):
        """Return dataset length."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get preprocessed data sample with all preprocessing features."""
        # Get raw data from dynamic dataset (has CLIP embeddings in x)
        data = self.dataset[idx]
        
        # Store original CLIP embeddings
        clip_embeddings = data.x.clone()
        data.clip_embedding = clip_embeddings
        
        num_nodes = data.x.shape[0]
        device_data = data.x.device
        
        # Initialize feature list
        features = []
        
        # 1. Add random features if enabled
        if self.config.random_in_channels > 0:
            random_features = torch.rand(num_nodes, self.config.random_in_channels, 
                                        dtype=torch.float, device=device_data)
            features.append(random_features)
        
        # 2. Add beacon features if enabled
        if self.config.use_beacons:
            beacon_features = self._compute_beacon_features(data)
            features.append(beacon_features)
        
        # 3. Add spectral features if enabled
        if self.config.laplace_eigvec > 0:
            spectral_features = self._compute_spectral_features(data)
            features.append(spectral_features)
        
        # 4. Add random projection features if enabled
        random_projection_dim = getattr(self.config, 'random_projection_dim', 0)
        if random_projection_dim > 0:
            rp_features = self._compute_random_projection_features(data, clip_embeddings)
            features.append(rp_features)
        
        # 5. Add PCA features if enabled
        pca_dim = getattr(self.config, 'pca_dim', 0)
        if pca_dim > 0:
            pca_features = self._compute_pca_features(data, clip_embeddings)
            features.append(pca_features)
        
        # 6. Add CLIP embeddings (sliced to clip_in_channels dimensions for GNN input)
        if self.config.clip_in_channels > 0:
            features.append(clip_embeddings[:, :self.config.clip_in_channels])
        
        # Concatenate all features
        data.x = torch.cat(features, dim=1)
        data.x_orig = torch.clone(data.x)
        
        # Precompute rewiring edges if enabled
        rewiring_precompute = getattr(self.config, 'rewiring_precompute', False)
        if rewiring_precompute:
            precompute_rewiring_edges(data, self.config)
        
        return data
    
    def _compute_beacon_features(self, data):
        """Compute beacon positional encoding features for a single graph."""
        num_nodes = data.num_nodes
        num_beacons = self.config.num_beacons
        encoding_size_per_beacon = self.config.encoding_size_per_beacon
        
        # Select random beacon nodes
        starting_nodes = random.sample(range(num_nodes), num_beacons)
        
        # Initialize distances
        distances = torch.empty(num_nodes, num_beacons, device=data.x.device).fill_(float('Inf'))
        for i in range(num_beacons):
            distances[starting_nodes[i], i] = 0
        
        # Run BFS to compute shortest paths
        bfs_distances = self.bfs(data, distances, num_nodes)
        
        # Compute positional encodings
        div_term = torch.exp(
            torch.arange(0, encoding_size_per_beacon, 2) * 
            (-math.log(10000.0) / encoding_size_per_beacon)
        ).to(bfs_distances.device)
        
        pes = []
        for beacon_index in range(num_beacons):
            pe = torch.zeros(num_nodes, encoding_size_per_beacon, device=bfs_distances.device)
            pe[:, 0::2] = torch.sin(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
            pe[:, 1::2] = torch.cos(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
            pes.append(pe)
        
        return torch.cat(pes, dim=1)
    
    def _compute_spectral_features(self, data):
        """Compute spectral (Laplacian eigenvector) features for a single graph."""
        pe_transform = AddLaplacian(k=self.config.laplace_eigvec, 
                                   attr_name="laplace_ev", 
                                   is_undirected=True, 
                                   use_cupy=self.config.use_cupy)
        data = pe_transform(data)
        return data.laplace_ev
    
    def _compute_random_projection_features(self, data, clip_embeddings):
        """Compute Gaussian Random Projection features from CLIP embeddings (GPU)."""
        random_projection_dim = getattr(self.config, 'random_projection_dim', 0)
        rp_transform = AddGaussianRandomProjection(n_components=random_projection_dim, attr_name="random_projection_pe")
        # Temporarily set clip_embedding so AddGaussianRandomProjection can use it
        data.clip_embedding = clip_embeddings
        data = rp_transform(data)
        return data.random_projection_pe
    
    def _compute_pca_features(self, data, clip_embeddings):
        """Compute PCA features from CLIP embeddings (per-graph PCA)."""
        pca_dim = getattr(self.config, 'pca_dim', 0)
        pca_transform = AddPCAProjection(n_components=pca_dim, attr_name="pca_pe")
        # Temporarily set clip_embedding so AddPCAProjection can use it
        data.clip_embedding = clip_embeddings
        data = pca_transform(data)
        return data.pca_pe


class PreprocessingWrapperDRGPU:
    """
    GPU-optimized preprocessing wrapper for dynamic DR datasets.
    
    Applies full preprocessing (random/beacon/spectral) on-the-fly while keeping 
    all tensors on GPU - no CPU transfers.
    """
    
    def __init__(self, dataset, config):
        """
        Initialize GPU-optimized preprocessing wrapper for DR datasets.
        
        Args:
            dataset: GPU-resident dynamic dataset (MNISTClipDynamicGPU)
            config: Configuration object with feature settings
        """
        self.dataset = dataset
        self.config = config
        self.bfs = BFS()
    
    def __len__(self):
        """Return dataset length."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get preprocessed data sample with all features on GPU (no CPU transfers)."""
        # Get raw data from GPU dynamic dataset (all tensors already on GPU)
        data = self.dataset[idx]
        
        # Store original CLIP embeddings (already on GPU)
        clip_embeddings = data.x.clone()  # GPU
        data.clip_embedding = clip_embeddings  # GPU
        
        num_nodes = data.x.shape[0]
        device_data = data.x.device  # GPU device
        
        # Initialize feature list
        features = []
        
        # 1. Add random features if enabled (keep on GPU)
        if self.config.random_in_channels > 0:
            random_features = torch.rand(num_nodes, self.config.random_in_channels, 
                                        dtype=torch.float, device=device_data)  # GPU
            features.append(random_features)
        
        # 2. Add beacon features if enabled (keep on GPU)
        if self.config.use_beacons:
            beacon_features = self._compute_beacon_features_gpu(data)  # GPU
            features.append(beacon_features)
        
        # 3. Add spectral features if enabled (keep on GPU)
        if self.config.laplace_eigvec > 0:
            spectral_features = self._compute_spectral_features_gpu(data)  # GPU
            features.append(spectral_features)
        
        # 4. Add random projection features if enabled (fully GPU, no CPU transfer)
        random_projection_dim = getattr(self.config, 'random_projection_dim', 0)
        if random_projection_dim > 0:
            rp_features = self._compute_random_projection_features_gpu(data, clip_embeddings)  # GPU
            features.append(rp_features)
        
        # 5. Add PCA features if enabled (fully GPU)
        pca_dim = getattr(self.config, 'pca_dim', 0)
        if pca_dim > 0:
            pca_features = self._compute_pca_features_gpu(data, clip_embeddings)  # GPU
            features.append(pca_features)
        
        # 6. Add CLIP embeddings (sliced to clip_in_channels dimensions for GNN input, GPU)
        if self.config.clip_in_channels > 0:
            features.append(clip_embeddings[:, :self.config.clip_in_channels])  # GPU
        
        # Concatenate all features (stays on GPU)
        data.x = torch.cat(features, dim=1)  # GPU
        data.x_orig = torch.clone(data.x)  # GPU
        
        # Precompute rewiring edges if enabled
        rewiring_precompute = getattr(self.config, 'rewiring_precompute', False)
        if rewiring_precompute:
            precompute_rewiring_edges(data, self.config)
        
        return data
    
    def _compute_beacon_features_gpu(self, data):
        """Compute beacon positional encoding features on GPU."""
        num_nodes = data.num_nodes
        num_beacons = self.config.num_beacons
        encoding_size_per_beacon = self.config.encoding_size_per_beacon
        device_data = data.x.device
        
        # Select random beacon nodes (GPU)
        starting_nodes = random.sample(range(num_nodes), num_beacons)
        
        # Initialize distances (GPU)
        distances = torch.empty(num_nodes, num_beacons, device=device_data).fill_(float('Inf'))
        for i in range(num_beacons):
            distances[starting_nodes[i], i] = 0
        
        # Run BFS to compute shortest paths (GPU)
        bfs_distances = self.bfs(data, distances, num_nodes)
        
        # Compute positional encodings (GPU)
        div_term = torch.exp(
            torch.arange(0, encoding_size_per_beacon, 2, device=device_data) * 
            (-math.log(10000.0) / encoding_size_per_beacon)
        ).to(bfs_distances.device)
        
        pes = []
        for beacon_index in range(num_beacons):
            pe = torch.zeros(num_nodes, encoding_size_per_beacon, device=bfs_distances.device)
            pe[:, 0::2] = torch.sin(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
            pe[:, 1::2] = torch.cos(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
            pes.append(pe)
        
        return torch.cat(pes, dim=1)  # GPU
    
    def _compute_spectral_features_gpu(self, data):
        """Compute spectral (Laplacian eigenvector) features on GPU."""
        pe_transform = AddLaplacian(k=self.config.laplace_eigvec, 
                                   attr_name="laplace_ev", 
                                   is_undirected=True, 
                                   use_cupy=self.config.use_cupy)
        data = pe_transform(data)
        return data.laplace_ev  # GPU
    
    def _compute_random_projection_features_gpu(self, data, clip_embeddings):
        """Compute Gaussian Random Projection features from CLIP embeddings on GPU.
        
        This is fully GPU-optimized - no CPU transfers required.
        Uses Johnson-Lindenstrauss random projection for distance-preserving DR.
        """
        random_projection_dim = getattr(self.config, 'random_projection_dim', 0)
        rp_transform = AddGaussianRandomProjection(n_components=random_projection_dim, attr_name="random_projection_pe")
        # Temporarily set clip_embedding so AddGaussianRandomProjection can use it
        data.clip_embedding = clip_embeddings
        data = rp_transform(data)
        return data.random_projection_pe  # GPU
    
    def _compute_pca_features_gpu(self, data, clip_embeddings):
        """Compute PCA features from CLIP embeddings on GPU (per-graph PCA).
        
        This is fully GPU-optimized - uses torch.linalg.svd for GPU-accelerated SVD.
        """
        pca_dim = getattr(self.config, 'pca_dim', 0)
        pca_transform = AddPCAProjection(n_components=pca_dim, attr_name="pca_pe")
        # Temporarily set clip_embedding so AddPCAProjection can use it
        data.clip_embedding = clip_embeddings
        data = pca_transform(data)
        return data.pca_pe  # GPU


def precompute_rewiring_edges(data, config):
    """
    Precompute rewired edges for a single graph during preprocessing.
    
    This function samples negative edges once and stores them in the data object,
    eliminating the need to recompute during each forward pass.
    
    Currently supports:
    - 'neg_sample': Sample edges that are not in the graph (for repulsive forces in DR)
    
    Args:
        data: PyG Data object with edge_index
        config: Config namespace with rewiring parameters:
            - rewiring: Rewiring type ('neg_sample' supported)
            - neg_sample_multiplier: Multiplier for number of negative samples
            - neg_sample_force_undirected: Whether to make edges undirected
            
    Returns:
        Data object with precomputed_rewiring_edge_index attribute added
    """
    from torch_geometric.utils import to_undirected, negative_sampling
    
    rewiring = getattr(config, 'rewiring', 'none')
    
    if rewiring == 'neg_sample':
        # Get config parameters
        neg_sample_multiplier = getattr(config, 'neg_sample_multiplier', 1.0)
        neg_sample_force_undirected = getattr(config, 'neg_sample_force_undirected', True)
        
        # Number of negative samples based on number of edges
        num_neg_samples = max(1, int(data.edge_index.shape[1] * neg_sample_multiplier))
        
        # Sample negative edges using PyG's negative_sampling
        # This samples edges that don't exist in the graph
        neg_edges = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=num_neg_samples,
            method='sparse'
        )
        
        # Make undirected if configured
        if neg_sample_force_undirected:
            neg_edges = to_undirected(neg_edges)
        
        # Store in data object
        data.precomputed_rewiring_edge_index = neg_edges
        
    else:
        # Other rewiring methods (knn) depend on positions
        # and cannot be precomputed statically
        pass
    
    return data


def precompute_rewiring_edges_batch(datalist, config):
    """
    Precompute rewired edges for a list of graphs.
    
    Args:
        datalist: List of PyG Data objects
        config: Config namespace with rewiring parameters
        
    Returns:
        List of Data objects with precomputed_rewiring_edge_index
    """
    rewiring_precompute = getattr(config, 'rewiring_precompute', False)
    
    if not rewiring_precompute:
        return datalist
    
    for data in datalist:
        precompute_rewiring_edges(data, config)
    
    return datalist
