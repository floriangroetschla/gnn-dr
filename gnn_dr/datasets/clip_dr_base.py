"""Base class for CLIP-based dimensionality reduction datasets.

This module provides base classes for GPU-optimized CLIP DR datasets.
The base class handles all graph construction logic, so subclasses only
need to implement embedding loading.
"""

import torch
import random
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_cluster import knn

from gnn_dr.utils.umap_weights import compute_edge_weights


class CLIPDRDatasetGPUBase(ABC):
    """
    Base class for GPU-optimized CLIP DR datasets.
    
    Provides shared graph construction logic for all CLIP-based datasets.
    Subclasses only need to implement the `_load_embeddings_and_labels()` method.
    
    Attributes:
        embeddings: Tensor of CLIP embeddings on GPU [N, D]
        labels: Optional tensor of labels on GPU [N] (None for unlabeled datasets)
        device: GPU device string
        knn_k: Number of neighbors for KNN graph
        subset_sizes: List of subset sizes for training
        n_samples_per_size: Number of samples to generate per subset size
        total_length: Total number of graphs (len(subset_sizes) * n_samples_per_size)
    """
    
    # Subclasses should set these
    embeddings: torch.Tensor
    labels: Optional[torch.Tensor]
    device: str
    knn_k: int
    subset_sizes: List[int]
    n_samples_per_size: int
    total_length: int
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return the dataset name (used for caching and logging)."""
        pass
    
    def _init_common(
        self,
        subset_sizes: Optional[List[int]] = None,
        knn_k: int = 15,
        seed: int = 42,
        n_samples_per_size: int = 10,
        device: str = 'cuda',
        edge_weight_method: str = 'umap',
        tsne_perplexity: float = 10.0,
    ):
        """
        Initialize common parameters. Call this from subclass __init__.

        Args:
            subset_sizes: List of subset sizes to cycle through during training
            knn_k: Number of neighbors for KNN graph
            seed: Random seed for reproducibility
            n_samples_per_size: Number of graphs to generate per subset size
            device: GPU device ('cuda', 'cuda:0', etc.)
            edge_weight_method: 'umap' (fuzzy simplicial set) or 'tsne' (perplexity-based)
            tsne_perplexity: t-SNE perplexity (only used when edge_weight_method='tsne')
        """
        self.subset_sizes = subset_sizes or [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        self.knn_k = knn_k
        self.seed = seed
        self.n_samples_per_size = n_samples_per_size
        self.device = device
        self.edge_weight_method = edge_weight_method
        self.tsne_perplexity = tsne_perplexity
        
        # Verify GPU device (fall back to CPU if unavailable)
        if not torch.cuda.is_available() or 'cuda' not in str(device):
            import logging
            logging.getLogger(__name__).warning(
                f"CUDA not available, {self.__class__.__name__} will run on CPU (slower)"
            )
            self.device = 'cpu'

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        self.total_length = len(self.subset_sizes) * n_samples_per_size
    
    def _build_graph_from_embeddings(
        self, 
        subset_embeddings: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Data:
        """
        Build KNN graph with UMAP weights from embeddings.
        
        This is the shared graph construction logic used by all CLIP DR datasets.
        All computation happens on GPU for maximum performance.
        
        Args:
            subset_embeddings: [N, D] tensor of embeddings (on GPU)
            labels: [N] tensor of labels (on GPU) or None for dummy labels
            
        Returns:
            PyG Data object with graph structure and UMAP weights (all on GPU)
        """
        subset_size = subset_embeddings.shape[0]
        
        # Ensure embeddings are float32
        subset_embeddings = subset_embeddings.float() if subset_embeddings.dtype != torch.float32 else subset_embeddings
        
        # Keep batch_idx on GPU
        batch_idx = torch.zeros(subset_size, dtype=torch.long, device=self.device)
        
        # Build DIRECTED KNN graph on GPU
        # IMPORTANT: Keep directed for UMAP weight computation (needs directed edges!)
        edge_index_knn = knn(
            subset_embeddings, 
            subset_embeddings,
            k=min(self.knn_k, subset_size - 1),
            batch_x=batch_idx, 
            batch_y=batch_idx
        )

        # Remove self-loops from directed KNN edges (keep directed!)
        edge_index_directed, _ = remove_self_loops(edge_index_knn)
        
        # Compute edge weights on DIRECTED edges - ALL ON GPU (no CPU transfers!)
        src = edge_index_directed[0]
        dst = edge_index_directed[1]
        
        # Compute cosine distances for directed edges on GPU
        # (All CLIP-based datasets use cosine metric)
        d_ij = 1.0 - torch.nn.functional.cosine_similarity(
            subset_embeddings[src], 
            subset_embeddings[dst]
        )
        
        # Compute edge weights - returns UNDIRECTED edges (u < v) with weights
        # Dispatches to UMAP fuzzy simplicial set or t-SNE perplexity-based weights
        try:
            edge_index_und, edge_weight_und = compute_edge_weights(
                method=self.edge_weight_method,
                edge_index=edge_index_directed,  # DIRECTED input, GPU tensor
                d_ij=d_ij,  # GPU tensor
                num_nodes=subset_size,
                k=self.knn_k,
                perplexity=self.tsne_perplexity,
            )
            
            # Convert to BIDIRECTIONAL for message passing: (u,v) AND (v,u) - all on GPU
            edge_index = torch.cat([edge_index_und, edge_index_und.flip(0)], dim=1)
            edge_weight = torch.cat([edge_weight_und, edge_weight_und])
        except Exception as e:
            print(f"Warning: UMAP weight computation failed ({e}), using uniform weights")
            edge_index = to_undirected(edge_index_directed)
            edge_weight = torch.ones(edge_index.shape[1], device=self.device)
        
        # Use provided labels or create dummy labels
        if labels is None:
            labels = torch.zeros(subset_size, dtype=torch.long, device=self.device)
        
        # Create Data object with all GPU tensors (no CPU transfers)
        data = Data(
            x=subset_embeddings,  # GPU
            edge_index=edge_index,  # GPU - BIDIRECTIONAL edges
            clip_embedding=subset_embeddings.clone(),  # GPU
            edge_weight=edge_weight,  # GPU - UMAP fuzzy simplicial set weights
            edge_attr=edge_weight.view(-1, 1),  # GPU - edge features for GNN
            batch=torch.zeros(subset_size, dtype=torch.long, device=self.device),  # GPU
            y=labels,  # GPU
        )
        
        return data
    
    def __len__(self) -> int:
        """Total number of graphs (cycles through all subset sizes)."""
        return self.total_length
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get graph by index with fully GPU-accelerated computation.
        
        Cycles through subset sizes: idx % len(subset_sizes) determines the size.
        
        Args:
            idx: Graph index
            
        Returns:
            PyG Data object with GPU tensors
        """
        # Cycle through subset sizes
        size_idx = idx % len(self.subset_sizes)
        subset_size = self.subset_sizes[size_idx]
        
        # Sample random subset (on GPU)
        n_total = self.embeddings.shape[0]
        indices = torch.randperm(n_total, device=self.device)[:subset_size]
        subset_embeddings = self.embeddings[indices]  # GPU
        
        # Get corresponding labels if available
        if self.labels is not None:
            subset_labels = self.labels[indices]
        else:
            subset_labels = None
        
        # Use base class method for graph construction
        return self._build_graph_from_embeddings(subset_embeddings, subset_labels)
    
    def get_full_graph(self) -> Data:
        """
        Get full dataset as single graph with fully GPU-accelerated computation.
        
        Returns:
            PyG Data object containing all embeddings with GPU tensors
        """
        # Ensure embeddings are float32
        embeddings = self.embeddings.float() if self.embeddings.dtype != torch.float32 else self.embeddings
        
        return self._build_graph_from_embeddings(embeddings, self.labels)
    
    def get_random_subset_graph(self, subset_size: int) -> Data:
        """
        Get a random subset as a single graph with fully GPU-accelerated computation.
        
        Args:
            subset_size: Number of nodes to include in the subset
            
        Returns:
            PyG Data object with GPU tensors
        """
        embeddings = self.embeddings.float() if self.embeddings.dtype != torch.float32 else self.embeddings
        
        n_nodes = embeddings.shape[0]
        
        # Sample random subset (on GPU)
        indices = torch.randperm(n_nodes, device=self.device)[:subset_size]
        subset_embeddings = embeddings[indices]  # GPU
        
        # Get corresponding labels if available
        if self.labels is not None:
            subset_labels = self.labels[indices]
        else:
            subset_labels = None
        
        return self._build_graph_from_embeddings(subset_embeddings, subset_labels)
    
    def get_num_embeddings(self) -> int:
        """Return the total number of embeddings in the dataset."""
        return self.embeddings.shape[0]
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embeddings.shape[1]
