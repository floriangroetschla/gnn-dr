"""Evaluation metrics for dimensionality reduction."""

import numpy as np
import torch
from typing import Dict, Optional, Union
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


class DRMetrics:
    """
    Comprehensive evaluation metrics for dimensionality reduction quality.
    
    Computes both standard DR metrics and UMAP loss.
    """
    
    def __init__(self, k_neighbors: int = 15, metric: str = 'euclidean'):
        """
        Initialize metrics calculator.
        
        Args:
            k_neighbors: Number of neighbors for local structure metrics
            metric: Distance metric to use ('euclidean' or 'cosine')
        """
        self.k_neighbors = k_neighbors
        self.metric = metric
    
    @staticmethod
    def _compute_distances(embeddings: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
        """
        Compute pairwise distances efficiently.
        
        Args:
            embeddings: Data array of shape (n_samples, n_features)
            metric: Distance metric
            
        Returns:
            Pairwise distance matrix of shape (n_samples, n_samples)
        """
        if metric == 'euclidean':
            distances = squareform(pdist(embeddings, metric='euclidean'))
        elif metric == 'cosine':
            # Normalize rows
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            # Compute cosine distance as 1 - cosine similarity
            distances = squareform(pdist(embeddings_norm, metric='cosine'))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        return distances
    
    def trustworthiness(self, high_dim: np.ndarray, low_dim: np.ndarray) -> float:
        """
        Compute trustworthiness metric.
        
        Measures the extent to which the k nearest neighbors in the low-dimensional
        space are also neighbors in the high-dimensional space.
        
        Range: [0, 1], higher is better.
        
        Args:
            high_dim: High-dimensional embeddings (n_samples, n_features_high)
            low_dim: Low-dimensional projections (n_samples, n_features_low)
            
        Returns:
            Trustworthiness score
        """
        k = self.k_neighbors
        n = high_dim.shape[0]
        
        high_dists = self._compute_distances(high_dim, metric=self.metric)
        low_dists = self._compute_distances(low_dim, metric='euclidean')
        
        # Get k-NN indices in both spaces
        high_neighbors = np.argsort(high_dists, axis=1)[:, 1:k+1]
        low_neighbors = np.argsort(low_dists, axis=1)[:, 1:k+1]
        
        # For each sample, compute rank of low-dim neighbors in high-dim space
        trust = 0.0
        for i in range(n):
            # Find which low-dim neighbors are NOT in high-dim neighbors
            for j, neighbor_idx in enumerate(low_neighbors[i]):
                if neighbor_idx not in high_neighbors[i]:
                    # Rank of this neighbor in high-dim distance
                    rank = np.where(np.argsort(high_dists[i]) == neighbor_idx)[0][0]
                    trust += (rank - k)
        
        trust = 1.0 - (2.0 / (n * k * (2 * n - 3 * k - 1))) * trust
        
        return max(0.0, trust)  # Clamp to [0, 1]
    
    def continuity(self, high_dim: np.ndarray, low_dim: np.ndarray) -> float:
        """
        Compute continuity metric.
        
        Measures the extent to which the k nearest neighbors in the high-dimensional
        space are also neighbors in the low-dimensional space.
        
        Range: [0, 1], higher is better.
        
        Args:
            high_dim: High-dimensional embeddings
            low_dim: Low-dimensional projections
            
        Returns:
            Continuity score
        """
        k = self.k_neighbors
        n = high_dim.shape[0]
        
        high_dists = self._compute_distances(high_dim, metric=self.metric)
        low_dists = self._compute_distances(low_dim, metric='euclidean')
        
        # Get k-NN indices in both spaces
        high_neighbors = np.argsort(high_dists, axis=1)[:, 1:k+1]
        low_neighbors = np.argsort(low_dists, axis=1)[:, 1:k+1]
        
        # For each sample, compute rank of high-dim neighbors in low-dim space
        cont = 0.0
        for i in range(n):
            # Find which high-dim neighbors are NOT in low-dim neighbors
            for neighbor_idx in high_neighbors[i]:
                if neighbor_idx not in low_neighbors[i]:
                    # Rank of this neighbor in low-dim distance
                    rank = np.where(np.argsort(low_dists[i]) == neighbor_idx)[0][0]
                    cont += (rank - k)
        
        cont = 1.0 - (2.0 / (n * k * (2 * n - 3 * k - 1))) * cont
        
        return max(0.0, cont)  # Clamp to [0, 1]
    
    def knn_recall(self, high_dim: np.ndarray, low_dim: np.ndarray) -> float:
        """
        Compute k-NN recall (fraction of k nearest neighbors preserved).
        
        Range: [0, 1], higher is better.
        
        Args:
            high_dim: High-dimensional embeddings
            low_dim: Low-dimensional projections
            
        Returns:
            k-NN recall score
        """
        k = self.k_neighbors
        n = high_dim.shape[0]
        
        high_dists = self._compute_distances(high_dim, metric=self.metric)
        low_dists = self._compute_distances(low_dim, metric='euclidean')
        
        high_neighbors = np.argsort(high_dists, axis=1)[:, 1:k+1]
        low_neighbors = np.argsort(low_dists, axis=1)[:, 1:k+1]
        
        # Count how many high-dim neighbors are in low-dim neighbors
        matches = 0
        for i in range(n):
            matches += len(np.intersect1d(high_neighbors[i], low_neighbors[i]))
        
        return matches / (n * k)
    
    def distance_correlation(self, high_dim: np.ndarray, low_dim: np.ndarray) -> float:
        """
        Compute Spearman correlation between high-dim and low-dim distances.
        
        Measures global structure preservation. Range: [-1, 1], higher is better.
        
        Args:
            high_dim: High-dimensional embeddings
            low_dim: Low-dimensional projections
            
        Returns:
            Spearman correlation coefficient
        """
        # Get pairwise distances
        high_dists = pdist(high_dim, metric=self.metric)
        low_dists = pdist(low_dim, metric='euclidean')
        
        # Compute Spearman correlation
        corr, _ = spearmanr(high_dists, low_dists)
        
        return float(corr)
    
    def silhouette(self, low_dim: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute silhouette score for cluster separation.
        
        Range: [-1, 1], higher indicates better-separated clusters.
        
        Args:
            low_dim: Low-dimensional projections
            labels: Class labels for samples
            
        Returns:
            Silhouette score
        """
        if len(np.unique(labels)) < 2:
            return 0.0
        
        return silhouette_score(low_dim, labels)
    
    def umap_loss(
        self,
        high_dim: Union[np.ndarray, torch.Tensor],
        low_dim: Union[np.ndarray, torch.Tensor],
        k_neighbors: int = 15,
        neg_sample_rate: float = 5.0,
        repulsion_weight: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        metric: str = 'euclidean',
    ) -> float:
        """
        Compute UMAP loss for any projection using the codebase's loss function.
        
        Args:
            high_dim: High-dimensional embeddings
            low_dim: Low-dimensional projections
            k_neighbors: Number of neighbors for KNN graph
            neg_sample_rate: Negatives per positive sample (default: 5)
            repulsion_weight: Weight for repulsion term
            min_dist: Minimum distance threshold
            spread: Spread parameter for distance scaling
            metric: Distance metric for high-dim space ('euclidean' or 'cosine')
            
        Returns:
            UMAP loss value
        """
        from gnn_dr.network.losses import UMAPLoss
        from gnn_dr.utils.umap_weights import compute_umap_fuzzy_weights
        from torch_geometric.nn import knn_graph
        
        # Convert to torch tensors if needed
        if isinstance(high_dim, np.ndarray):
            high_dim = torch.from_numpy(high_dim).float()
        if isinstance(low_dim, np.ndarray):
            low_dim = torch.from_numpy(low_dim).float()
        
        n = high_dim.shape[0]
        device = low_dim.device if low_dim.is_cuda else 'cpu'
        high_dim = high_dim.to(device)
        low_dim = low_dim.to(device)
        
        # Build KNN graph in high-dim space using PyG
        edge_index = knn_graph(high_dim, k=k_neighbors, loop=False)
        
        if edge_index.shape[1] == 0:
            return 0.0
        
        # Compute UMAP fuzzy simplicial set weights (REQUIRED by UMAPLoss)
        edge_weight = compute_umap_fuzzy_weights(
            high_dim, 
            edge_index, 
            metric=metric
        )
        
        # Create a batch-like object with required attributes
        class SimpleBatch:
            def __init__(self, edge_index, edge_weight):
                self.edge_index = edge_index
                self.edge_weight = edge_weight
                self.batch = torch.zeros(n, dtype=torch.long, device=device)
        
        batch = SimpleBatch(edge_index, edge_weight)
        
        # Use the UMAP loss from the codebase
        loss_fn = UMAPLoss(
            num_positive_samples=min(1000, edge_index.shape[1]),
            num_negatives_per_edge=int(neg_sample_rate),
            repulsion_weight=repulsion_weight,
            min_dist=min_dist,
            spread=spread,
        )
        
        with torch.no_grad():
            loss = loss_fn(low_dim, batch)
        
        return float(loss.item()) if isinstance(loss, torch.Tensor) else float(loss)
    
    def compute_all(
        self,
        high_dim: np.ndarray,
        low_dim: np.ndarray,
        labels: Optional[np.ndarray] = None,
        compute_umap_loss: bool = False,
        umap_loss_kwargs: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Compute all available metrics at once.
        
        Args:
            high_dim: High-dimensional embeddings
            low_dim: Low-dimensional projections
            labels: Optional class labels for silhouette score
            compute_umap_loss: Whether to compute UMAP loss
            umap_loss_kwargs: Additional kwargs for UMAP loss computation
            
        Returns:
            Dictionary with all computed metrics
        """
        metrics = {
            'trustworthiness': self.trustworthiness(high_dim, low_dim),
            'continuity': self.continuity(high_dim, low_dim),
            'knn_recall': self.knn_recall(high_dim, low_dim),
            'distance_correlation': self.distance_correlation(high_dim, low_dim),
        }
        
        if labels is not None:
            metrics['silhouette'] = self.silhouette(low_dim, labels)
        
        if compute_umap_loss:
            umap_kwargs = umap_loss_kwargs or {}
            metrics['umap_loss'] = self.umap_loss(high_dim, low_dim, **umap_kwargs)
        
        return metrics
