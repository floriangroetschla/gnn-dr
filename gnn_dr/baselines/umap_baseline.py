"""UMAP baseline for dimensionality reduction evaluation."""

import numpy as np
import time
from typing import Optional

from .base import BaselineMethod


class UMAPBaseline(BaselineMethod):
    """
    UMAP baseline for comparison.
    
    Uses the official umap-learn implementation.
    """
    
    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        random_state: Optional[int] = None,
        metric: str = 'euclidean',
    ):
        """
        Initialize UMAP baseline.
        
        Args:
            n_neighbors: Number of neighbors for KNN graph
            min_dist: Minimum distance parameter
            n_components: Output dimensionality (default: 2)
            random_state: Random seed for reproducibility
            metric: Distance metric ('euclidean', 'cosine', etc.)
        """
        super().__init__(name='UMAP', random_state=random_state)
        
        try:
            import umap
            self.umap = umap
        except ImportError:
            raise ImportError("umap-learn is not installed. Please install it with: pip install umap-learn")
        
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.metric = metric
        self.model = None
    
    def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit UMAP on high-dimensional embeddings.
        
        Args:
            embeddings: High-dimensional embeddings of shape (n_samples, n_features)
        """
        start_time = time.time()
        
        self.model = self.umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            metric=self.metric,
            random_state=self.random_state,
            verbose=0,
        )
        
        self.model.fit(embeddings)
        
        self._train_time = time.time() - start_time
        self.is_fitted = True
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings to low-dimensional space.
        
        Args:
            embeddings: High-dimensional embeddings
            
        Returns:
            Low-dimensional projections
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform")
        
        start_time = time.time()
        result = self.model.transform(embeddings)
        self._inference_time = time.time() - start_time
        
        return result
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            embeddings: High-dimensional embeddings
            
        Returns:
            Low-dimensional projections
        """
        start_time = time.time()
        
        self.model = self.umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            metric=self.metric,
            random_state=self.random_state,
            verbose=0,
        )
        
        result = self.model.fit_transform(embeddings)
        
        self._train_time = time.time() - start_time
        self.is_fitted = True
        
        return result
