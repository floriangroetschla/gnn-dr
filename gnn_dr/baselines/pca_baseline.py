"""PCA baseline for dimensionality reduction evaluation."""

import numpy as np
import time
from typing import Optional
from sklearn.decomposition import PCA

from .base import BaselineMethod


class PCABaseline(BaselineMethod):
    """
    PCA baseline for comparison.
    
    Simple linear baseline using Principal Component Analysis.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = None,
    ):
        """
        Initialize PCA baseline.
        
        Args:
            n_components: Output dimensionality (default: 2)
            random_state: Random seed for reproducibility
        """
        super().__init__(name='PCA', random_state=random_state)
        
        self.n_components = n_components
        self.model = PCA(n_components=n_components, random_state=random_state)
    
    def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit PCA on high-dimensional embeddings.
        
        Args:
            embeddings: High-dimensional embeddings of shape (n_samples, n_features)
        """
        start_time = time.time()
        
        self.model = PCA(n_components=self.n_components, random_state=self.random_state)
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
        
        self.model = PCA(n_components=self.n_components, random_state=self.random_state)
        result = self.model.fit_transform(embeddings)
        
        self._train_time = time.time() - start_time
        self.is_fitted = True
        
        return result
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get the explained variance ratio for each component."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        return self.model.explained_variance_ratio_
