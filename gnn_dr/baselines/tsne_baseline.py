"""
t-SNE baseline for dimensionality reduction evaluation.

Uses scikit-learn's t-SNE implementation.
"""

import time
import numpy as np
from typing import Optional

from sklearn.manifold import TSNE

from .base import BaselineMethod


class TSNEBaseline(BaselineMethod):
    """
    t-SNE (t-distributed Stochastic Neighbor Embedding) baseline.
    
    This is a transductive method - it must be fit on the data it will transform.
    For fair comparison with inductive methods, we fit directly on the test data.
    
    Reference:
        van der Maaten, L., & Hinton, G. (2008). 
        Visualizing data using t-SNE.
        Journal of Machine Learning Research, 9(Nov), 2579-2605.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,  # Keep for API compatibility
        early_exaggeration: float = 12.0,
        metric: str = 'euclidean',
        init: str = 'pca',
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        verbose: int = 0,
    ):
        """
        Initialize t-SNE baseline.
        
        Args:
            n_components: Dimension of the embedded space (usually 2)
            perplexity: Related to the number of nearest neighbors. 
                       Typical values are between 5 and 50.
            learning_rate: Learning rate for t-SNE optimization.
                          Typical values are between 10 and 1000.
            n_iter: Maximum number of iterations for optimization.
            early_exaggeration: Controls tightness of clusters in early phase.
            metric: Distance metric to use.
            init: Initialization method ('random', 'pca', or ndarray).
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel jobs (-1 = all cores).
            verbose: Verbosity level.
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = n_iter  # scikit-learn 1.2+ uses max_iter instead of n_iter
        self.early_exaggeration = early_exaggeration
        self.metric = metric
        self.init = init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self._model = None
        self._fit_time = 0.0
        self._inference_time = 0.0
        self._embedding = None
    
    def fit(self, train_data: np.ndarray) -> 'TSNEBaseline':
        """
        Fit t-SNE to training data.
        
        Note: t-SNE is transductive, so fit and transform happen together.
        We store the data here and the actual fitting happens in transform().
        
        Args:
            train_data: Training data array of shape (n_samples, n_features)
            
        Returns:
            self
        """
        # t-SNE is transductive - fitting happens during transform
        # Just record that we've been "fit"
        self._fit_time = 0.0
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using t-SNE.
        
        Since t-SNE is transductive, this actually performs the fitting
        on the provided data.
        
        Args:
            data: Data array of shape (n_samples, n_features)
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        start_time = time.time()
        
        # Adjust perplexity if dataset is too small
        effective_perplexity = min(self.perplexity, (data.shape[0] - 1) / 3)
        if effective_perplexity < 5:
            effective_perplexity = 5
        
        self._model = TSNE(
            n_components=self.n_components,
            perplexity=effective_perplexity,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,  # scikit-learn 1.2+ uses max_iter
            early_exaggeration=self.early_exaggeration,
            metric=self.metric,
            init=self.init,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        
        self._embedding = self._model.fit_transform(data)
        
        total_time = time.time() - start_time
        self._fit_time = total_time
        self._inference_time = total_time  # For t-SNE, fit and transform are the same
        
        return self._embedding
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit t-SNE to data and transform it.
        
        Args:
            data: Data array of shape (n_samples, n_features)
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        self.fit(data)
        return self.transform(data)
    
    def get_train_time(self) -> float:
        """Return training time in seconds."""
        return self._fit_time
    
    def get_inference_time(self) -> float:
        """Return inference time in seconds."""
        return self._inference_time
    
    def get_name(self) -> str:
        """Return baseline name."""
        return f"t-SNE (perp={self.perplexity})"
