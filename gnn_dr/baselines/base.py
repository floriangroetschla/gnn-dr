"""Abstract base class for dimensionality reduction baseline methods."""

from abc import ABC, abstractmethod
import torch
from typing import Optional, Dict, Tuple
import numpy as np


class BaselineMethod(ABC):
    """
    Abstract base class for all dimensionality reduction baseline methods.
    
    All baselines should inherit from this class and implement the required methods
    to ensure consistent interfaces across different DR algorithms.
    """
    
    def __init__(self, name: str, random_state: Optional[int] = None):
        """
        Initialize baseline method.
        
        Args:
            name: Name of the baseline method
            random_state: Random seed for reproducibility
        """
        self.name = name
        self.random_state = random_state
        self.is_fitted = False
        self._train_time = None
        self._inference_time = None
    
    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> None:
        """
        Fit the baseline method on high-dimensional embeddings.
        
        Args:
            embeddings: High-dimensional embeddings of shape (n_samples, n_features)
        """
        pass
    
    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform high-dimensional embeddings to low-dimensional space.
        
        Args:
            embeddings: High-dimensional embeddings of shape (n_samples, n_features)
            
        Returns:
            Low-dimensional projections of shape (n_samples, n_dims)
        """
        pass
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            embeddings: High-dimensional embeddings of shape (n_samples, n_features)
            
        Returns:
            Low-dimensional projections of shape (n_samples, n_dims)
        """
        self.fit(embeddings)
        return self.transform(embeddings)
    
    def get_train_time(self) -> Optional[float]:
        """Get training time in seconds."""
        return self._train_time
    
    def get_inference_time(self) -> Optional[float]:
        """Get inference time in seconds."""
        return self._inference_time
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class NeuralBaselineMethod(BaselineMethod):
    """
    Base class for neural network-based dimensionality reduction methods.
    
    Adds support for PyTorch Lightning models and GPU acceleration.
    """
    
    def __init__(self, name: str, random_state: Optional[int] = None, device: str = 'cpu'):
        """
        Initialize neural baseline method.
        
        Args:
            name: Name of the baseline method
            random_state: Random seed for reproducibility
            device: Device to use ('cpu' or 'cuda')
        """
        super().__init__(name, random_state)
        self.device = device
        self.model = None
    
    def to_device(self, data: torch.Tensor) -> torch.Tensor:
        """Move tensor to device."""
        return data.to(self.device)
    
    @abstractmethod
    def get_model_size(self) -> int:
        """Get model size in number of parameters."""
        pass
