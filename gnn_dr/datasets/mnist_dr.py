"""MNIST CLIP embeddings dataset for dimensionality reduction tasks.

This module provides MNIST dataset classes with CLIP embeddings for DR.
The main class `MNISTClipDynamicGPU` uses the new extensible base class architecture.
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path
from typing import Optional, List

from gnn_dr.datasets.torchvision_clip import (
    TorchvisionCLIPDatasetGPU,
    register_torchvision_clip_dataset,
)
from gnn_dr.datasets.clip_embedding_utils import CLIPEmbeddingExtractor


@register_torchvision_clip_dataset('mnist_clip')
class MNISTClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic MNIST CLIP dataset.
    
    Generates random subset graphs on-the-fly, keeping all tensors on GPU for maximum
    performance. CLIP embeddings are extracted once and cached to disk.
    
    This is the recommended class for training DR models on MNIST.
    
    Example:
        ```python
        # Create dataset for training
        train_dataset = MNISTClipDynamicGPU(
            root='data/MNIST',
            train=True,
            subset_sizes=[100, 500, 1000],
            knn_k=15,
        )
        
        # Get a training graph
        graph = train_dataset[0]
        
        # Get full dataset as single graph for validation
        full_graph = train_dataset.get_full_graph()
        ```
    """
    
    @property
    def dataset_name(self) -> str:
        return "mnist"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return MNIST dataset."""
        return datasets.MNIST(
            root=str(self.root),
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )


# =============================================================================
# Legacy classes for backward compatibility
# =============================================================================

class MNISTClipDynamic:
    """
    LEGACY: Dynamic MNIST CLIP dataset (CPU-based, for backward compatibility).
    
    Use MNISTClipDynamicGPU instead for better performance.
    """
    
    def __init__(self, 
                 root='data/MNIST',
                 train=True,
                 subset_sizes=None,
                 knn_k=15,
                 clip_model='openai/CLIP-vit-base-patch32',
                 seed=42,
                 n_samples_per_size=10,
                 device=None):
        """
        Initialize dynamic MNIST CLIP dataset.
        
        Note: This is a legacy class. Use MNISTClipDynamicGPU for new code.
        """
        import warnings
        warnings.warn(
            "MNISTClipDynamic is deprecated. Use MNISTClipDynamicGPU for better performance.",
            DeprecationWarning
        )
        
        # Create the GPU version internally
        self._gpu_dataset = MNISTClipDynamicGPU(
            root=root,
            train=train,
            subset_sizes=subset_sizes,
            knn_k=knn_k,
            clip_model=clip_model,
            seed=seed,
            n_samples_per_size=n_samples_per_size,
            device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Copy attributes for compatibility
        self.train = train
        self.subset_sizes = self._gpu_dataset.subset_sizes
        self.knn_k = knn_k
        self.clip_model = clip_model
        self.seed = seed
        self.n_samples_per_size = n_samples_per_size
        self.root = Path(root)
        self.device = self._gpu_dataset.device
        self.embeddings = self._gpu_dataset.embeddings
        self.labels = self._gpu_dataset.labels
        self.total_length = self._gpu_dataset.total_length
    
    def __len__(self):
        return len(self._gpu_dataset)
    
    def __getitem__(self, idx):
        return self._gpu_dataset[idx]
    
    def get_full_graph(self):
        return self._gpu_dataset.get_full_graph()
    
    def get_random_subset_graph(self, subset_size):
        return self._gpu_dataset.get_random_subset_graph(subset_size)


# Keep the old CLIPEmbeddingExtractor import working
# (it's now in clip_embedding_utils.py, but legacy code might import from here)
__all__ = [
    'MNISTClipDynamicGPU',
    'MNISTClipDynamic',
    'CLIPEmbeddingExtractor',
]
