"""Generic torchvision dataset with CLIP embeddings for dimensionality reduction.

This module provides a base class for any torchvision image dataset. Subclasses
only need to define how to load the specific dataset, and all CLIP embedding
extraction, caching, and graph construction is handled automatically.
"""

import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import Optional, List
from abc import abstractmethod

from gnn_dr.datasets.clip_dr_base import CLIPDRDatasetGPUBase
from gnn_dr.datasets.clip_embedding_utils import (
    extract_and_cache_clip_embeddings,
    get_embeddings_cache_path,
    get_images_from_torchvision_dataset,
    get_labels_from_torchvision_dataset,
)
from gnn_dr.datasets.transforms import normalize_embeddings


class TorchvisionCLIPDatasetGPU(CLIPDRDatasetGPUBase):
    """
    Generic base class for torchvision image datasets with CLIP embeddings.
    
    Subclasses only need to implement:
    - `_get_torchvision_dataset(train: bool)`: Return the specific torchvision dataset
    - `dataset_name` property: Return the dataset name for caching
    
    All CLIP embedding extraction, caching, and graph construction is handled
    by this base class and CLIPDRDatasetGPUBase.
    
    Example subclass:
        ```python
        class CIFAR10ClipDynamicGPU(TorchvisionCLIPDatasetGPU):
            @property
            def dataset_name(self):
                return "cifar10"
            
            def _get_torchvision_dataset(self, train: bool):
                return torchvision.datasets.CIFAR10(
                    root=str(self.root), train=train, download=True
                )
        ```
    """
    
    def __init__(
        self,
        root: str = 'data',
        train: bool = True,
        subset_sizes: Optional[List[int]] = None,
        knn_k: int = 15,
        clip_model: str = 'openai/CLIP-vit-base-patch32',
        seed: int = 42,
        n_samples_per_size: int = 10,
        device: str = 'cuda',
        edge_weight_method: str = 'umap',
        tsne_perplexity: float = 10.0,
    ):
        """
        Initialize torchvision CLIP dataset.

        Args:
            root: Data root directory
            train: Whether to use training or test split
            subset_sizes: List of subset sizes to cycle through during training
            knn_k: Number of neighbors for KNN graph
            clip_model: CLIP model to use for embedding extraction
            seed: Random seed for reproducibility
            n_samples_per_size: Number of graphs to generate per subset size
            device: GPU device ('cuda', 'cuda:0', etc.)
            edge_weight_method: 'umap' (fuzzy simplicial set) or 'tsne' (perplexity-based)
            tsne_perplexity: t-SNE perplexity (only used when edge_weight_method='tsne')
        """
        self.root = Path(root)
        self.train = train
        self.clip_model = clip_model

        # Initialize common parameters from base class
        self._init_common(
            subset_sizes=subset_sizes,
            knn_k=knn_k,
            seed=seed,
            n_samples_per_size=n_samples_per_size,
            device=device,
            edge_weight_method=edge_weight_method,
            tsne_perplexity=tsne_perplexity,
        )
        
        # Load embeddings and labels
        self._load_embeddings_and_labels()
    
    @abstractmethod
    def _get_torchvision_dataset(self, train: bool):
        """
        Return the torchvision dataset for the specified split.
        
        Subclasses must implement this method to return the specific dataset.
        The dataset should return (image, label) pairs.
        
        Args:
            train: Whether to load the training or test split
            
        Returns:
            A torchvision dataset instance
        """
        pass
    
    def _load_embeddings_and_labels(self):
        """Load or extract CLIP embeddings and keep on GPU."""
        # Get cache path
        split = 'train' if self.train else 'test'
        cache_path = get_embeddings_cache_path(
            cache_dir=self.root / self.dataset_name,
            dataset_name=self.dataset_name,
            split=split,
            clip_model=self.clip_model
        )
        
        # Check if embeddings are cached
        if cache_path.exists():
            print(f"Loading cached {self.dataset_name} CLIP embeddings from {cache_path}")
            embeddings = torch.load(cache_path, weights_only=True)
            embeddings = normalize_embeddings(embeddings)
            
            # Load labels from dataset (they're not cached, but fast to load)
            tv_dataset = self._get_torchvision_dataset(self.train)
            self.labels = get_labels_from_torchvision_dataset(tv_dataset)
        else:
            print(f"Extracting CLIP embeddings for {self.dataset_name}...")
            
            # Load torchvision dataset
            tv_dataset = self._get_torchvision_dataset(self.train)
            
            # Extract images (as PIL for CLIP preprocessing)
            images = get_images_from_torchvision_dataset(tv_dataset, to_pil=True)
            
            # Extract and cache embeddings
            embeddings = extract_and_cache_clip_embeddings(
                images=images,
                cache_path=cache_path,
                clip_model=self.clip_model,
                device=self.device,
                batch_size=256
            )
            
            # Extract labels
            self.labels = get_labels_from_torchvision_dataset(tv_dataset)
        
        # Move embeddings to GPU and keep there for entire training
        print(f"Moving {self.dataset_name} embeddings to {self.device} ({embeddings.shape[0]} embeddings)")
        self.embeddings = embeddings.to(self.device)
        
        # Move labels to GPU for faster access
        self.labels = self.labels.to(self.device)
        
        print(f"Loaded {self.dataset_name}: {self.embeddings.shape[0]} embeddings, dim={self.embeddings.shape[1]}")


# Registry of available torchvision CLIP datasets
# This allows easy lookup by name for the multi-dataset training
_TORCHVISION_CLIP_DATASETS = {}


def register_torchvision_clip_dataset(name: str):
    """
    Decorator to register a torchvision CLIP dataset class.
    
    Usage:
        @register_torchvision_clip_dataset('cifar10')
        class CIFAR10ClipDynamicGPU(TorchvisionCLIPDatasetGPU):
            ...
    """
    def decorator(cls):
        _TORCHVISION_CLIP_DATASETS[name] = cls
        return cls
    return decorator


def get_registered_clip_dataset(name: str):
    """Get a registered CLIP dataset class by name."""
    if name not in _TORCHVISION_CLIP_DATASETS:
        raise ValueError(f"Unknown CLIP dataset: {name}. Available: {list(_TORCHVISION_CLIP_DATASETS.keys())}")
    return _TORCHVISION_CLIP_DATASETS[name]


def list_registered_clip_datasets() -> List[str]:
    """List all registered CLIP dataset names."""
    return list(_TORCHVISION_CLIP_DATASETS.keys())
