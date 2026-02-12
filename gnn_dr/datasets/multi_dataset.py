"""Multi-dataset wrapper for training on multiple CLIP datasets simultaneously.

This module provides a wrapper that samples graphs from multiple CLIP datasets
during training, enabling more robust DR models that generalize across datasets.
"""

import torch
import random
import numpy as np
from typing import List, Optional, Dict, Any, Union
from torch_geometric.data import Data

from gnn_dr.datasets.clip_dr_base import CLIPDRDatasetGPUBase


class MultiDatasetCLIPDynamicGPU:
    """
    Wrapper that samples graphs from multiple CLIP datasets.
    
    Supports weighted sampling from different datasets, allowing you to control
    how often each dataset is used during training.
    
    Example:
        ```python
        from gnn_dr.datasets.mnist_dr import MNISTClipDynamicGPU
        from gnn_dr.datasets.cifar_dr import CIFAR10ClipDynamicGPU
        
        # Create individual datasets
        mnist = MNISTClipDynamicGPU(root='data/MNIST', train=True)
        cifar = CIFAR10ClipDynamicGPU(root='data', train=True)
        
        # Combine them with equal weights
        multi_dataset = MultiDatasetCLIPDynamicGPU(
            datasets=[mnist, cifar],
            weights=[1.0, 1.0],  # Equal sampling probability
        )
        
        # Get a graph (sampled from one of the datasets)
        graph = multi_dataset[0]
        ```
    """
    
    def __init__(
        self,
        datasets: List[CLIPDRDatasetGPUBase],
        weights: Optional[List[float]] = None,
        seed: int = 42,
    ):
        """
        Initialize multi-dataset wrapper.
        
        Args:
            datasets: List of CLIP DR dataset instances
            weights: Optional sampling weights for each dataset (default: uniform)
            seed: Random seed for reproducibility
        """
        if not datasets:
            raise ValueError("At least one dataset is required")
        
        self.datasets = datasets
        self.num_datasets = len(datasets)
        
        # Normalize weights to probabilities
        if weights is None:
            weights = [1.0] * self.num_datasets
        
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        
        # Store dataset names for logging
        self.dataset_names = [d.dataset_name for d in datasets]
        
        # Compute total length as sum of all dataset lengths
        self.total_length = sum(len(d) for d in datasets)
        
        # Set random seed
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get common parameters from first dataset
        self.device = datasets[0].device
        self.knn_k = datasets[0].knn_k
        self.subset_sizes = datasets[0].subset_sizes
        
        # Track statistics
        self._sample_counts = {name: 0 for name in self.dataset_names}
    
    @property
    def dataset_name(self) -> str:
        """Return combined name of all datasets."""
        return "+".join(self.dataset_names)
    
    def __len__(self) -> int:
        """Total number of graphs across all datasets."""
        return self.total_length
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get a graph by sampling from one of the datasets based on weights.
        
        Note: The idx is not used to determine which dataset to sample from.
        Instead, we sample based on the configured weights. The idx is passed
        to the selected dataset to get a graph.
        
        Args:
            idx: Graph index (used to get graph from selected dataset)
            
        Returns:
            PyG Data object from one of the datasets
        """
        # Sample a dataset based on weights
        dataset_idx = np.random.choice(self.num_datasets, p=self.weights)
        selected_dataset = self.datasets[dataset_idx]
        
        # Update statistics
        self._sample_counts[self.dataset_names[dataset_idx]] += 1
        
        # Get a graph from the selected dataset
        # Use modulo to wrap index within dataset length
        local_idx = idx % len(selected_dataset)
        data = selected_dataset[local_idx]
        
        # Add metadata about which dataset this came from
        data.source_dataset = self.dataset_names[dataset_idx]
        data.source_dataset_idx = dataset_idx
        
        return data
    
    def get_full_graph(self, dataset_idx: int = 0) -> Data:
        """
        Get full graph from a specific dataset.
        
        Args:
            dataset_idx: Index of dataset to get full graph from
            
        Returns:
            PyG Data object with all embeddings from the specified dataset
        """
        return self.datasets[dataset_idx].get_full_graph()
    
    def get_random_subset_graph(self, subset_size: int, dataset_idx: Optional[int] = None) -> Data:
        """
        Get a random subset graph from a specific or randomly sampled dataset.
        
        Args:
            subset_size: Number of nodes in the subset
            dataset_idx: Index of dataset (if None, samples based on weights)
            
        Returns:
            PyG Data object with subset of embeddings
        """
        if dataset_idx is None:
            dataset_idx = np.random.choice(self.num_datasets, p=self.weights)
        
        data = self.datasets[dataset_idx].get_random_subset_graph(subset_size)
        data.source_dataset = self.dataset_names[dataset_idx]
        data.source_dataset_idx = dataset_idx
        
        return data
    
    def get_sample_counts(self) -> Dict[str, int]:
        """Get the number of samples drawn from each dataset."""
        return self._sample_counts.copy()
    
    def reset_sample_counts(self):
        """Reset sample count statistics."""
        self._sample_counts = {name: 0 for name in self.dataset_names}
    
    def get_dataset_by_name(self, name: str) -> CLIPDRDatasetGPUBase:
        """Get a specific dataset by name."""
        for dataset in self.datasets:
            if dataset.dataset_name == name:
                return dataset
        raise ValueError(f"Dataset '{name}' not found. Available: {self.dataset_names}")
    
    def __repr__(self) -> str:
        dataset_info = ", ".join([
            f"{name}({len(d)} graphs, weight={w:.2f})"
            for name, d, w in zip(self.dataset_names, self.datasets, self.weights)
        ])
        return f"MultiDatasetCLIPDynamicGPU([{dataset_info}])"


def create_multi_dataset_from_config(
    dataset_configs: List[Dict[str, Any]],
    common_params: Dict[str, Any],
) -> MultiDatasetCLIPDynamicGPU:
    """
    Create a multi-dataset from configuration dictionaries.
    
    Args:
        dataset_configs: List of dicts with 'name' and optional 'weight' keys
        common_params: Common parameters for all datasets (root, knn_k, etc.)
        
    Returns:
        MultiDatasetCLIPDynamicGPU instance
        
    Example config:
        dataset_configs = [
            {'name': 'mnist_clip', 'weight': 1.0},
            {'name': 'cifar10_clip', 'weight': 2.0},
        ]
        common_params = {
            'root': 'data',
            'knn_k': 15,
            'subset_sizes': [100, 500, 1000],
            'n_samples_per_size': 100,
            'device': 'cuda',
        }
    """
    from gnn_dr.datasets.torchvision_clip import get_registered_clip_dataset
    
    datasets = []
    weights = []
    
    for config in dataset_configs:
        name = config['name']
        weight = config.get('weight', 1.0)
        
        # Get the dataset class
        dataset_cls = get_registered_clip_dataset(name)
        
        # Create dataset instance with common params
        # Handle root directory per-dataset if needed
        params = common_params.copy()
        if 'root' in config:
            params['root'] = config['root']
        
        dataset = dataset_cls(**params)
        datasets.append(dataset)
        weights.append(weight)
    
    return MultiDatasetCLIPDynamicGPU(
        datasets=datasets,
        weights=weights,
        seed=common_params.get('seed', 42),
    )


__all__ = [
    'MultiDatasetCLIPDynamicGPU',
    'create_multi_dataset_from_config',
]
