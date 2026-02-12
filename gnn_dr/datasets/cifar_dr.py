"""CIFAR-10 and CIFAR-100 CLIP embeddings datasets for dimensionality reduction.

This module provides CIFAR dataset classes with CLIP embeddings for DR tasks.
Both CIFAR-10 (10 classes) and CIFAR-100 (100 fine-grained classes) are supported.
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


@register_torchvision_clip_dataset('cifar10_clip')
class CIFAR10ClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic CIFAR-10 CLIP dataset.
    
    CIFAR-10 contains 60,000 32x32 color images in 10 classes, with 6,000 images
    per class. There are 50,000 training images and 10,000 test images.
    
    Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    
    Generates random subset graphs on-the-fly, keeping all tensors on GPU for maximum
    performance. CLIP embeddings are extracted once and cached to disk.
    
    Example:
        ```python
        # Create dataset for training
        train_dataset = CIFAR10ClipDynamicGPU(
            root='data',
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
        return "cifar10"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return CIFAR-10 dataset."""
        return datasets.CIFAR10(
            root=str(self.root),
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )


@register_torchvision_clip_dataset('cifar100_clip')
class CIFAR100ClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic CIFAR-100 CLIP dataset.
    
    CIFAR-100 contains 60,000 32x32 color images in 100 fine-grained classes.
    There are 500 training images and 100 testing images per class.
    
    The 100 classes are grouped into 20 superclasses. Each image comes with a
    "fine" label (the class) and a "coarse" label (the superclass).
    
    Generates random subset graphs on-the-fly, keeping all tensors on GPU for maximum
    performance. CLIP embeddings are extracted once and cached to disk.
    
    Example:
        ```python
        # Create dataset for training
        train_dataset = CIFAR100ClipDynamicGPU(
            root='data',
            train=True,
            subset_sizes=[100, 500, 1000],
            knn_k=15,
        )
        
        # Get a training graph
        graph = train_dataset[0]
        ```
    """
    
    @property
    def dataset_name(self) -> str:
        return "cifar100"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return CIFAR-100 dataset."""
        return datasets.CIFAR100(
            root=str(self.root),
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )


__all__ = [
    'CIFAR10ClipDynamicGPU',
    'CIFAR100ClipDynamicGPU',
]
