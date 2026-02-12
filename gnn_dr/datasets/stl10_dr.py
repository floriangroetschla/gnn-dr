"""STL-10 CLIP embeddings dataset for dimensionality reduction.

The STL-10 dataset is an image recognition dataset for developing unsupervised 
feature learning, deep learning, and self-taught learning algorithms. It is 
inspired by the CIFAR-10 dataset but with some modifications:

Key differences from CIFAR-10:
- Higher resolution: 96x96 pixels vs 32x32 in CIFAR-10
- Fewer labeled training examples: 500 per class (5,000 total)
- Large set of unlabeled images (100,000) for unsupervised learning
- 10 classes, but slightly different from CIFAR-10

Dataset statistics:
- Training set: 5,000 labeled images (500 per class)
- Test set: 8,000 images (800 per class)
- Unlabeled set: 100,000 images (can contain images from classes outside the 10)
- Image size: 96x96 RGB

Classes:
    airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck

Reference:
    Coates, Lee, Ng, "An Analysis of Single-Layer Networks in Unsupervised 
    Feature Learning", AISTATS 2011
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


@register_torchvision_clip_dataset('stl10_clip')
class STL10ClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic STL-10 CLIP dataset.
    
    STL-10 contains 13,000 labeled 96x96 RGB images in 10 classes, plus
    100,000 unlabeled images. The higher resolution compared to CIFAR-10
    makes it useful for testing with more detailed visual information.
    
    Classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
    
    Generates random subset graphs on-the-fly, keeping all tensors on GPU for maximum
    performance. CLIP embeddings are extracted once and cached to disk.
    
    Example:
        ```python
        # Create dataset for training (uses labeled train split)
        train_dataset = STL10ClipDynamicGPU(
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
    
    Note:
        STL-10 uses split names ('train', 'test', 'unlabeled', 'train+unlabeled')
        instead of train=True/False. This class maps train=True to 'train' split
        (labeled data only) and train=False to 'test' split.
    """
    
    # Class names for reference (same as folder names in STL-10)
    CLASS_NAMES = [
        'airplane', 'bird', 'car', 'cat', 'deer',
        'dog', 'horse', 'monkey', 'ship', 'truck'
    ]
    
    @property
    def dataset_name(self) -> str:
        return "stl10"
    
    def _get_torchvision_dataset(self, train: bool):
        """
        Return STL-10 dataset.
        
        Note: STL-10 uses split names ('train', 'test', 'unlabeled', 'train+unlabeled')
        instead of the standard train=True/False convention.
        """
        split = 'train' if train else 'test'
        return datasets.STL10(
            root=str(self.root),
            split=split,
            download=True,
            transform=transforms.ToTensor()
        )


@register_torchvision_clip_dataset('stl10_unlabeled_clip')
class STL10UnlabeledClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic STL-10 CLIP dataset with unlabeled data.
    
    This variant uses the unlabeled split of STL-10 which contains 100,000
    images. Useful for large-scale unsupervised DR experiments.
    
    Note: Labels will be dummy values (all zeros) since the data is unlabeled.
    
    Example:
        ```python
        # Create dataset with unlabeled data
        train_dataset = STL10UnlabeledClipDynamicGPU(
            root='data',
            train=True,  # train=True uses 'unlabeled', train=False uses 'test'
            subset_sizes=[1000, 5000, 10000],
            knn_k=15,
        )
        ```
    """
    
    @property
    def dataset_name(self) -> str:
        return "stl10_unlabeled"
    
    def _get_torchvision_dataset(self, train: bool):
        """
        Return STL-10 unlabeled dataset for train, or test for test split.
        """
        split = 'unlabeled' if train else 'test'
        return datasets.STL10(
            root=str(self.root),
            split=split,
            download=True,
            transform=transforms.ToTensor()
        )


__all__ = [
    'STL10ClipDynamicGPU',
    'STL10UnlabeledClipDynamicGPU',
]
