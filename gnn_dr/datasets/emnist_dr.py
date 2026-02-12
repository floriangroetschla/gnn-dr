"""EMNIST CLIP embeddings dataset for dimensionality reduction.

EMNIST (Extended MNIST) extends the original MNIST dataset to include
handwritten letters as well as digits. It contains several different
splits depending on the use case.

Dataset splits and statistics:
- ByClass: 814,255 images, 62 classes (digits 0-9 + letters A-Z + a-z)
- ByMerge: 814,255 images, 47 classes (merged similar letters)
- Balanced: 131,600 images, 47 classes (balanced version)
- Letters: 145,600 images, 26 classes (only letters)
- Digits: 280,000 images, 10 classes (same as MNIST but more data)
- MNIST: 70,000 images, 10 classes (identical to original MNIST)

Image size: 28x28 grayscale

Reference:
    Cohen et al., "EMNIST: Extending MNIST to handwritten letters"
    arXiv:1702.05373, 2017
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


@register_torchvision_clip_dataset('emnist_clip')
class EMNISTClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic EMNIST CLIP dataset (ByClass split).
    
    Uses the ByClass split with 62 classes (digits + uppercase + lowercase letters).
    This is the largest and most complete split with 814K images.
    
    Example:
        ```python
        train_dataset = EMNISTClipDynamicGPU(
            root='data',
            train=True,
            subset_sizes=[100, 500, 1000, 5000],
            knn_k=15,
        )
        ```
    """
    
    @property
    def dataset_name(self) -> str:
        return "emnist_byclass"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return EMNIST ByClass dataset."""
        return datasets.EMNIST(
            root=str(self.root),
            split='byclass',
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )


@register_torchvision_clip_dataset('emnist_balanced_clip')
class EMNISTBalancedClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic EMNIST CLIP dataset (Balanced split).
    
    Uses the Balanced split with 47 classes and 131K images.
    Similar letters are merged and class sizes are balanced.
    """
    
    @property
    def dataset_name(self) -> str:
        return "emnist_balanced"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return EMNIST Balanced dataset."""
        return datasets.EMNIST(
            root=str(self.root),
            split='balanced',
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )


@register_torchvision_clip_dataset('emnist_letters_clip')
class EMNISTLettersClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic EMNIST CLIP dataset (Letters split).
    
    Uses the Letters split with 26 classes (A-Z only) and 145K images.
    """
    
    @property
    def dataset_name(self) -> str:
        return "emnist_letters"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return EMNIST Letters dataset."""
        return datasets.EMNIST(
            root=str(self.root),
            split='letters',
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )


__all__ = [
    'EMNISTClipDynamicGPU',
    'EMNISTBalancedClipDynamicGPU',
    'EMNISTLettersClipDynamicGPU',
]
