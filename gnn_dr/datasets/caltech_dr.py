"""Caltech101 and Caltech256 CLIP embeddings datasets for dimensionality reduction.

Caltech101 and Caltech256 are classic object recognition datasets from Caltech.
They contain images of objects belonging to various categories.

Caltech101 statistics:
- Total: ~9,146 images
- Classes: 101 object categories + 1 background category
- Images per class: 40-800
- Image size: Variable (~300x200 typical)

Caltech256 statistics:
- Total: ~30,607 images
- Classes: 256 object categories + 1 clutter category
- Images per class: 80-827
- Image size: Variable

Reference:
    Fei-Fei, Fergus, Perona, "Learning generative visual models from few training
    examples: An incremental Bayesian approach tested on 101 object categories"
    CVPR Workshop, 2004
    
    Griffin, Holub, Perona, "Caltech-256 Object Category Dataset"
    Caltech Technical Report, 2007
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


@register_torchvision_clip_dataset('caltech101_clip')
class Caltech101ClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic Caltech101 CLIP dataset.
    
    Caltech101 contains ~9K images of 101 object categories plus a 
    background/clutter category. This is a classic benchmark for
    object recognition.
    
    Classes: 102 (101 objects + BACKGROUND_Google)
    
    Example:
        ```python
        train_dataset = Caltech101ClipDynamicGPU(
            root='data',
            train=True,
            subset_sizes=[100, 500, 1000, 5000],
            knn_k=15,
        )
        ```
    
    Note:
        Caltech101 doesn't have official train/test splits. This implementation
        uses the full dataset for both train=True and train=False. For proper
        evaluation, use random splits via subset sampling.
    """
    
    @property
    def dataset_name(self) -> str:
        return "caltech101"
    
    def _get_torchvision_dataset(self, train: bool):
        """
        Return Caltech101 dataset.
        
        Note: Caltech101 doesn't have official train/test splits, so we
        return the full dataset for both cases. Users should create their
        own splits if needed.
        """
        return datasets.Caltech101(
            root=str(self.root),
            target_type='category',
            download=True,
            transform=transforms.ToTensor()
        )


@register_torchvision_clip_dataset('caltech256_clip')
class Caltech256ClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic Caltech256 CLIP dataset.
    
    Caltech256 contains ~30K images of 256 object categories plus a
    clutter category. It's an extension of Caltech101 with more categories
    and more challenging images.
    
    Classes: 257 (256 objects + clutter)
    
    Example:
        ```python
        train_dataset = Caltech256ClipDynamicGPU(
            root='data',
            train=True,
            subset_sizes=[100, 500, 1000, 5000, 10000],
            knn_k=15,
        )
        ```
    
    Note:
        Caltech256 doesn't have official train/test splits. This implementation
        uses the full dataset for both train=True and train=False.
    """
    
    @property
    def dataset_name(self) -> str:
        return "caltech256"
    
    def _get_torchvision_dataset(self, train: bool):
        """
        Return Caltech256 dataset.
        
        Note: Caltech256 doesn't have official train/test splits.
        """
        return datasets.Caltech256(
            root=str(self.root),
            download=True,
            transform=transforms.ToTensor()
        )


__all__ = [
    'Caltech101ClipDynamicGPU',
    'Caltech256ClipDynamicGPU',
]
