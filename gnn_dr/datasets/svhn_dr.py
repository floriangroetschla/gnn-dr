"""SVHN CLIP embeddings dataset for dimensionality reduction.

The Street View House Numbers (SVHN) Dataset is obtained from house numbers in 
Google Street View images. It contains over 600,000 digit images in total.

SVHN is a real-world dataset that's more challenging than MNIST because:
- Images come from natural scenes with varying lighting and backgrounds
- Numbers can have varying fonts, colors, and orientations
- Adjacent digits may be visible in the image

Dataset statistics:
- Training set: 73,257 images
- Test set: 26,032 images
- Extra set: 531,131 additional training images (not used by default)
- Image size: 32x32 RGB
- Classes: 10 (digits 0-9, where 0 is labeled as 10 in original dataset)

Reference:
    Netzer et al., "Reading Digits in Natural Images with Unsupervised Feature Learning"
    NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011
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


@register_torchvision_clip_dataset('svhn_clip')
class SVHNClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic SVHN CLIP dataset.
    
    SVHN (Street View House Numbers) contains 99,289 32x32 RGB images of house
    numbers captured from Google Street View. Unlike MNIST, these are real-world
    images with varying backgrounds, lighting conditions, and fonts.
    
    Classes: Digits 0-9
    
    Generates random subset graphs on-the-fly, keeping all tensors on GPU for maximum
    performance. CLIP embeddings are extracted once and cached to disk.
    
    Example:
        ```python
        # Create dataset for training
        train_dataset = SVHNClipDynamicGPU(
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
        SVHN uses 'train'/'test'/'extra' splits instead of train=True/False.
        This class maps train=True to 'train' split and train=False to 'test' split.
        The 'extra' split (500K+ additional images) is not used by default.
    """
    
    # Class names for reference
    CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    @property
    def dataset_name(self) -> str:
        return "svhn"
    
    def _get_torchvision_dataset(self, train: bool):
        """
        Return SVHN dataset.
        
        Note: SVHN uses split names ('train', 'test', 'extra') instead of
        the standard train=True/False convention.
        """
        split = 'train' if train else 'test'
        return datasets.SVHN(
            root=str(self.root),
            split=split,
            download=True,
            transform=transforms.ToTensor()
        )


__all__ = [
    'SVHNClipDynamicGPU',
]
