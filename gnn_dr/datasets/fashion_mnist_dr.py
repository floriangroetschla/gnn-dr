"""Fashion-MNIST CLIP embeddings dataset for dimensionality reduction.

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set
of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28
grayscale image, associated with a label from 10 classes.

Classes:
    0: T-shirt/top
    1: Trouser
    2: Pullover
    3: Dress
    4: Coat
    5: Sandal
    6: Shirt
    7: Sneaker
    8: Bag
    9: Ankle boot

Fashion-MNIST is intended to serve as a direct drop-in replacement for the original
MNIST dataset for benchmarking machine learning algorithms. It shares the same image
size and structure of training and testing splits.
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


@register_torchvision_clip_dataset('fashion_mnist_clip')
class FashionMNISTClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic Fashion-MNIST CLIP dataset.
    
    Fashion-MNIST contains 70,000 28x28 grayscale images in 10 clothing classes.
    There are 60,000 training images and 10,000 test images.
    
    Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
    
    Generates random subset graphs on-the-fly, keeping all tensors on GPU for maximum
    performance. CLIP embeddings are extracted once and cached to disk.
    
    Example:
        ```python
        # Create dataset for training
        train_dataset = FashionMNISTClipDynamicGPU(
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
    
    # Class names for reference
    CLASS_NAMES = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]
    
    @property
    def dataset_name(self) -> str:
        return "fashion_mnist"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return Fashion-MNIST dataset."""
        return datasets.FashionMNIST(
            root=str(self.root),
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )


__all__ = [
    'FashionMNISTClipDynamicGPU',
]
