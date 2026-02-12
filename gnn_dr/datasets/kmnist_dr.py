"""KMNIST CLIP embeddings dataset for dimensionality reduction.

Kuzushiji-MNIST (KMNIST) is a drop-in replacement for MNIST that uses
cursive Japanese (Kuzushiji) characters. It represents a different
visual domain from Western handwritten digits.

Dataset statistics:
- Training set: 60,000 images
- Test set: 10,000 images
- Image size: 28x28 grayscale
- Classes: 10 (10 Hiragana characters)

The 10 classes represent the 10 rows of the Hiragana alphabet:
    お (o), き (ki), す (su), つ (tsu), な (na),
    は (ha), ま (ma), や (ya), れ (re), を (wo)

Reference:
    Clanuwat et al., "Deep Learning for Classical Japanese Literature"
    arXiv:1812.01718, 2018
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


@register_torchvision_clip_dataset('kmnist_clip')
class KMNISTClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic KMNIST CLIP dataset.
    
    Kuzushiji-MNIST contains 70,000 28x28 grayscale images of cursive
    Japanese (Hiragana) characters in 10 classes. It serves as a 
    drop-in replacement for MNIST with a different visual domain.
    
    Classes: 10 Hiragana characters (お, き, す, つ, な, は, ま, や, れ, を)
    
    Example:
        ```python
        train_dataset = KMNISTClipDynamicGPU(
            root='data',
            train=True,
            subset_sizes=[100, 500, 1000],
            knn_k=15,
        )
        ```
    """
    
    # Class names (Hiragana characters)
    CLASS_NAMES = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']
    
    @property
    def dataset_name(self) -> str:
        return "kmnist"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return KMNIST dataset."""
        return datasets.KMNIST(
            root=str(self.root),
            train=train,
            download=True,
            transform=transforms.ToTensor()
        )


__all__ = [
    'KMNISTClipDynamicGPU',
]
