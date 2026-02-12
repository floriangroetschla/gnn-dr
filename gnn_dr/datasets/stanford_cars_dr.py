"""Stanford Cars CLIP embeddings dataset for dimensionality reduction.

The Stanford Cars dataset contains images of 196 classes of cars. The data
is split into 8,144 training images and 8,041 testing images, where each
class has been split roughly in a 50-50 split.

Dataset statistics:
- Training: 8,144 images
- Test: 8,041 images
- Classes: 196 (Make, Model, Year combinations)
- Image size: Variable

Classes represent combinations like:
- "AM General Hummer SUV 2000"
- "Audi A5 Coupe 2012"
- "BMW M3 Coupe 2012"

Reference:
    Krause et al., "3D Object Representations for Fine-Grained Categorization"
    4th IEEE Workshop on 3D Representation and Recognition, 2013
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


@register_torchvision_clip_dataset('stanford_cars_clip')
class StanfordCarsClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic Stanford Cars CLIP dataset.
    
    Stanford Cars contains ~16K images of 196 car classes (Make, Model, Year).
    This is a challenging fine-grained classification benchmark with subtle
    differences between classes.
    
    Classes: 196 (e.g., "BMW M3 Coupe 2012", "Audi A5 Coupe 2012")
    
    Example:
        ```python
        train_dataset = StanfordCarsClipDynamicGPU(
            root='data',
            train=True,
            subset_sizes=[100, 500, 1000, 5000],
            knn_k=15,
        )
        ```
    """
    
    @property
    def dataset_name(self) -> str:
        return "stanford_cars"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return Stanford Cars dataset."""
        split = 'train' if train else 'test'
        return datasets.StanfordCars(
            root=str(self.root),
            split=split,
            download=True,
            transform=transforms.ToTensor()
        )


__all__ = [
    'StanfordCarsClipDynamicGPU',
]
