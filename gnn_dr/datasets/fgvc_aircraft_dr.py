"""FGVC Aircraft CLIP embeddings dataset for dimensionality reduction.

The FGVC Aircraft dataset contains images of aircraft from different 
manufacturers and models. It is a fine-grained visual classification
benchmark with a hierarchical class structure.

Dataset statistics:
- Total: 10,200 images
- Training: 3,334 images
- Validation: 3,333 images  
- Test: 3,333 images
- Classes: 100 aircraft model variants
- Hierarchy: 70 families, 30 manufacturers

The dataset has three levels of hierarchy:
- Variant: e.g., "Boeing 737-200"
- Family: e.g., "Boeing 737"
- Manufacturer: e.g., "Boeing"

Reference:
    Maji et al., "Fine-Grained Visual Classification of Aircraft"
    arXiv:1306.5151, 2013
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


@register_torchvision_clip_dataset('fgvc_aircraft_clip')
class FGVCAircraftClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic FGVC Aircraft CLIP dataset.
    
    FGVC Aircraft contains 10K images of 100 aircraft model variants.
    This is a challenging fine-grained classification benchmark where
    classes differ in subtle visual details.
    
    Classes: 100 aircraft variants (e.g., "Boeing 737-200", "Airbus A320")
    
    Example:
        ```python
        train_dataset = FGVCAircraftClipDynamicGPU(
            root='data',
            train=True,
            subset_sizes=[100, 500, 1000, 3000],
            knn_k=15,
        )
        ```
    """
    
    @property
    def dataset_name(self) -> str:
        return "fgvc_aircraft"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return FGVC Aircraft dataset."""
        split = 'train' if train else 'test'
        return datasets.FGVCAircraft(
            root=str(self.root),
            split=split,
            annotation_level='variant',
            download=True,
            transform=transforms.ToTensor()
        )


@register_torchvision_clip_dataset('fgvc_aircraft_family_clip')
class FGVCAircraftFamilyClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic FGVC Aircraft CLIP dataset (Family level).
    
    Uses the family-level annotations (70 classes) instead of variant-level.
    This is a coarser classification than the default variant level.
    """
    
    @property
    def dataset_name(self) -> str:
        return "fgvc_aircraft_family"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return FGVC Aircraft dataset with family-level annotations."""
        split = 'train' if train else 'test'
        return datasets.FGVCAircraft(
            root=str(self.root),
            split=split,
            annotation_level='family',
            download=True,
            transform=transforms.ToTensor()
        )


__all__ = [
    'FGVCAircraftClipDynamicGPU',
    'FGVCAircraftFamilyClipDynamicGPU',
]
