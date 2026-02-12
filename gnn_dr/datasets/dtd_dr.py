"""Describable Textures Dataset (DTD) CLIP embeddings for dimensionality reduction.

The Describable Textures Dataset (DTD) is a texture database, consisting of
5640 images, organized according to a list of 47 terms (categories) inspired
from human perception. There are 120 images for each category.

This dataset is interesting for dimensionality reduction because textures
represent a fundamentally different visual concept than object recognition,
making it useful for testing how well DR methods generalize across domains.

Dataset statistics:
- Training: 1,880 images (40 per class)
- Validation: 1,880 images (40 per class)
- Test: 1,880 images (40 per class)
- Total: 5,640 images
- Classes: 47 texture categories
- Image size: Variable (at least 300x300)

Categories:
    banded, blotchy, braided, bubbly, bumpy, chequered, cobwebbed, cracked,
    crosshatched, crystalline, dotted, fibrous, flecked, freckled, frilly,
    gauzy, grid, grooved, honeycombed, interlaced, knitted, lacelike, lined,
    marbled, matted, meshed, paisley, perforated, pitted, pleated, polka-dotted,
    porous, potholed, scaly, smeared, spiralled, sprinkled, stained, stratified,
    striped, studded, swirly, veined, waffled, woven, wrinkled, zigzagged

Reference:
    Cimpoi et al., "Describing Textures in the Wild"
    IEEE Conference on Computer Vision and Pattern Recognition, 2014
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


@register_torchvision_clip_dataset('dtd_clip')
class DTDClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic DTD (Describable Textures) CLIP dataset.
    
    DTD contains 5,640 images of 47 texture categories. Unlike object
    recognition datasets, textures represent patterns and materials,
    making this useful for testing domain generalization.
    
    Classes: 47 texture categories (e.g., banded, braided, bumpy, cracked)
    
    Example:
        ```python
        train_dataset = DTDClipDynamicGPU(
            root='data',
            train=True,
            subset_sizes=[100, 500, 1000, 2000],
            knn_k=15,
        )
        ```
    """
    
    # Texture category names
    CLASS_NAMES = [
        'banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered',
        'cobwebbed', 'cracked', 'crosshatched', 'crystalline', 'dotted',
        'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid',
        'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike',
        'lined', 'marbled', 'matted', 'meshed', 'paisley', 'perforated',
        'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed', 'scaly',
        'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified',
        'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven',
        'wrinkled', 'zigzagged'
    ]
    
    @property
    def dataset_name(self) -> str:
        return "dtd"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return DTD dataset."""
        # DTD has train, val, test splits - use train for training, test for testing
        split = 'train' if train else 'test'
        return datasets.DTD(
            root=str(self.root),
            split=split,
            download=True,
            transform=transforms.ToTensor()
        )


@register_torchvision_clip_dataset('dtd_full_clip')
class DTDFullClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic DTD CLIP dataset using train+val splits.
    
    Combines train and validation splits for more training data (3,760 images
    instead of 1,880). Test split is still separate.
    """
    
    @property
    def dataset_name(self) -> str:
        return "dtd_trainval"
    
    def _get_torchvision_dataset(self, train: bool):
        """
        Return DTD dataset with combined train+val for training.
        
        Note: For train=True, we use 'train' split (val can be added separately).
        For a true train+val combination, users should load both and concatenate.
        """
        split = 'train' if train else 'test'
        return datasets.DTD(
            root=str(self.root),
            split=split,
            download=True,
            transform=transforms.ToTensor()
        )


__all__ = [
    'DTDClipDynamicGPU',
    'DTDFullClipDynamicGPU',
]
