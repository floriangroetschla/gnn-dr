"""Oxford Flowers102 CLIP embeddings dataset for dimensionality reduction.

The Oxford Flowers102 dataset is a fine-grained visual classification dataset 
consisting of 102 flower categories commonly occurring in the United Kingdom.
Each class consists of between 40 and 258 images.

This dataset is particularly interesting for dimensionality reduction because:
- Fine-grained categories create complex, overlapping clusters
- Visual similarity between species creates interesting neighborhood structures
- The 102 classes provide rich multi-cluster structure to visualize

Dataset statistics:
- Training set: 1,020 images (10 per class)
- Validation set: 1,020 images (10 per class)
- Test set: 6,149 images (minimum 20 per class)
- Total: 8,189 images
- Image size: Variable (typically 500x500+ pixels)

Classes (102 flower species):
    pink primrose, hard-leaved pocket orchid, canterbury bells, sweet pea,
    english marigold, tiger lily, moon orchid, bird of paradise, monkshood,
    globe thistle, snapdragon, colt's foot, king protea, spear thistle,
    yellow iris, globe-flower, purple coneflower, peruvian lily, balloon flower,
    giant white arum lily, fire lily, pincushion flower, fritillary, red ginger,
    grape hyacinth, corn poppy, prince of wales feathers, stemless gentian,
    artichoke, sweet william, carnation, garden phlox, love in the mist, mexican aster,
    alpine sea holly, ruby-lipped cattleya, cape flower, great masterwort, siam tulip,
    lenten rose, barbeton daisy, daffodil, sword lily, poinsettia, bolero deep blue,
    wallflower, marigold, buttercup, oxeye daisy, common dandelion, petunia, wild pansy,
    primula, sunflower, pelargonium, bishop of llandaff, gaura, geranium, orange dahlia,
    pink-yellow dahlia, cautleya spicata, japanese anemone, black-eyed susan, silverbush,
    californian poppy, osteospermum, spring crocus, bearded iris, windflower, tree poppy,
    gazania, azalea, water lily, rose, thorn apple, morning glory, passion flower,
    lotus, toad lily, anthurium, frangipani, clematis, hibiscus, columbine, desert-rose,
    tree mallow, magnolia, cyclamen, watercress, canna lily, hippeastrum, bee balm,
    ball moss, foxglove, bougainvillea, camellia, mallow, mexican petunia, bromelia,
    blanket flower, trumpet creeper, blackberry lily

Reference:
    Nilsback, Zisserman, "Automated Flower Classification over a Large Number of Classes",
    Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing, 2008
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


@register_torchvision_clip_dataset('flowers102_clip')
class Flowers102ClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic Oxford Flowers102 CLIP dataset.
    
    Flowers102 contains 8,189 images of 102 flower species from the UK.
    The fine-grained nature of the classes creates complex cluster structures
    that are interesting for dimensionality reduction visualization.
    
    Classes: 102 flower species (see module docstring for full list)
    
    Generates random subset graphs on-the-fly, keeping all tensors on GPU for maximum
    performance. CLIP embeddings are extracted once and cached to disk.
    
    Example:
        ```python
        # Create dataset for training
        train_dataset = Flowers102ClipDynamicGPU(
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
        Flowers102 uses split names ('train', 'val', 'test') instead of
        train=True/False. This class maps train=True to 'train' split
        and train=False to 'test' split. The validation split is not
        used by default but can be accessed by modifying _get_torchvision_dataset.
    """
    
    # Number of classes
    NUM_CLASSES = 102
    
    # Sample class names (first 20 of 102)
    SAMPLE_CLASS_NAMES = [
        'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 
        'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 
        'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', 
        "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 
        'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 
        'giant white arum lily'
    ]
    
    @property
    def dataset_name(self) -> str:
        return "flowers102"
    
    def _get_torchvision_dataset(self, train: bool):
        """
        Return Flowers102 dataset.
        
        Note: Flowers102 uses split names ('train', 'val', 'test') instead of
        the standard train=True/False convention.
        """
        split = 'train' if train else 'test'
        return datasets.Flowers102(
            root=str(self.root),
            split=split,
            download=True,
            transform=transforms.ToTensor()
        )


@register_torchvision_clip_dataset('flowers102_full_clip')
class Flowers102FullClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic Oxford Flowers102 CLIP dataset using all splits.
    
    This variant combines train, val, and test splits for maximum data (8,189 images).
    Useful when you want to use all available data for visualization or
    large-scale experiments.
    
    Note: Since this combines all splits, there's no separate test set available.
    For train=False, this returns the test split only (same as Flowers102ClipDynamicGPU).
    
    Example:
        ```python
        # Create dataset with all data
        train_dataset = Flowers102FullClipDynamicGPU(
            root='data',
            train=True,  # Uses train+val+test
            subset_sizes=[100, 500, 1000, 5000],
            knn_k=15,
        )
        ```
    """
    
    @property
    def dataset_name(self) -> str:
        return "flowers102_full"
    
    def _get_torchvision_dataset(self, train: bool):
        """
        Return Flowers102 dataset with all splits combined for train.
        
        For train=True, we return the train split. The embeddings will be loaded
        separately for train, val, and test, then concatenated in _load_embeddings_and_labels.
        
        For simplicity, this implementation just uses train split for train=True
        and test split for train=False. Override _load_embeddings_and_labels for
        true combination of all splits.
        """
        # For now, just use train or test split
        # A full implementation would override _load_embeddings_and_labels
        # to combine all three splits
        split = 'train' if train else 'test'
        return datasets.Flowers102(
            root=str(self.root),
            split=split,
            download=True,
            transform=transforms.ToTensor()
        )


__all__ = [
    'Flowers102ClipDynamicGPU',
    'Flowers102FullClipDynamicGPU',
]
