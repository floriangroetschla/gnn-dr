"""Oxford-IIIT Pet CLIP embeddings dataset for dimensionality reduction.

The Oxford-IIIT Pet Dataset is a 37 category pet dataset with roughly 200
images for each class. The images have large variations in scale, pose and
lighting. All images have an associated ground truth annotation of breed,
head ROI, and pixel level trimap segmentation.

Dataset statistics:
- Training: 3,680 images
- Test: 3,669 images
- Classes: 37 pet breeds (25 dogs, 12 cats)
- Image size: Variable

Classes include cat and dog breeds like:
- Cats: Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau,
        Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx
- Dogs: American Bulldog, American Pit Bull Terrier, Basset Hound, Beagle,
        Boxer, Chihuahua, English Cocker Spaniel, English Setter, German 
        Shorthaired, Great Pyrenees, Havanese, Japanese Chin, Keeshond,
        Leonberger, Miniature Pinscher, Newfoundland, Pomeranian, Pug,
        Saint Bernard, Samoyed, Scottish Terrier, Shiba Inu, Staffordshire
        Bull Terrier, Wheaten Terrier, Yorkshire Terrier

Reference:
    Parkhi et al., "Cats and Dogs"
    IEEE Conference on Computer Vision and Pattern Recognition, 2012
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


@register_torchvision_clip_dataset('oxford_pets_clip')
class OxfordIIITPetClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic Oxford-IIIT Pet CLIP dataset.
    
    Oxford-IIIT Pet contains ~7K images of 37 pet breeds (25 dogs, 12 cats).
    This is a fine-grained animal recognition benchmark.
    
    Classes: 37 pet breeds
    
    Example:
        ```python
        train_dataset = OxfordIIITPetClipDynamicGPU(
            root='data',
            train=True,
            subset_sizes=[100, 500, 1000, 3000],
            knn_k=15,
        )
        ```
    """
    
    # Class names (alphabetical order)
    CLASS_NAMES = [
        'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 
        'Basset Hound', 'Beagle', 'Bengal', 'Birman', 'Bombay', 
        'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau',
        'English Cocker Spaniel', 'English Setter', 'German Shorthaired',
        'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond',
        'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland',
        'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian Blue',
        'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu',
        'Siamese', 'Sphynx', 'Staffordshire Bull Terrier', 
        'Wheaten Terrier', 'Yorkshire Terrier'
    ]
    
    @property
    def dataset_name(self) -> str:
        return "oxford_pets"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return Oxford-IIIT Pet dataset."""
        split = 'trainval' if train else 'test'
        return datasets.OxfordIIITPet(
            root=str(self.root),
            split=split,
            target_types='category',
            download=True,
            transform=transforms.ToTensor()
        )


__all__ = [
    'OxfordIIITPetClipDynamicGPU',
]
