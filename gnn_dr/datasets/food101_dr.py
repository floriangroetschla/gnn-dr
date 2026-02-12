"""Food-101 CLIP embeddings dataset for dimensionality reduction.

The Food-101 dataset consists of 101 food categories, with 101,000 images.
For each class, 250 manually reviewed test images are provided as well as
750 training images. The training images were not cleaned, and thus still
contain some amount of noise (incorrect labels, images with text, etc.).

Dataset statistics:
- Training: 75,750 images (750 per class)
- Test: 25,250 images (250 per class)
- Total: 101,000 images
- Classes: 101 food categories
- Image size: Variable (rescaled to max 512 pixels)

Example categories:
    apple_pie, baby_back_ribs, baklava, beef_carpaccio, beef_tartare,
    beet_salad, beignets, bibimbap, bread_pudding, breakfast_burrito,
    bruschetta, caesar_salad, cannoli, caprese_salad, carrot_cake,
    ceviche, cheesecake, cheese_plate, chicken_curry, chicken_quesadilla,
    chicken_wings, chocolate_cake, chocolate_mousse, churros, clam_chowder,
    club_sandwich, crab_cakes, creme_brulee, croque_madame, cup_cakes,
    deviled_eggs, donuts, dumplings, edamame, eggs_benedict, escargots,
    falafel, filet_mignon, fish_and_chips, foie_gras, french_fries,
    french_onion_soup, french_toast, fried_calamari, fried_rice,
    frozen_yogurt, garlic_bread, gnocchi, greek_salad, grilled_cheese_sandwich,
    grilled_salmon, guacamole, gyoza, hamburger, hot_and_sour_soup, hot_dog,
    huevos_rancheros, hummus, ice_cream, lasagna, lobster_bisque, 
    lobster_roll_sandwich, macaroni_and_cheese, macarons, miso_soup, mussels,
    nachos, omelette, onion_rings, oysters, pad_thai, paella, pancakes,
    panna_cotta, peking_duck, pho, pizza, pork_chop, poutine, prime_rib,
    pulled_pork_sandwich, ramen, ravioli, red_velvet_cake, risotto, samosa,
    sashimi, scallops, seaweed_salad, shrimp_and_grits, spaghetti_bolognese,
    spaghetti_carbonara, spring_rolls, steak, strawberry_shortcake, sushi,
    tacos, takoyaki, tiramisu, tuna_tartare, waffles

Reference:
    Bossard, Guillaumin, Van Gool, "Food-101 – Mining Discriminative Components 
    with Random Forests", ECCV 2014
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


@register_torchvision_clip_dataset('food101_clip')
class Food101ClipDynamicGPU(TorchvisionCLIPDatasetGPU):
    """
    GPU-optimized dynamic Food-101 CLIP dataset.
    
    Food-101 contains 101K images of 101 food categories. This is a 
    large-scale food recognition benchmark with challenging variations
    in presentation and lighting.
    
    Classes: 101 food categories
    
    Example:
        ```python
        train_dataset = Food101ClipDynamicGPU(
            root='data',
            train=True,
            subset_sizes=[100, 500, 1000, 5000, 10000],
            knn_k=15,
        )
        ```
    """
    
    @property
    def dataset_name(self) -> str:
        return "food101"
    
    def _get_torchvision_dataset(self, train: bool):
        """Return Food-101 dataset."""
        split = 'train' if train else 'test'
        return datasets.Food101(
            root=str(self.root),
            split=split,
            download=True,
            transform=transforms.ToTensor()
        )


__all__ = [
    'Food101ClipDynamicGPU',
]
