"""Shared CLIP embedding extraction and caching utilities.

This module provides reusable functions for extracting CLIP embeddings from images
and caching them to disk for efficient reuse across different datasets.
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Optional, Union, Callable
from PIL import Image

try:
    import clip
except ImportError:
    clip = None


class CLIPEmbeddingExtractor:
    """Extracts and caches CLIP embeddings from images.
    
    Supports multiple CLIP model variants and efficient batch processing.
    """
    
    # Mapping from common model names to CLIP model names
    MODEL_MAPPING = {
        'openai/CLIP-vit-base-patch32': 'ViT-B/32',
        'openai/CLIP-vit-base-patch16': 'ViT-B/16',
        'openai/CLIP-vit-large-patch14': 'ViT-L/14',
        'openai/CLIP-vit-large-patch14@336px': 'ViT-L/14@336px',
        'ViT-B/32': 'ViT-B/32',
        'ViT-B/16': 'ViT-B/16',
        'ViT-L/14': 'ViT-L/14',
        'ViT-L/14@336px': 'ViT-L/14@336px',
        'RN50': 'RN50',
        'RN101': 'RN101',
        'RN50x4': 'RN50x4',
        'RN50x16': 'RN50x16',
        'RN50x64': 'RN50x64',
    }
    
    def __init__(
        self, 
        model_name: str = 'openai/CLIP-vit-base-patch32', 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize CLIP embedding extractor.
        
        Args:
            model_name: CLIP model identifier (supports multiple naming conventions)
            device: Device to run model on ('cuda', 'cpu', 'cuda:0', etc.)
        """
        if clip is None:
            raise ImportError("Please install clip: pip install openai-clip")
        
        self.device = device
        self.model_name = model_name
        
        # Get the correct model name
        if model_name in self.MODEL_MAPPING:
            clip_model_name = self.MODEL_MAPPING[model_name]
        else:
            # Try to extract from path and map
            model_short = model_name.split('/')[-1]
            clip_model_name = self.MODEL_MAPPING.get(model_short, model_short)
        
        print(f"Loading CLIP model: {clip_model_name}")
        self.model, self.preprocess = clip.load(clip_model_name, device=device)
        self.model.eval()
    
    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the loaded model."""
        # ViT-B models have 512-dim embeddings, ViT-L have 768-dim
        model_name = self.model_name.lower()
        if 'vit-l' in model_name or 'large' in model_name:
            return 768
        return 512
    
    def extract_embeddings(
        self, 
        images: List[Union[Image.Image, torch.Tensor]], 
        batch_size: int = 256,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Extract embeddings from images.
        
        Args:
            images: List of PIL images or tensor images
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Embeddings tensor of shape (N, embedding_dim), normalized to unit length
        """
        embeddings = []
        
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting CLIP embeddings")
        
        with torch.no_grad():
            for i in iterator:
                batch = images[i:i+batch_size]
                
                # Preprocess and move to device
                if isinstance(batch[0], torch.Tensor):
                    # Already tensor - just stack and move to device
                    batch_processed = torch.stack(batch).to(self.device)
                else:
                    # PIL image - apply CLIP preprocessing
                    batch_processed = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
                
                # Extract embeddings
                batch_embeddings = self.model.encode_image(batch_processed)
                # Normalize to unit length (L2 normalization)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                
                embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(embeddings, dim=0)


def get_embeddings_cache_path(
    cache_dir: Path,
    dataset_name: str,
    split: str,
    clip_model: str = 'openai/CLIP-vit-base-patch32'
) -> Path:
    """
    Get the standardized cache path for embeddings.
    
    Args:
        cache_dir: Base cache directory
        dataset_name: Name of the dataset (e.g., 'mnist', 'cifar10')
        split: Dataset split ('train' or 'test')
        clip_model: CLIP model name (used for cache key)
        
    Returns:
        Path to the cached embeddings file
    """
    # Create a safe model name for the filename
    model_suffix = clip_model.replace('/', '_').replace('@', '_')
    return cache_dir / f'{dataset_name}_clip_embeddings_{split}_{model_suffix}.pt'


def extract_and_cache_clip_embeddings(
    images: List[Union[Image.Image, torch.Tensor]],
    cache_path: Path,
    clip_model: str = 'openai/CLIP-vit-base-patch32',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 256,
    force_recompute: bool = False
) -> torch.Tensor:
    """
    Extract CLIP embeddings from images and cache them to disk.
    
    If cached embeddings exist, loads from disk. Otherwise, extracts embeddings
    using CLIP and saves them for future use.
    
    Args:
        images: List of PIL images or tensor images
        cache_path: Path to save/load cached embeddings
        clip_model: CLIP model identifier
        device: Device to run CLIP model on
        batch_size: Batch size for CLIP inference
        force_recompute: If True, recompute even if cache exists
        
    Returns:
        Embeddings tensor of shape (N, embedding_dim), normalized to unit length
    """
    from gnn_dr.datasets.transforms import normalize_embeddings
    
    cache_path = Path(cache_path)
    
    # Check cache
    if cache_path.exists() and not force_recompute:
        print(f"Loading cached CLIP embeddings from {cache_path}")
        embeddings = torch.load(cache_path, weights_only=True)
        # Ensure embeddings are normalized
        embeddings = normalize_embeddings(embeddings)
        return embeddings
    
    # Extract embeddings
    print(f"Extracting CLIP embeddings ({len(images)} images)...")
    extractor = CLIPEmbeddingExtractor(model_name=clip_model, device=device)
    embeddings = extractor.extract_embeddings(images, batch_size=batch_size)
    
    # Cache embeddings
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, cache_path)
    print(f"Cached CLIP embeddings to {cache_path}")
    
    return embeddings


def get_images_from_torchvision_dataset(
    dataset,
    to_pil: bool = True
) -> List[Union[Image.Image, torch.Tensor]]:
    """
    Extract images from a torchvision dataset.
    
    Args:
        dataset: A torchvision dataset with (image, label) items
        to_pil: If True, convert tensor images to PIL Images
        
    Returns:
        List of images (PIL or Tensor depending on to_pil)
    """
    import torchvision.transforms as transforms
    
    images = []
    to_pil_transform = transforms.ToPILImage()
    
    for img, _ in tqdm(dataset, desc="Loading images"):
        if to_pil:
            if isinstance(img, torch.Tensor):
                img = to_pil_transform(img)
            # If already PIL, keep as is
        images.append(img)
    
    return images


def get_labels_from_torchvision_dataset(dataset) -> torch.Tensor:
    """
    Extract labels from a torchvision dataset.
    
    Args:
        dataset: A torchvision dataset with (image, label) items
        
    Returns:
        Tensor of labels with shape (N,)
    """
    labels = [label for _, label in dataset]
    return torch.tensor(labels, dtype=torch.long)
