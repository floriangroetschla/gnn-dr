"""LAION-400M CLIP embeddings dataset for dimensionality reduction tasks.

LAION-400M contains ~400 million image-text pairs with pre-computed CLIP embeddings.
This module downloads embedding chunks and uses them for DR training.

Unlike torchvision-based datasets, LAION embeddings are pre-computed and downloaded
directly, so this dataset has a different architecture but still inherits from
CLIPDRDatasetGPUBase for graph construction.
"""

import torch
import numpy as np
import urllib.request
from pathlib import Path
from typing import Optional, List

from gnn_dr.datasets.clip_dr_base import CLIPDRDatasetGPUBase
from gnn_dr.datasets.transforms import normalize_embeddings
from gnn_dr.datasets.torchvision_clip import register_torchvision_clip_dataset


@register_torchvision_clip_dataset('laion_clip')
class LAIONClipDynamicGPU(CLIPDRDatasetGPUBase):
    """
    GPU-optimized dynamic LAION-400M CLIP dataset.
    
    Downloads LAION embedding chunks on-the-fly (or loads from cache) and keeps them 
    on GPU for maximum performance. Generates random subset graphs during training.
    
    Note: LAION embeddings are pre-computed (not extracted from images), so this dataset
    downloads .npy files containing embeddings directly.
    
    Example:
        ```python
        # Create dataset for training (downloads 10 chunks, ~10M embeddings)
        train_dataset = LAIONClipDynamicGPU(
            root='data/laion_embeddings',
            num_chunks=10,
            subset_sizes=[100, 500, 1000],
            knn_k=15,
        )
        
        # Get a training graph
        graph = train_dataset[0]
        ```
    """
    
    # Base URL for LAION embeddings
    LAION_BASE_URL = "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb"
    
    def __init__(
        self,
        root: str = 'data/laion_embeddings',
        num_chunks: int = 10,
        subset_sizes: Optional[List[int]] = None,
        knn_k: int = 15,
        seed: int = 42,
        n_samples_per_size: int = 10,
        device: str = 'cuda',
        # For compatibility with TorchvisionCLIPDatasetGPU interface
        train: bool = True,  # Ignored for LAION
        clip_model: str = 'openai/CLIP-vit-base-patch32',  # Ignored, LAION has fixed embeddings
        edge_weight_method: str = 'umap',
        tsne_perplexity: float = 10.0,
    ):
        """
        Initialize GPU-optimized dynamic LAION CLIP dataset.

        Args:
            root: Directory to cache downloaded embeddings
            num_chunks: Number of 1M-embedding chunks to load (0-410, each ~1M embeddings)
            subset_sizes: List of subset sizes to cycle through during training
            knn_k: Number of neighbors for KNN graph
            seed: Random seed
            n_samples_per_size: Number of graphs to generate per subset size
            device: GPU device ('cuda', 'cuda:0', etc.). Must be GPU device.
            train: Ignored (LAION doesn't have train/test split)
            clip_model: Ignored (LAION embeddings are pre-computed with ViT-B/32)
            edge_weight_method: 'umap' (fuzzy simplicial set) or 'tsne' (perplexity-based)
            tsne_perplexity: t-SNE perplexity (only used when edge_weight_method='tsne')
        """
        self.root = Path(root)
        self.num_chunks = num_chunks

        # Initialize common parameters from base class
        self._init_common(
            subset_sizes=subset_sizes,
            knn_k=knn_k,
            seed=seed,
            n_samples_per_size=n_samples_per_size,
            device=device,
            edge_weight_method=edge_weight_method,
            tsne_perplexity=tsne_perplexity,
        )
        
        # LAION has no labels
        self.labels = None
        
        # Download or load embeddings
        self._download_or_load_embeddings()
    
    @property
    def dataset_name(self) -> str:
        return "laion"
    
    def _download_or_load_embeddings(self):
        """Download or load LAION embeddings from disk and keep on GPU."""
        # Create cache directory
        self.root.mkdir(parents=True, exist_ok=True)
        
        embeddings_list = []
        
        print(f"Loading LAION embeddings ({self.num_chunks} chunks, ~{self.num_chunks}M embeddings)...")
        
        for chunk_id in range(self.num_chunks):
            embedding_path = self.root / f'img_emb_{chunk_id:03d}.npy'
            
            # Download if not cached
            if not embedding_path.exists():
                print(f"Downloading LAION embeddings chunk {chunk_id} ({chunk_id + 1}/{self.num_chunks})...")
                url = f"{self.LAION_BASE_URL}/img_emb_{chunk_id}.npy"
                try:
                    urllib.request.urlretrieve(
                        url, 
                        str(embedding_path),
                        reporthook=self._download_progress_hook
                    )
                    print(f"\nDownloaded chunk {chunk_id}")
                except Exception as e:
                    print(f"Error downloading chunk {chunk_id}: {e}")
                    raise
            else:
                print(f"Found cached chunk {chunk_id}")
            
            # Load chunk
            chunk_embeddings = np.load(str(embedding_path))
            # Convert to torch and normalize
            chunk_embeddings = torch.from_numpy(chunk_embeddings).float()
            chunk_embeddings = normalize_embeddings(chunk_embeddings)
            embeddings_list.append(chunk_embeddings)
        
        # Concatenate all chunks and move to GPU
        self.embeddings = torch.cat(embeddings_list, dim=0).to(self.device)
        print(f"Loaded LAION: {self.embeddings.shape[0]} embeddings, dim={self.embeddings.shape[1]} on {self.device}")
    
    def _download_progress_hook(self, block_num, block_size, total_size):
        """Simple progress hook for downloads."""
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100.0 / total_size, 100.0)
            if block_num % 50 == 0:  # Print every 50 blocks to avoid spam
                print(f"  Downloaded {percent:.1f}%", end='\r')


__all__ = [
    'LAIONClipDynamicGPU',
]
