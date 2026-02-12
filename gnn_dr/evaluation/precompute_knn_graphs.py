#!/usr/bin/env python
"""
Precompute KNN graphs with UMAP edge weights for all vision datasets.

Builds KNN graphs for each dataset split (train/test/val) and saves them
to disk for faster evaluation. This avoids recomputing graphs at inference time.

Usage:
    python -m gnn_dr.evaluation.precompute_knn_graphs
    python -m gnn_dr.evaluation.precompute_knn_graphs --datasets mnist_clip cifar10_clip
    python -m gnn_dr.evaluation.precompute_knn_graphs --k 30 --output-dir data/precomputed_k30

Output format (.pt files):
    {
        'embeddings': tensor(N, D),      # CLIP embeddings (float16)
        'edge_index': tensor(2, E),      # Bidirectional edges (int64)
        'edge_weight': tensor(E),        # UMAP fuzzy weights (float32)
        'labels': tensor(N),             # Class labels (int64, or None)
        'metadata': {
            'dataset': str,
            'split': 'train'/'test'/'val',
            'k': int,
            'n_samples': int,
            'n_edges': int,
            'embedding_dim': int,
        }
    }
"""

import argparse
import importlib
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# Import dataset registry
from .run_full_evaluation import DATASET_REGISTRY


def build_knn_graph_for_precompute(
    embeddings: torch.Tensor,
    k: int = 15,
    device: str = 'cuda',
    edge_weight_method: str = 'umap',
    tsne_perplexity: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build KNN graph with edge weights (UMAP or t-SNE).

    Uses GPU-accelerated approach:
    - torch_cluster.knn for GPU KNN construction
    - compute_edge_weights for edge weight computation (dispatches to UMAP or t-SNE)
    - Returns bidirectional edges for GNN message passing

    Args:
        embeddings: CLIP embeddings tensor (N, D) on device
        k: Number of nearest neighbors
        device: Device for computation
        edge_weight_method: 'umap' (fuzzy simplicial set) or 'tsne' (perplexity-based)
        tsne_perplexity: t-SNE perplexity (only used when edge_weight_method='tsne')

    Returns:
        Tuple of (edge_index, edge_weight)
    """
    from torch_cluster import knn
    from torch_geometric.utils import remove_self_loops, to_undirected
    from gnn_dr.utils.umap_weights import compute_edge_weights

    N = embeddings.shape[0]

    # Ensure embeddings are float32 and on device
    embeddings = embeddings.float().to(device)
    batch_idx = torch.zeros(N, dtype=torch.long, device=device)

    # Build DIRECTED KNN graph on GPU
    edge_index_knn = knn(
        embeddings,
        embeddings,
        k=min(k, N - 1),
        batch_x=batch_idx,
        batch_y=batch_idx
    )

    # Remove self-loops from directed KNN edges
    edge_index_directed, _ = remove_self_loops(edge_index_knn)

    # Compute edge weights on DIRECTED edges
    src = edge_index_directed[0]
    dst = edge_index_directed[1]

    # Compute cosine distances (same as CLIP datasets)
    d_ij = 1.0 - torch.nn.functional.cosine_similarity(
        embeddings[src],
        embeddings[dst]
    )

    # Compute edge weights - returns UNDIRECTED edges with weights
    try:
        edge_index_und, edge_weight_und = compute_edge_weights(
            method=edge_weight_method,
            edge_index=edge_index_directed,
            d_ij=d_ij,
            num_nodes=N,
            k=k,
            perplexity=tsne_perplexity,
        )

        # Convert to BIDIRECTIONAL for message passing
        edge_index = torch.cat([edge_index_und, edge_index_und.flip(0)], dim=1)
        edge_weight = torch.cat([edge_weight_und, edge_weight_und])
    except Exception as e:
        print(f"    Warning: Edge weight computation failed ({e}), using uniform weights")
        edge_index = to_undirected(edge_index_directed)
        edge_weight = torch.ones(edge_index.shape[1], device=device)

    return edge_index, edge_weight


def load_dataset_split(
    dataset_name: str,
    split: str = 'train',
    device: str = 'cuda',
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load embeddings and labels from a dataset split.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'mnist_clip')
        split: 'train', 'test', or 'val'
        device: Device for loading
        
    Returns:
        Tuple of (embeddings, labels) as tensors, or (None, None) on failure
    """
    if dataset_name not in DATASET_REGISTRY:
        print(f"  Unknown dataset: {dataset_name}")
        return None, None
    
    config = DATASET_REGISTRY[dataset_name]
    
    try:
        # Import the dataset class dynamically
        module = importlib.import_module(config['module'])
        dataset_class = getattr(module, config['class'])
        
        # Determine train flag based on split
        if split == 'train':
            use_train = True
        elif split == 'test':
            if not config['has_test']:
                return None, None  # Dataset has no test split
            use_train = False
        elif split == 'val':
            # Check if dataset supports validation split
            # Most torchvision datasets don't have explicit val split
            # We'll skip validation for now
            return None, None
        else:
            print(f"  Unknown split: {split}")
            return None, None
        
        # Create dataset to extract embeddings
        dataset = dataset_class(
            root=config['root'],
            train=use_train,
            subset_sizes=[100],  # Minimal - we just need the embeddings
            knn_k=15,
            device=device,
        )
        
        # Get embeddings and labels (keep on GPU)
        embeddings = dataset.embeddings
        labels = dataset.labels
        
        return embeddings, labels
    
    except Exception as e:
        print(f"  Error loading {dataset_name}/{split}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def precompute_and_save(
    dataset_name: str,
    split: str,
    output_dir: Path,
    k: int = 15,
    device: str = 'cuda',
    verbose: bool = True,
) -> bool:
    """
    Precompute KNN graph for a single dataset split and save to file.
    
    Args:
        dataset_name: Name of the dataset
        split: 'train' or 'test'
        output_dir: Base output directory
        k: Number of nearest neighbors
        device: Device for computation
        verbose: Whether to print progress
        
    Returns:
        True if successful, False otherwise
    """
    if verbose:
        print(f"  Processing {split} split...")
    
    # Load embeddings and labels
    embeddings, labels = load_dataset_split(dataset_name, split, device)
    
    if embeddings is None:
        if verbose:
            print(f"    Skipped (no {split} split)")
        return False
    
    N, D = embeddings.shape
    if verbose:
        print(f"    Loaded {N} samples, dim={D}")
    
    # Build KNN graph
    start_time = time.time()
    edge_index, edge_weight = build_knn_graph_for_precompute(embeddings, k=k, device=device)
    graph_time = time.time() - start_time
    
    if verbose:
        print(f"    Built KNN graph in {graph_time:.2f}s ({edge_index.shape[1]} edges)")
    
    # Prepare output directory
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data dict (move to CPU for saving)
    data = {
        'embeddings': embeddings.half().cpu(),  # Save as float16 to reduce size
        'edge_index': edge_index.cpu(),
        'edge_weight': edge_weight.cpu(),
        'labels': labels.cpu() if labels is not None else None,
        'metadata': {
            'dataset': dataset_name,
            'split': split,
            'k': k,
            'n_samples': N,
            'n_edges': edge_index.shape[1],
            'embedding_dim': D,
        }
    }
    
    # Save to file
    output_path = dataset_dir / f'{split}.pt'
    torch.save(data, output_path)
    
    if verbose:
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"    Saved to {output_path} ({file_size:.1f} MB)")
    
    return True


def precompute_all_datasets(
    datasets: Optional[List[str]] = None,
    output_dir: str = 'data/precomputed',
    k: int = 15,
    device: str = 'cuda',
    verbose: bool = True,
) -> Dict[str, Dict[str, bool]]:
    """
    Precompute KNN graphs for all datasets.
    
    Args:
        datasets: List of datasets to process (None = all)
        output_dir: Base output directory
        k: Number of nearest neighbors
        device: Device for computation
        verbose: Whether to print progress
        
    Returns:
        Dictionary of {dataset: {split: success}}
    """
    if datasets is None:
        datasets = list(DATASET_REGISTRY.keys())
    
    # Auto-detect device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    output_dir = Path(output_dir)
    results = {}
    
    if verbose:
        print(f"Precomputing KNN graphs (k={k}) to {output_dir}")
        print(f"Device: {device}")
        print(f"Datasets: {len(datasets)}")
        print("=" * 60)
    
    for dataset_name in datasets:
        if verbose:
            print(f"\nDataset: {dataset_name}")
            print("-" * 40)
        
        results[dataset_name] = {}
        
        # Process train split
        results[dataset_name]['train'] = precompute_and_save(
            dataset_name, 'train', output_dir, k=k, device=device, verbose=verbose
        )
        
        # Process test split
        results[dataset_name]['test'] = precompute_and_save(
            dataset_name, 'test', output_dir, k=k, device=device, verbose=verbose
        )
        
        # Clear GPU memory between datasets
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Summary:")
        print("=" * 60)
        total_success = 0
        total_skipped = 0
        for ds, splits in results.items():
            status = []
            for split, success in splits.items():
                if success:
                    status.append(f"{split}:✓")
                    total_success += 1
                else:
                    status.append(f"{split}:✗")
                    total_skipped += 1
            print(f"  {ds}: {', '.join(status)}")
        print(f"\nTotal: {total_success} splits saved, {total_skipped} skipped")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Precompute KNN graphs with UMAP weights for vision datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Precompute all datasets
  python -m gnn_dr.evaluation.precompute_knn_graphs
  
  # Precompute specific datasets
  python -m gnn_dr.evaluation.precompute_knn_graphs --datasets mnist_clip cifar10_clip
  
  # Use different k value
  python -m gnn_dr.evaluation.precompute_knn_graphs --k 30 --output-dir data/precomputed_k30
        """
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='Datasets to precompute (default: all). Options: ' + ', '.join(DATASET_REGISTRY.keys())
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/precomputed',
        help='Output directory (default: data/precomputed)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=15,
        help='Number of KNN neighbors (default: 15)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for computation (default: cuda)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    precompute_all_datasets(
        datasets=args.datasets,
        output_dir=args.output_dir,
        k=args.k,
        device=args.device,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
