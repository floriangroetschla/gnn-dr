#!/usr/bin/env python
"""
Evaluate GNN-based dimensionality reduction models from checkpoints.

Loads a trained CoRe-DR model from a Lightning checkpoint and evaluates it on
multiple datasets using tf-projection-qm metrics. Output is compatible with
run_full_evaluation.py and show_metric_table.py.

Usage:
    python -m gnn_dr.evaluation.evaluate_model \
        --checkpoint models/last.ckpt \
        --config configs/config_mnist_dr.yaml \
        --output results/gnn_results.csv
    
    python -m gnn_dr.evaluation.evaluate_model \
        --checkpoint models/last.ckpt \
        --datasets mnist_clip cifar10_clip \
        --rounds 10 \
        --output results/gnn_results.csv
"""

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# Import dataset registry from baseline evaluation
from .run_full_evaluation import DATASET_REGISTRY, compute_tfpqm_metrics


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: str = 'cuda',
) -> Tuple[torch.nn.Module, dict]:
    """
    Load a trained model from a Lightning checkpoint.
    
    Args:
        checkpoint_path: Path to .ckpt file
        config_path: Optional path to config YAML (uses checkpoint config if None)
        device: Device to load model on
        
    Returns:
        Tuple of (model, config_dict)
    """
    from gnn_dr.network.lightning_module import CoReGDLightningModule
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    
    # If config_path provided, load it instead
    if config_path:
        from gnn_dr.config.loader import load_config
        config = load_config(config_path)
    else:
        # Reconstruct config from hparams (basic case)
        from gnn_dr.config.config import ExperimentConfig
        config = ExperimentConfig.from_flat_dict(hparams)
    
    # Create Lightning module and load state dict
    # Use strict=False to ignore extra keys like UMAP a/b parameters
    module = CoReGDLightningModule(config)
    module.load_state_dict(checkpoint['state_dict'], strict=False)
    module.eval()
    module = module.to(device)
    
    return module, config


def build_knn_graph(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    k: int = 15,
    device: str = 'cuda',
    edge_weight_method: str = 'umap',
    tsne_perplexity: float = 10.0,
) -> Data:
    """
    Build a KNN graph from embeddings with edge weights (UMAP or t-SNE).

    Uses the same GPU-accelerated approach as CLIPDRDatasetGPUBase:
    - torch_cluster.knn for GPU KNN construction
    - compute_edge_weights for edge weight computation (dispatches to UMAP or t-SNE)
    - Bidirectional edges for GNN message passing

    Args:
        embeddings: High-dimensional embeddings (N, D)
        labels: Optional labels (N,)
        k: Number of nearest neighbors
        device: Device for tensors
        edge_weight_method: 'umap' (fuzzy simplicial set) or 'tsne' (perplexity-based)
        tsne_perplexity: t-SNE perplexity (only used when edge_weight_method='tsne')

    Returns:
        PyG Data object with KNN graph structure and edge weights
    """
    from torch_cluster import knn
    from torch_geometric.utils import remove_self_loops, to_undirected
    from gnn_dr.utils.umap_weights import compute_edge_weights

    N, D = embeddings.shape

    # Convert to GPU tensor
    subset_embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
    batch_idx = torch.zeros(N, dtype=torch.long, device=device)

    # Build DIRECTED KNN graph on GPU (same as CLIPDRDatasetGPUBase)
    edge_index_knn = knn(
        subset_embeddings,
        subset_embeddings,
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
        subset_embeddings[src],
        subset_embeddings[dst]
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
        print(f"Warning: Edge weight computation failed ({e}), using uniform weights")
        edge_index = to_undirected(edge_index_directed)
        edge_weight = torch.ones(edge_index.shape[1], device=device)
    
    # Create Data object (same structure as CLIPDRDatasetGPUBase._build_graph_from_embeddings)
    data = Data(
        x=subset_embeddings,  # CLIP embeddings (will be preprocessed)
        edge_index=edge_index,  # BIDIRECTIONAL edges
        clip_embedding=subset_embeddings.clone(),
        edge_weight=edge_weight,  # UMAP fuzzy weights
        edge_attr=edge_weight.view(-1, 1),  # Edge features for GNN (required for use_edge_attr=True)
        num_nodes=N,
        batch=torch.zeros(N, dtype=torch.long, device=device),
    )
    
    # Add labels if available
    if labels is not None:
        data.y = torch.tensor(labels, dtype=torch.long, device=device)
    else:
        data.y = torch.zeros(N, dtype=torch.long, device=device)
    
    return data


def preprocess_for_inference(data: Data, config) -> Data:
    """
    Preprocess data for model inference, computing all required input features.
    
    Mirrors the preprocessing done during training (random features, beacons,
    spectral features, PCA, rewiring precomputation, etc.) based on the model's config.
    
    Args:
        data: Data object with clip_embedding and edge_index
        config: Model configuration (ExperimentConfig or flat dict with model params)
        
    Returns:
        Data object with properly computed x and x_orig
    """
    from gnn_dr.network.preprocessing import (
        AddLaplacian, AddPCAProjection, AddGaussianRandomProjection, BFS,
        precompute_rewiring_edges
    )
    import math
    import random
    
    # Get config values (handle both ExperimentConfig and flat dict)
    if hasattr(config, 'model'):
        # ExperimentConfig
        model_cfg = config.model
        random_in_channels = model_cfg.random_in_channels
        use_beacons = model_cfg.use_beacons
        num_beacons = model_cfg.num_beacons
        encoding_size_per_beacon = model_cfg.encoding_size_per_beacon
        laplace_eigvec = model_cfg.laplace_eigvec
        pca_dim = model_cfg.pca_dim
        random_projection_dim = model_cfg.random_projection_dim
        clip_in_channels = model_cfg.clip_in_channels
        use_cupy = getattr(config, 'use_cupy', False)
    else:
        # Flat dict
        random_in_channels = config.get('random_in_channels', 1)
        use_beacons = config.get('use_beacons', True)
        num_beacons = config.get('num_beacons', 2)
        encoding_size_per_beacon = config.get('encoding_size_per_beacon', 8)
        laplace_eigvec = config.get('laplace_eigvec', 8)
        pca_dim = config.get('pca_dim', 0)
        random_projection_dim = config.get('random_projection_dim', 0)
        clip_in_channels = config.get('clip_in_channels', 0)
        use_cupy = config.get('use_cupy', False)
    
    # Get CLIP embeddings
    clip_embeddings = data.clip_embedding
    device = clip_embeddings.device
    num_nodes = data.num_nodes
    
    # Build feature list
    features = []
    
    # 1. Random features
    if random_in_channels > 0:
        random_features = torch.rand(num_nodes, random_in_channels, dtype=torch.float32, device=device)
        features.append(random_features)
    
    # 2. Beacon features (BFS-based positional encoding)
    if use_beacons:
        bfs = BFS()
        starting_nodes = random.sample(range(num_nodes), min(num_beacons, num_nodes))
        distances = torch.empty(num_nodes, num_beacons, device=device).fill_(float('Inf'))
        for i, node in enumerate(starting_nodes):
            distances[node, i] = 0
        
        bfs_distances = bfs(data, distances, num_nodes)
        
        div_term = torch.exp(
            torch.arange(0, encoding_size_per_beacon, 2, device=device) * 
            (-math.log(10000.0) / encoding_size_per_beacon)
        )
        
        pes = []
        for beacon_idx in range(num_beacons):
            pe = torch.zeros(num_nodes, encoding_size_per_beacon, device=device)
            pe[:, 0::2] = torch.sin(bfs_distances[:, beacon_idx].unsqueeze(1) * div_term)
            pe[:, 1::2] = torch.cos(bfs_distances[:, beacon_idx].unsqueeze(1) * div_term)
            pes.append(pe)
        
        beacon_features = torch.cat(pes, dim=1)
        features.append(beacon_features)
    
    # 3. Spectral features (Laplacian eigenvectors)
    if laplace_eigvec > 0:
        pe_transform = AddLaplacian(k=laplace_eigvec, attr_name="laplace_ev", 
                                   is_undirected=True, use_cupy=use_cupy)
        data = pe_transform(data)
        features.append(data.laplace_ev)
    
    # 4. Random projection features
    if random_projection_dim > 0:
        rp_transform = AddGaussianRandomProjection(n_components=random_projection_dim, 
                                                   attr_name="random_projection_pe")
        data = rp_transform(data)
        features.append(data.random_projection_pe)
    
    # 5. PCA features
    if pca_dim > 0:
        pca_transform = AddPCAProjection(n_components=pca_dim, attr_name="pca_pe")
        data = pca_transform(data)
        features.append(data.pca_pe)
    
    # 6. CLIP embedding features (sliced if clip_in_channels > 0)
    if clip_in_channels > 0:
        features.append(clip_embeddings[:, :clip_in_channels])
    
    # Concatenate all features
    if features:
        data.x = torch.cat(features, dim=1)
    else:
        # Fallback: use zeros (shouldn't happen with proper config)
        data.x = torch.zeros(num_nodes, 1, dtype=torch.float32, device=device)
    
    # Set x_orig (required by model)
    data.x_orig = data.x.clone()
    
    # 7. Precompute rewiring edges if enabled (for neg_sample rewiring)
    # This avoids recomputing negative edges during each forward pass
    if hasattr(config, 'model'):
        rewiring_precompute = getattr(config.model, 'rewiring_precompute', False)
    else:
        rewiring_precompute = config.get('rewiring_precompute', False)
    
    if rewiring_precompute:
        precompute_rewiring_edges(data, config.model if hasattr(config, 'model') else config)
    
    return data


def run_model_inference(
    model,
    graph: Data,
    rounds: int = 10,
) -> np.ndarray:
    """
    Run model inference to get 2D projections.
    
    Args:
        model: Lightning module with CoRe-DR model
        graph: PyG Data object with KNN graph
        rounds: Number of GNN message passing rounds
        
    Returns:
        2D projections (N, 2) as numpy array
    """
    model.eval()
    with torch.no_grad():
        # Forward pass with encoding (CLIP -> hidden dim -> 2D output)
        predictions, _ = model(graph, rounds, return_layers=True, encode=True)
    
    return predictions.cpu().numpy()


def load_dataset_for_model(
    dataset_name: str,
    split: str = 'test',
    max_samples: int = 10000,
    device: str = 'cuda',
    random_state: int = 42,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Load embeddings from a dataset for model evaluation.
    
    Uses the same loading logic as run_full_evaluation.py but returns
    numpy arrays directly.
    """
    import importlib
    
    if dataset_name not in DATASET_REGISTRY:
        print(f"  Warning: Unknown dataset: {dataset_name}")
        return None, None, 0
    
    config = DATASET_REGISTRY[dataset_name]
    
    try:
        # Import the dataset class dynamically
        module = importlib.import_module(config['module'])
        dataset_class = getattr(module, config['class'])
        
        # Determine train flag
        use_train = (split == 'train') or (not config['has_test'])
        
        # Create dataset to extract embeddings
        dataset = dataset_class(
            root=config['root'],
            train=use_train,
            subset_sizes=[100],  # Minimal - we just need the embeddings
            knn_k=15,
            device=device,
        )
        
        # Get embeddings and labels
        embeddings = dataset.embeddings.cpu().numpy()
        labels = dataset.labels.cpu().numpy() if dataset.labels is not None else None
        
        # Subsample if needed
        if max_samples is not None and embeddings.shape[0] > max_samples:
            np.random.seed(random_state)
            indices = np.random.permutation(embeddings.shape[0])[:max_samples]
            indices = np.sort(indices)
            embeddings = embeddings[indices]
            if labels is not None:
                labels = labels[indices]
        
        return embeddings, labels, embeddings.shape[0]
    
    except Exception as e:
        print(f"  Warning: Could not load {dataset_name}: {e}")
        return None, None, 0


def evaluate_model_on_dataset(
    model,
    embeddings: np.ndarray,
    labels: Optional[np.ndarray],
    config,
    k: int = 15,
    rounds: int = 10,
    device: str = 'cuda',
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Evaluate model on a single dataset.
    
    Args:
        model: Lightning module
        embeddings: High-dimensional embeddings
        labels: Optional labels
        config: Model configuration for preprocessing
        k: Number of KNN neighbors for graph construction
        rounds: Number of GNN rounds
        device: Device for computation
        verbose: Whether to print GPU memory stats
        
    Returns:
        Dictionary with timing and metrics (includes GPU memory stats)
    """
    result = {}
    is_cuda = (device == 'cuda' or (isinstance(device, str) and device.startswith('cuda'))) and torch.cuda.is_available()
    
    # Build KNN graph
    start_time = time.time()
    graph = build_knn_graph(embeddings, labels, k=k, device=device)
    graph_time = time.time() - start_time
    
    # Preprocess graph (compute input features like PCA, beacons, etc.)
    start_time = time.time()
    graph = preprocess_for_inference(graph, config)
    preprocess_time = time.time() - start_time
    
    # ===== GPU Memory Tracking: Before GNN Inference =====
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before_gnn = torch.cuda.memory_allocated()
        mem_reserved_before = torch.cuda.memory_reserved()
    
    # Run inference
    start_time = time.time()
    projection = run_model_inference(model, graph, rounds=rounds)
    inference_time = time.time() - start_time
    
    # ===== GPU Memory Tracking: After GNN Inference =====
    if is_cuda:
        torch.cuda.synchronize()
        mem_after_gnn = torch.cuda.memory_allocated()
        peak_mem_gnn = torch.cuda.max_memory_allocated()
        mem_reserved_after = torch.cuda.memory_reserved()
        
        # Store memory stats
        result['gnn_memory_allocated_mb'] = mem_after_gnn / (1024 * 1024)
        result['gnn_peak_memory_mb'] = peak_mem_gnn / (1024 * 1024)
        result['gnn_memory_delta_mb'] = (mem_after_gnn - mem_before_gnn) / (1024 * 1024)
        result['gnn_reserved_memory_mb'] = mem_reserved_after / (1024 * 1024)
        
        if verbose:
            print(f"    GNN Memory: allocated={result['gnn_memory_allocated_mb']:.1f}MB, "
                  f"peak={result['gnn_peak_memory_mb']:.1f}MB, "
                  f"reserved={result['gnn_reserved_memory_mb']:.1f}MB")
    
    result['time_seconds'] = graph_time + preprocess_time + inference_time
    result['graph_build_time'] = graph_time
    result['preprocess_time'] = preprocess_time
    result['inference_time'] = inference_time
    
    # ===== GPU Memory Tracking: Before Metrics =====
    if is_cuda:
        torch.cuda.synchronize()
        mem_before_metrics = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    
    # Ensure float32 for metrics (same as baseline evaluation)
    embeddings_f32 = embeddings.astype(np.float32)
    projection_f32 = projection.astype(np.float32)
    
    # Compute metrics (uses tf-projection-qm which may use TensorFlow)
    metrics = compute_tfpqm_metrics(embeddings_f32, projection_f32, labels, k=k)
    result.update(metrics)
    
    # ===== GPU Memory Tracking: After Metrics =====
    if is_cuda:
        torch.cuda.synchronize()
        mem_after_metrics = torch.cuda.memory_allocated()
        peak_mem_metrics = torch.cuda.max_memory_allocated()
        
        result['metrics_memory_delta_mb'] = (mem_after_metrics - mem_before_metrics) / (1024 * 1024)
        result['metrics_peak_memory_mb'] = peak_mem_metrics / (1024 * 1024)
        result['total_memory_mb'] = mem_after_metrics / (1024 * 1024)
        
        if verbose:
            print(f"    Metrics Memory: delta={result['metrics_memory_delta_mb']:.1f}MB, "
                  f"peak={result['metrics_peak_memory_mb']:.1f}MB, "
                  f"total={result['total_memory_mb']:.1f}MB")
    
    return result


def run_model_evaluation(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    datasets: Optional[List[str]] = None,
    max_samples: int = 10000,
    k: int = 15,
    rounds: int = 10,
    random_state: int = 42,
    output_path: str = 'results/gnn_results.csv',
    device: str = 'cuda',
    model_name: str = 'CoRe-DR',
    verbose: bool = True,
    compile_model: bool = False,
    compile_mode: str = 'reduce-overhead',
    warmup: bool = True,
) -> pd.DataFrame:
    """
    Run full model evaluation on multiple datasets.
    
    Args:
        checkpoint_path: Path to Lightning checkpoint
        config_path: Optional config YAML path
        datasets: List of datasets to evaluate (None = all)
        max_samples: Maximum samples per dataset
        k: Number of KNN neighbors
        rounds: Number of GNN rounds
        random_state: Random seed
        output_path: Path to save CSV results
        device: Device for computation
        model_name: Name to use in output (for comparison with baselines)
        verbose: Whether to print progress
        compile_model: Whether to use torch.compile for faster inference
        compile_mode: torch.compile mode ('default', 'reduce-overhead', 'max-autotune')
        warmup: Whether to run a warmup inference pass before timing
        
    Returns:
        DataFrame with all results
    """
    if datasets is None:
        datasets = list(DATASET_REGISTRY.keys())
    
    # Auto-detect device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    # Load model
    if verbose:
        print(f"Loading model from: {checkpoint_path}")
    
    model, config = load_model_from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        device=device
    )
    
    if verbose:
        print(f"  Model loaded, device: {device}")
        print(f"  GNN rounds: {rounds}")
        print(f"  KNN k: {k}")
    
    # ===== Apply torch.compile if enabled =====
    if compile_model:
        torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if torch_version < (2, 0):
            print(f"Warning: torch.compile requires PyTorch 2.0+, got {torch.__version__}")
            print("  Skipping compilation.")
        else:
            if verbose:
                print(f"  Compiling model with mode='{compile_mode}'...")
            model.model = torch.compile(
                model.model,
                mode=compile_mode,
                dynamic=True,
                fullgraph=False,
            )

    # ===== Warmup pass =====
    # Always run warmup (unless --no-warmup) to stabilize CUDA kernels,
    # memory allocators, and trigger any lazy init / torch.compile.
    if warmup:
        if verbose:
            print("  Running warmup inference pass...")
        try:
            warmup_embeddings = np.random.randn(200, 768).astype(np.float32)
            warmup_graph = build_knn_graph(warmup_embeddings, k=k, device=device)
            warmup_graph = preprocess_for_inference(warmup_graph, config)
            with torch.no_grad():
                _ = model(warmup_graph, rounds, return_layers=True, encode=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            del warmup_embeddings, warmup_graph
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if verbose:
                print("  ✓ Warmup complete")
        except Exception as e:
            print(f"  Warning: Warmup failed: {e}")
    
    all_results = []
    
    for dataset_name in datasets:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print('='*60)
        
        # Load embeddings
        if verbose:
            print("  Loading embeddings...")
        
        embeddings, labels, actual_size = load_dataset_for_model(
            dataset_name,
            split='test',
            max_samples=max_samples,
            device=device,
            random_state=random_state,
        )
        
        if embeddings is None:
            if verbose:
                print(f"  Skipping {dataset_name} (failed to load)")
            continue
        
        if verbose:
            print(f"  Loaded {actual_size} samples, dim={embeddings.shape[1]}")
            if labels is not None:
                n_classes = len(np.unique(labels))
                print(f"  Classes: {n_classes}")
        
        # Run evaluation
        if verbose:
            print(f"  Running {model_name}...")
        
        try:
            result = evaluate_model_on_dataset(
                model, embeddings, labels, config,
                k=k, rounds=rounds, device=device, verbose=verbose
            )
            
            result['method'] = model_name
            result['dataset'] = dataset_name
            result['n_samples'] = actual_size
            result['n_classes'] = len(np.unique(labels)) if labels is not None else 0
            
            all_results.append(result)
            
            if verbose:
                print(f"    ✓ Completed in {result['time_seconds']:.2f}s")
        
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        finally:
            # ===== Cleanup between datasets =====
            # Clear local variables that hold GPU tensors
            del embeddings, labels
            if 'graph' in dir():
                del graph
            if 'projection' in dir():
                del projection
            
            # Clear PyTorch CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force Python garbage collection
            gc.collect()
            
            if verbose and torch.cuda.is_available():
                mem_after_cleanup = torch.cuda.memory_allocated() / (1024 * 1024)
                print(f"    Memory after cleanup: {mem_after_cleanup:.1f}MB")
    
    # Create DataFrame
    if not all_results:
        if verbose:
            print("\nERROR: No datasets were successfully evaluated!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    # Reorder columns
    first_cols = ['dataset', 'method', 'n_samples', 'n_classes', 'time_seconds']
    first_cols = [c for c in first_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in first_cols and c != 'error']
    if 'error' in df.columns:
        other_cols.append('error')
    df = df[first_cols + other_cols]
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_path}")
        print('='*60)
        print("\nSummary:")
        summary_cols = [c for c in ['dataset', 'method', 'n_samples', 'time_seconds'] if c in df.columns]
        print(df[summary_cols].to_string(index=False))
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GNN-based DR model from checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on all datasets
  python -m gnn_dr.evaluation.evaluate_model \\
      --checkpoint models/last.ckpt \\
      --config configs/config_mnist_dr.yaml
  
  # Evaluate on specific datasets with custom settings
  python -m gnn_dr.evaluation.evaluate_model \\
      --checkpoint models/last.ckpt \\
      --datasets mnist_clip cifar10_clip \\
      --rounds 10 --k 15 \\
      --output results/gnn_eval.csv
        """
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to Lightning checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML (uses checkpoint config if not provided)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='Datasets to evaluate (default: all). Options: ' + ', '.join(DATASET_REGISTRY.keys())
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10000,
        help='Maximum samples per dataset (default: 10000)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=15,
        help='Number of KNN neighbors for graph construction (default: 15)'
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=10,
        help='Number of GNN message passing rounds (default: 10)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/gnn_results.csv',
        help='Output CSV path (default: results/gnn_results.csv)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for computation (default: cuda)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='CoRe-DR',
        help='Model name in output (default: CoRe-DR)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--compile',
        action='store_true',
        help='Use torch.compile for faster inference (requires PyTorch 2.0+)'
    )
    parser.add_argument(
        '--compile-mode',
        type=str,
        default='reduce-overhead',
        choices=['default', 'reduce-overhead', 'max-autotune'],
        help='torch.compile mode (default: reduce-overhead for inference)'
    )
    parser.add_argument(
        '--no-warmup',
        action='store_true',
        help='Skip warmup inference pass before timing'
    )
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    run_model_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        datasets=args.datasets,
        max_samples=args.max_samples,
        k=args.k,
        rounds=args.rounds,
        random_state=args.random_state,
        output_path=args.output,
        device=args.device,
        model_name=args.model_name,
        verbose=not args.quiet,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        warmup=not args.no_warmup,
    )


if __name__ == '__main__':
    main()
