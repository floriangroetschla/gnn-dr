"""
Comprehensive baseline evaluation script for dimensionality reduction.

Runs all baseline methods (PCA, t-SNE, UMAP, Parametric UMAP) on all available
datasets and computes tf-projection-qm metrics.

Usage:
    python -m gnn_dr.evaluation.run_full_evaluation
    python -m gnn_dr.evaluation.run_full_evaluation --output results/baselines.csv
    python -m gnn_dr.evaluation.run_full_evaluation --max-samples 10000 --k 15
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# Dataset Registry
# =============================================================================

# All available datasets with their configurations
DATASET_REGISTRY = {
    'mnist_clip': {
        'module': 'gnn_dr.datasets.mnist_dr',
        'class': 'MNISTClipDynamicGPU',
        'root': 'data/MNIST',
        'has_test': True,
    },
    'fashion_mnist_clip': {
        'module': 'gnn_dr.datasets.fashion_mnist_dr',
        'class': 'FashionMNISTClipDynamicGPU',
        'root': 'data/FashionMNIST',
        'has_test': True,
    },
    'cifar10_clip': {
        'module': 'gnn_dr.datasets.cifar_dr',
        'class': 'CIFAR10ClipDynamicGPU',
        'root': 'data/CIFAR10',
        'has_test': True,
    },
    'cifar100_clip': {
        'module': 'gnn_dr.datasets.cifar_dr',
        'class': 'CIFAR100ClipDynamicGPU',
        'root': 'data/CIFAR100',
        'has_test': True,
    },
    'svhn_clip': {
        'module': 'gnn_dr.datasets.svhn_dr',
        'class': 'SVHNClipDynamicGPU',
        'root': 'data/SVHN',
        'has_test': True,
    },
    'stl10_clip': {
        'module': 'gnn_dr.datasets.stl10_dr',
        'class': 'STL10ClipDynamicGPU',
        'root': 'data/STL10',
        'has_test': True,
    },
    'emnist_clip': {
        'module': 'gnn_dr.datasets.emnist_dr',
        'class': 'EMNISTClipDynamicGPU',
        'root': 'data/EMNIST',
        'has_test': True,
    },
    'kmnist_clip': {
        'module': 'gnn_dr.datasets.kmnist_dr',
        'class': 'KMNISTClipDynamicGPU',
        'root': 'data/KMNIST',
        'has_test': True,
    },
    'flowers102_clip': {
        'module': 'gnn_dr.datasets.flowers102_dr',
        'class': 'Flowers102ClipDynamicGPU',
        'root': 'data/Flowers102',
        'has_test': True,
    },
    'caltech101_clip': {
        'module': 'gnn_dr.datasets.caltech_dr',
        'class': 'Caltech101ClipDynamicGPU',
        'root': 'data/Caltech101',
        'has_test': False,  # Uses train split
    },
    'fgvc_aircraft_clip': {
        'module': 'gnn_dr.datasets.fgvc_aircraft_dr',
        'class': 'FGVCAircraftClipDynamicGPU',
        'root': 'data/FGVCAircraft',
        'has_test': True,
    },
    'stanford_cars_clip': {
        'module': 'gnn_dr.datasets.stanford_cars_dr',
        'class': 'StanfordCarsClipDynamicGPU',
        'root': 'data/StanfordCars',
        'has_test': True,
    },
    'oxford_pets_clip': {
        'module': 'gnn_dr.datasets.oxford_pets_dr',
        'class': 'OxfordIIITPetClipDynamicGPU',
        'root': 'data/OxfordIIITPet',
        'has_test': True,
    },
    'food101_clip': {
        'module': 'gnn_dr.datasets.food101_dr',
        'class': 'Food101ClipDynamicGPU',
        'root': 'data/Food101',
        'has_test': True,
    },
    'dtd_clip': {
        'module': 'gnn_dr.datasets.dtd_dr',
        'class': 'DTDClipDynamicGPU',
        'root': 'data/DTD',
        'has_test': True,
    },
}


def load_dataset_embeddings(
    dataset_name: str,
    split: str = 'test',
    max_samples: int = 10000,
    device: str = 'cpu',
    random_state: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """
    Load CLIP embeddings from a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'mnist_clip')
        split: 'train' or 'test'
        max_samples: Maximum number of samples to load
        device: Device for loading (use 'cpu' to save GPU memory)
        random_state: Random seed for subsampling
        
    Returns:
        Tuple of (embeddings, labels, actual_size)
    """
    import importlib
    
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")
    
    config = DATASET_REGISTRY[dataset_name]
    
    # Import the dataset class dynamically
    module = importlib.import_module(config['module'])
    dataset_class = getattr(module, config['class'])
    
    # Determine train flag
    use_train = (split == 'train') or (not config['has_test'])
    
    try:
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


# =============================================================================
# Metrics Computation using tf-projection-qm
# =============================================================================

def compute_tfpqm_metrics(
    X_high: np.ndarray,
    X_low: np.ndarray,
    labels: Optional[np.ndarray] = None,
    k: int = 15,
) -> Dict[str, float]:
    """
    Compute all tf-projection-qm metrics.
    
    Args:
        X_high: High-dimensional data (N, D)
        X_low: Low-dimensional projection (N, 2)
        labels: Optional class labels (N,)
        k: Number of neighbors for local metrics
        
    Returns:
        Dictionary of metric names to values
    """
    try:
        # Import tf-projection-qm (same as training evaluation)
        sys.path.insert(0, 'tf-projection-qm/src')
        from tensorflow_projection_qm.metrics import run_all_metrics as tfpqm_run_all_metrics
        
        # Ensure float32 for consistent dtype (as done in training)
        # CLIP embeddings may be float16, projection output may be float64
        X_high = X_high.astype(np.float32)
        X_low = X_low.astype(np.float32)
        
        # Compute number of classes if labels provided
        n_classes = len(np.unique(labels)) if labels is not None else None
        
        # Compute all metrics (same call signature as training evaluation)
        all_metrics = tfpqm_run_all_metrics(
            X=X_high,
            X_2d=X_low,
            y=labels,
            k=k,
            n_classes=n_classes,
            as_numpy=True,  # Return numpy values instead of TF tensors
        )
        
        # Convert to dictionary format
        metrics = {}
        for metric_name, metric_value in all_metrics.items():
            # Handle both scalar and array values
            if hasattr(metric_value, 'numpy'):
                metric_value = metric_value.numpy()
            if isinstance(metric_value, np.ndarray):
                metric_value = float(metric_value)
            metrics[metric_name] = metric_value
        
        return metrics
    
    except ImportError as e:
        print(f"  Warning: tf-projection-qm not available: {e}")
        # Fall back to basic metrics
        return compute_fallback_metrics(X_high, X_low, labels, k)
    except Exception as e:
        print(f"  Warning: Error computing tfpqm metrics: {e}")
        import traceback
        traceback.print_exc()
        return {}


def compute_fallback_metrics(
    X_high: np.ndarray,
    X_low: np.ndarray,
    labels: Optional[np.ndarray] = None,
    k: int = 15,
) -> Dict[str, float]:
    """
    Compute basic DR metrics as fallback when tf-projection-qm is not available.
    """
    from gnn_dr.baselines.metrics import DRMetrics
    
    metrics_computer = DRMetrics(k_neighbors=k)
    results = {}
    
    try:
        results['Trustworthiness'] = metrics_computer.trustworthiness(X_high, X_low)
    except:
        results['Trustworthiness'] = np.nan
    
    try:
        results['Continuity'] = metrics_computer.continuity(X_high, X_low)
    except:
        results['Continuity'] = np.nan
    
    try:
        results['KNN_Recall'] = metrics_computer.knn_recall(X_high, X_low)
    except:
        results['KNN_Recall'] = np.nan
    
    try:
        results['Distance_Correlation'] = metrics_computer.distance_correlation(X_high, X_low)
    except:
        results['Distance_Correlation'] = np.nan
    
    if labels is not None:
        try:
            results['Silhouette'] = metrics_computer.silhouette(X_low, labels)
        except:
            results['Silhouette'] = np.nan
    
    return results


# =============================================================================
# Baseline Methods
# =============================================================================

def create_baselines(k: int = 15, random_state: int = 42) -> Dict[str, object]:
    """
    Create all baseline methods.
    
    Args:
        k: Number of neighbors for UMAP
        random_state: Random seed
        
    Returns:
        Dictionary of baseline names to baseline objects
    """
    from gnn_dr.baselines import PCABaseline, TSNEBaseline, UMAPBaseline
    
    baselines = {}
    
    # PCA (fast, linear baseline)
    baselines['PCA'] = PCABaseline(n_components=2, random_state=random_state)
    
    # t-SNE (standard perplexity=30)
    baselines['t-SNE'] = TSNEBaseline(
        n_components=2,
        perplexity=30.0,
        n_iter=1000,
        random_state=random_state,
    )
    
    # UMAP (transductive)
    try:
        baselines['UMAP'] = UMAPBaseline(
            n_neighbors=k,
            n_components=2,
            random_state=random_state,
        )
    except ImportError:
        print("Warning: umap-learn not installed, skipping UMAP")
    
    # Parametric UMAP (inductive, PyTorch-based)
    try:
        from gnn_dr.baselines import ParametricUMAPBaseline
        baselines['Parametric_UMAP'] = ParametricUMAPBaseline(
            n_neighbors=k,
            n_components=2,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
    except ImportError:
        print("Warning: parametric_umap not installed, skipping Parametric UMAP")
    
    return baselines


# Methods that support true inductive evaluation (fit on train, eval on test)
INDUCTIVE_METHODS = {'Parametric_UMAP'}


def warmup_baseline(baseline_name: str, baseline, embeddings: np.ndarray):
    """
    Run a warmup pass on a small subset to stabilize timing.

    Warms up CPU caches, JIT, and any lazy initialization.
    PCA is skipped (trivially fast, no benefit).
    Parametric_UMAP is skipped (TF manages its own warmup; re-fitting
    can leave stale GPU state that causes NaN).
    """
    if baseline_name in ('PCA', 'Parametric_UMAP'):
        return

    warmup_n = min(500, embeddings.shape[0])
    warmup_data = embeddings[:warmup_n]
    try:
        baseline.fit_transform(warmup_data)
        baseline.is_fitted = False
    except Exception:
        pass  # warmup failure is non-fatal


def run_baseline(
    baseline_name: str,
    baseline,
    test_embeddings: np.ndarray,
    test_labels: Optional[np.ndarray],
    train_embeddings: Optional[np.ndarray] = None,
    k: int = 15,
    warmup: bool = True,
) -> Dict[str, any]:
    """
    Run a single baseline and compute metrics.

    Inductive methods (PCA, Parametric UMAP) are fit on train_embeddings and
    evaluated on test_embeddings. Transductive methods (t-SNE, UMAP) are
    fit and evaluated on test_embeddings only.

    Args:
        baseline_name: Name of the baseline
        baseline: Baseline object
        test_embeddings: Test set embeddings (used for evaluation)
        test_labels: Optional test labels
        train_embeddings: Optional train set embeddings (used for fitting inductive methods)
        k: Number of neighbors for metrics
        warmup: Whether to run a warmup pass before timing

    Returns:
        Dictionary with method name, timing, and metrics
    """
    is_inductive = baseline_name in INDUCTIVE_METHODS and train_embeddings is not None
    mode = "inductive" if is_inductive else "transductive"
    print(f"    Running {baseline_name} ({mode})...")

    result = {'method': baseline_name, 'eval_mode': mode}

    try:
        if warmup:
            warmup_baseline(baseline_name, baseline, test_embeddings)

        if is_inductive:
            # Fit on train, transform test (timed together)
            start_time = time.time()
            baseline.fit(train_embeddings)
            projection = baseline.transform(test_embeddings)
            total_time = time.time() - start_time
        else:
            # Transductive: fit and transform on test data
            start_time = time.time()
            projection = baseline.fit_transform(test_embeddings)
            total_time = time.time() - start_time

        result['time_seconds'] = total_time
        result['fit_time'] = baseline.get_train_time()
        result['inference_time'] = baseline.get_inference_time()

        # Metrics are always computed on test embeddings vs test projection
        metrics = compute_tfpqm_metrics(test_embeddings, projection, test_labels, k=k)
        result.update(metrics)

        print(f"      ✓ Completed in {total_time:.2f}s")

    except Exception as e:
        print(f"      ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        result['error'] = str(e)

    return result


# =============================================================================
# Main Evaluation
# =============================================================================

def run_full_evaluation(
    datasets: Optional[List[str]] = None,
    max_samples: int = 10000,
    k: int = 15,
    random_state: int = 42,
    output_path: str = 'results/baselines.csv',
    device: str = 'cuda',
    verbose: bool = True,
    warmup: bool = True,
) -> pd.DataFrame:
    """
    Run full evaluation on all datasets with all baselines.

    Args:
        datasets: List of datasets to evaluate (None = all)
        max_samples: Maximum samples per dataset
        k: Number of neighbors for metrics
        random_state: Random seed
        output_path: Path to save CSV results
        device: Device for loading datasets ('cuda' or 'cpu')
        verbose: Whether to print progress
        warmup: Whether to run warmup passes before timing

    Returns:
        DataFrame with all results
    """
    if datasets is None:
        datasets = list(DATASET_REGISTRY.keys())
    
    # Auto-detect device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    
    # Create baselines
    if verbose:
        print("Creating baselines...")
    baselines = create_baselines(k=k, random_state=random_state)
    if verbose:
        print(f"  Baselines: {list(baselines.keys())}")
        print(f"  Device: {device}")
    
    # Check if any inductive methods need train data
    has_inductive = any(name in INDUCTIVE_METHODS for name in baselines)

    all_results = []

    for dataset_name in datasets:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print('='*60)

        # Load test embeddings
        if verbose:
            print("  Loading test embeddings...")

        test_embeddings, test_labels, test_size = load_dataset_embeddings(
            dataset_name,
            split='test',
            max_samples=max_samples,
            device=device,
            random_state=random_state,
        )

        if test_embeddings is None:
            if verbose:
                print(f"  Skipping {dataset_name} (failed to load)")
            continue

        if verbose:
            print(f"  Test: {test_size} samples, dim={test_embeddings.shape[1]}")
            if test_labels is not None:
                n_classes = len(np.unique(test_labels))
                print(f"  Classes: {n_classes}")

        # Load train embeddings for inductive methods
        train_embeddings = None
        if has_inductive:
            ds_config = DATASET_REGISTRY.get(dataset_name, {})
            if ds_config.get('has_test', False):
                if verbose:
                    print("  Loading train embeddings for inductive methods...")
                train_embeddings, _, train_size = load_dataset_embeddings(
                    dataset_name,
                    split='train',
                    max_samples=max_samples,
                    device=device,
                    random_state=random_state,
                )
                if train_embeddings is not None and verbose:
                    print(f"  Train: {train_size} samples")
            else:
                if verbose:
                    print("  No separate train split; inductive methods will use transductive eval")

        # Run all baselines
        for baseline_name, baseline in baselines.items():
            result = run_baseline(
                baseline_name, baseline,
                test_embeddings, test_labels,
                train_embeddings=train_embeddings,
                k=k, warmup=warmup,
            )
            result['dataset'] = dataset_name
            result['n_samples'] = test_size
            result['n_classes'] = len(np.unique(test_labels)) if test_labels is not None else 0
            all_results.append(result)
    
    # Create DataFrame
    if not all_results:
        if verbose:
            print("\n" + "="*60)
            print("ERROR: No datasets were successfully loaded!")
            print("="*60)
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    # Reorder columns (only include columns that exist)
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
        description='Run comprehensive baseline evaluation for dimensionality reduction'
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
        help='Number of neighbors for metrics (default: 15)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/baselines.csv',
        help='Output CSV path (default: results/baselines.csv)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for loading datasets (default: cuda)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--no-warmup',
        action='store_true',
        help='Skip warmup passes before timing'
    )

    args = parser.parse_args()

    run_full_evaluation(
        datasets=args.datasets,
        max_samples=args.max_samples,
        k=args.k,
        random_state=args.random_state,
        output_path=args.output,
        device=args.device,
        verbose=not args.quiet,
        warmup=not args.no_warmup,
    )


if __name__ == '__main__':
    main()
