"""Individual dataset loaders for better organization.

This module provides loader functions for all supported datasets, including:
- CLIP-based DR datasets (MNIST, CIFAR-10, CIFAR-100, Fashion-MNIST, LAION)
- Multi-dataset training support
"""

import os
from typing import Optional, List, Dict, Any

from gnn_dr.utils.constants import DATA_ROOT


# =============================================================================
# CLIP-based DR Datasets
# =============================================================================

def _get_clip_dataset_class(dataset_name: str):
    """Get the CLIP dataset class by name."""
    # Import here to ensure all datasets are registered via their decorators
    from . import mnist_dr  # noqa: F401
    from . import cifar_dr  # noqa: F401
    from . import fashion_mnist_dr  # noqa: F401
    from . import laion_dr  # noqa: F401
    from . import kmnist_dr  # noqa: F401
    from . import flowers102_dr  # noqa: F401
    from . import fgvc_aircraft_dr  # noqa: F401
    from . import oxford_pets_dr  # noqa: F401
    from . import food101_dr  # noqa: F401
    from . import emnist_dr  # noqa: F401
    from . import stl10_dr  # noqa: F401
    from . import svhn_dr  # noqa: F401
    from . import caltech_dr  # noqa: F401
    from . import dtd_dr  # noqa: F401
    from . import stanford_cars_dr  # noqa: F401
    from .torchvision_clip import get_registered_clip_dataset

    return get_registered_clip_dataset(dataset_name)


def load_clip_dataset(
    dataset_name: str,
    data_root: str = DATA_ROOT,
    subset_sizes: Optional[List[int]] = None,
    knn_k: int = 15,
    clip_model: str = 'openai/CLIP-vit-base-patch32',
    n_samples_per_size: int = 10,
    val_dataset: Optional[str] = None,
    test_dataset: Optional[str] = None,
    val_subset_size: Optional[int] = None,
    test_subset_size: Optional[int] = None,
    device: str = 'cuda',
    seed: int = 42,
    # LAION-specific
    num_chunks: int = 10,
    # Edge weight method
    edge_weight_method: str = 'umap',
    tsne_perplexity: float = 10.0,
    # Legacy parameter names (for backward compatibility)
    val_test_subset_size: Optional[int] = None,
    use_dynamic_dataset_gpu: bool = True,  # Ignored, always uses GPU
    **kwargs,  # Catch any other legacy parameters
):
    """
    Load any CLIP-based dataset for dimensionality reduction.
    
    This is the unified loader for all CLIP datasets (MNIST, CIFAR-10, CIFAR-100,
    Fashion-MNIST, LAION). It supports configurable validation/test datasets.
    
    Args:
        dataset_name: Name of the training dataset ('mnist_clip', 'cifar10_clip', etc.)
        data_root: Root directory for data
        subset_sizes: List of subset sizes for training
        knn_k: Number of neighbors for KNN graph
        clip_model: CLIP model identifier (ignored for LAION)
        n_samples_per_size: Number of samples per subset size
        val_dataset: Validation dataset name (default: same as training)
        test_dataset: Test dataset name (default: same as training)
        val_subset_size: Size of validation subset (None = full dataset)
        test_subset_size: Size of test subset (None = full dataset)
        device: GPU device
        seed: Random seed
        num_chunks: Number of chunks for LAION dataset
        
    Returns:
        Tuple of (train_dataset, val_dataset_list, test_dataset_list)
    """
    subset_sizes = subset_sizes or [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    # Handle legacy val_test_subset_size parameter
    if val_test_subset_size is not None:
        if val_subset_size is None:
            val_subset_size = val_test_subset_size
        if test_subset_size is None:
            test_subset_size = val_test_subset_size
    
    # Default val/test to same as training dataset
    val_dataset = val_dataset or dataset_name
    test_dataset = test_dataset or dataset_name
    
    print(f"Loading CLIP datasets: train={dataset_name}, val={val_dataset}, test={test_dataset}")
    
    # Get dataset classes
    train_cls = _get_clip_dataset_class(dataset_name)
    
    # Common parameters
    common_params = {
        'subset_sizes': subset_sizes,
        'knn_k': knn_k,
        'seed': seed,
        'device': device,
        'edge_weight_method': edge_weight_method,
        'tsne_perplexity': tsne_perplexity,
    }
    
    # Create training dataset
    train_params = common_params.copy()
    train_params['n_samples_per_size'] = n_samples_per_size
    train_params['train'] = True
    train_params['clip_model'] = clip_model
    
    # Handle LAION-specific parameters
    if dataset_name == 'laion_clip':
        train_params['root'] = os.path.join(data_root, 'laion_embeddings')
        train_params['num_chunks'] = num_chunks
    else:
        train_params['root'] = data_root
    
    train_dataset = train_cls(**train_params)
    
    # Keep track of loaded datasets to avoid reloading
    _loaded_datasets = {(dataset_name, True): train_dataset}  # (name, is_train) -> dataset
    
    def _get_or_create_dataset(name: str, is_train: bool):
        """Get a dataset from cache or create a new one to avoid duplicate loading."""
        key = (name, is_train)
        if key in _loaded_datasets:
            return _loaded_datasets[key]
        
        # Need to create a new dataset
        ds_cls = _get_clip_dataset_class(name)
        ds_params = common_params.copy()
        ds_params['n_samples_per_size'] = 1
        ds_params['train'] = is_train
        ds_params['clip_model'] = clip_model
        
        if name == 'laion_clip':
            ds_params['root'] = os.path.join(data_root, 'laion_embeddings')
            ds_params['num_chunks'] = num_chunks
        else:
            ds_params['root'] = data_root
        
        dataset = ds_cls(**ds_params)
        _loaded_datasets[key] = dataset
        return dataset
    
    # Get validation dataset (reuse train if same dataset)
    val_helper = _get_or_create_dataset(val_dataset, True)  # Use train split for val
    
    if val_subset_size is not None:
        val_graph = val_helper.get_random_subset_graph(val_subset_size)
    else:
        val_graph = val_helper.get_full_graph()
    val_graph.index = 0
    val_graph.coarsening_level = 0
    val_dataset_list = [val_graph]
    
    # Get test dataset (reuse if same dataset and split)
    # For test, we use the test split (train=False) unless it's LAION which has no test split
    if test_dataset == 'laion_clip':
        # LAION has no test split - reuse from train split
        test_helper = _get_or_create_dataset(test_dataset, True)
    else:
        test_helper = _get_or_create_dataset(test_dataset, False)
    
    if test_subset_size is not None:
        test_graph = test_helper.get_random_subset_graph(test_subset_size)
    else:
        test_graph = test_helper.get_full_graph()
    test_graph.index = 0
    test_graph.coarsening_level = 0
    test_dataset_list = [test_graph]
    
    return train_dataset, val_dataset_list, test_dataset_list


def load_multi_clip_dataset(
    train_datasets: List[Dict[str, Any]],
    data_root: str = DATA_ROOT,
    subset_sizes: Optional[List[int]] = None,
    knn_k: int = 15,
    clip_model: str = 'openai/CLIP-vit-base-patch32',
    n_samples_per_size: int = 10,
    val_dataset: str = 'mnist_clip',
    test_dataset: str = 'mnist_clip',
    val_subset_size: Optional[int] = None,
    test_subset_size: Optional[int] = None,
    device: str = 'cuda',
    seed: int = 42,
    # Edge weight method
    edge_weight_method: str = 'umap',
    tsne_perplexity: float = 10.0,
    # Legacy parameter names (for backward compatibility)
    val_test_subset_size: Optional[int] = None,
    use_dynamic_dataset_gpu: bool = True,  # Ignored, always uses GPU
    laion_num_chunks: int = 10,  # Default for LAION datasets
    **kwargs,  # Catch any other parameters
):
    """
    Load multiple CLIP datasets for joint training.
    
    Args:
        train_datasets: List of dicts with 'name' and optional 'weight' keys
            Example: [{'name': 'mnist_clip', 'weight': 1.0}, {'name': 'cifar10_clip', 'weight': 2.0}]
        data_root: Root directory for data
        subset_sizes: List of subset sizes for training
        knn_k: Number of neighbors for KNN graph
        clip_model: CLIP model identifier
        n_samples_per_size: Number of samples per subset size
        val_dataset: Validation dataset name
        test_dataset: Test dataset name
        val_subset_size: Size of validation subset (None = full dataset)
        test_subset_size: Size of test subset (None = full dataset)
        device: GPU device
        seed: Random seed
        
    Returns:
        Tuple of (multi_train_dataset, val_dataset_list, test_dataset_list)
    """
    from .multi_dataset import MultiDatasetCLIPDynamicGPU
    
    subset_sizes = subset_sizes or [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    
    # Handle legacy val_test_subset_size parameter
    if val_test_subset_size is not None:
        if val_subset_size is None:
            val_subset_size = val_test_subset_size
        if test_subset_size is None:
            test_subset_size = val_test_subset_size
    
    print(f"Loading multi-dataset: {[d['name'] for d in train_datasets]}")
    
    # Create individual training datasets
    datasets = []
    weights = []
    
    for config in train_datasets:
        name = config['name']
        weight = config.get('weight', 1.0)
        
        dataset_cls = _get_clip_dataset_class(name)
        
        params = {
            'root': data_root if name != 'laion_clip' else os.path.join(data_root, 'laion_embeddings'),
            'train': True,
            'subset_sizes': subset_sizes,
            'knn_k': knn_k,
            'clip_model': clip_model,
            'seed': seed,
            'n_samples_per_size': n_samples_per_size,
            'device': device,
            'edge_weight_method': edge_weight_method,
            'tsne_perplexity': tsne_perplexity,
        }
        
        if name == 'laion_clip':
            params['num_chunks'] = config.get('num_chunks', 10)
        
        dataset = dataset_cls(**params)
        datasets.append(dataset)
        weights.append(weight)
    
    # Create multi-dataset wrapper
    train_dataset = MultiDatasetCLIPDynamicGPU(
        datasets=datasets,
        weights=weights,
        seed=seed,
    )
    
    # Create validation dataset
    val_cls = _get_clip_dataset_class(val_dataset)
    val_params = {
        'root': data_root if val_dataset != 'laion_clip' else os.path.join(data_root, 'laion_embeddings'),
        'train': True,
        'subset_sizes': subset_sizes,
        'knn_k': knn_k,
        'clip_model': clip_model,
        'seed': seed,
        'n_samples_per_size': 1,
        'device': device,
        'edge_weight_method': edge_weight_method,
        'tsne_perplexity': tsne_perplexity,
    }
    val_helper = val_cls(**val_params)
    
    if val_subset_size is not None:
        val_graph = val_helper.get_random_subset_graph(val_subset_size)
    else:
        val_graph = val_helper.get_full_graph()
    val_graph.index = 0
    val_graph.coarsening_level = 0
    val_dataset_list = [val_graph]
    
    # Create test dataset
    test_cls = _get_clip_dataset_class(test_dataset)
    test_params = {
        'root': data_root if test_dataset != 'laion_clip' else os.path.join(data_root, 'laion_embeddings'),
        'train': False,
        'subset_sizes': subset_sizes,
        'knn_k': knn_k,
        'clip_model': clip_model,
        'seed': seed,
        'n_samples_per_size': 1,
        'device': device,
        'edge_weight_method': edge_weight_method,
        'tsne_perplexity': tsne_perplexity,
    }
    test_helper = test_cls(**test_params)
    
    if test_subset_size is not None:
        test_graph = test_helper.get_random_subset_graph(test_subset_size)
    else:
        test_graph = test_helper.get_full_graph()
    test_graph.index = 0
    test_graph.coarsening_level = 0
    test_dataset_list = [test_graph]
    
    return train_dataset, val_dataset_list, test_dataset_list


# =============================================================================
# Legacy Loaders (for backward compatibility)
# =============================================================================

def load_mnist_clip_dataset(
    data_root=DATA_ROOT, 
    subset_sizes=None, 
    knn_k=15, 
    clip_model='openai/CLIP-vit-base-patch32',
    n_samples_per_size=10, 
    val_test_subset_size=None,
    use_dynamic_dataset_gpu=True,
):
    """
    Load MNIST CLIP embedding dataset for dimensionality reduction.
    
    DEPRECATED: Use load_clip_dataset('mnist_clip', ...) instead.
    This function is kept for backward compatibility.
    """
    return load_clip_dataset(
        dataset_name='mnist_clip',
        data_root=data_root,
        subset_sizes=subset_sizes,
        knn_k=knn_k,
        clip_model=clip_model,
        n_samples_per_size=n_samples_per_size,
        val_subset_size=val_test_subset_size,
        test_subset_size=val_test_subset_size,
    )


def load_laion_clip_dataset(
    data_root=DATA_ROOT, 
    num_chunks=10, 
    subset_sizes=None, 
    knn_k=15, 
    n_samples_per_size=10, 
    val_test_subset_size=10000,
):
    """
    Load LAION-400M CLIP embedding dataset for dimensionality reduction.
    
    DEPRECATED: Use load_clip_dataset('laion_clip', ...) instead.
    This function is kept for backward compatibility.
    
    Note: Uses MNIST for validation and testing (LAION has no labeled test set).
    """
    return load_clip_dataset(
        dataset_name='laion_clip',
        data_root=data_root,
        subset_sizes=subset_sizes,
        knn_k=knn_k,
        n_samples_per_size=n_samples_per_size,
        val_dataset='mnist_clip',
        test_dataset='mnist_clip',
        val_subset_size=val_test_subset_size,
        test_subset_size=val_test_subset_size,
        num_chunks=num_chunks,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def list_available_clip_datasets() -> List[str]:
    """List all available CLIP dataset names."""
    from .torchvision_clip import list_registered_clip_datasets
    return list_registered_clip_datasets()
