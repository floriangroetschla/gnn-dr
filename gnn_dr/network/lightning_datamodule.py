"""
PyTorch Lightning DataModule for CoRe-DR.
"""

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import random
import numpy as np
import torch

from gnn_dr.network import preprocessing
from gnn_dr.network.preprocessing import preprocess_dataset_dr
from gnn_dr.datasets.loaders import load_clip_dataset, load_multi_clip_dataset
from gnn_dr.config import config_to_namespace
from torch_geometric.data import Data
import torch_geometric.data


def create_coarsened_dataset_dr(config, dataset):
    """
    Create coarsened dataset for dimensionality reduction tasks.

    Similar to create_coarsened_dataset but handles DR-specific attributes.
    DR graphs don't have full_edge_index/full_edge_attr but do have CLIP embeddings.

    Follows the same pattern as GD coarsening: coarsen first, then preprocess.

    Requires: pygsp, graph-coarsening (pip install pygsp graph-coarsening)

    Args:
        config: Configuration object with coarsening parameters
        dataset: List of PyG Data objects with CLIP embeddings and KNN graphs (raw, not preprocessed)

    Returns:
        Tuple of (coarsened_dataset, coarsened_pyg, coarsening_matrices)
    """
    import pygsp as gsp
    from torch_geometric.utils.convert import to_scipy_sparse_matrix, from_scipy_sparse_matrix
    from gnn_dr.coarsening import coarsen
    from tqdm import tqdm

    dataset_gsp = [gsp.graphs.Graph(to_scipy_sparse_matrix(G.edge_index)) for G in dataset]

    method = config.coarsen_algo
    r    = config.coarsen_r
    k    = config.coarsen_k 
        
    coarsened_pyg = []
    coarsening_matrices = []
    for i in tqdm(range(len(dataset)), desc="Coarsening DR dataset"):
        pyg_graphs = [dataset[i]]
        matrices = []
        while pyg_graphs[-1].num_nodes > config.coarsen_min_size:
            C, Gc, Call, Gall = coarsen(gsp.graphs.Graph(to_scipy_sparse_matrix(pyg_graphs[-1].edge_index)), K=k, r=r, method=method, max_levels=1)
            edge_index = from_scipy_sparse_matrix(Gall[1].W)[0]
            num_nodes = Gall[1].W.shape[0]
            
            # Aggregate node features (x and clip_embedding) using mean pooling
            # Call[0] is scipy sparse matrix: (num_coarse, num_fine)
            Call_dense = torch.FloatTensor(Call[0].toarray())
            
            # Get current graph's features
            current_x = pyg_graphs[-1].x.float()
            
            # Aggregate x using mean pooling
            summed_x = torch.mm(Call_dense, current_x)
            cluster_sizes = torch.FloatTensor(np.array(Call[0].sum(axis=1)).flatten()).view(-1, 1)
            coarse_x = summed_x / cluster_sizes
            
            # Aggregate clip_embedding if it exists
            if hasattr(pyg_graphs[-1], 'clip_embedding') and pyg_graphs[-1].clip_embedding is not None:
                current_clip = pyg_graphs[-1].clip_embedding.float()
                summed_clip = torch.mm(Call_dense, current_clip)
                coarse_clip = summed_clip / cluster_sizes
            else:
                coarse_clip = None
            
            # Create coarsened graph with aggregated features
            coarse_data = Data(
                x=coarse_x,
                edge_index=edge_index,
                num_nodes=num_nodes,
                batch=torch.zeros(num_nodes, dtype=torch.long),  # Required for collation
                y=torch.full((num_nodes,), -1, dtype=torch.long)  # Dummy labels for coarsened nodes
            )
            if coarse_clip is not None:
                coarse_data.clip_embedding = coarse_clip
            
            pyg_graphs.append(coarse_data)
            matrices.append(Call[0])
        coarsened_pyg.append(pyg_graphs)

        for i in range(len(matrices)):
            matrices[i][matrices[i] > 0] = 1.0
            matrices[i] = matrices[i].tocoo()
            matrices[i] = torch.sparse_coo_tensor(
                torch.LongTensor([matrices[i].row.tolist(), matrices[i].col.tolist()]),
                torch.FloatTensor(matrices[i].data),
                dtype=torch.float32
            )
        coarsening_matrices.append(matrices)
    
    # Preprocess all coarsening levels, not just the final one
    # This ensures each level has proper x_orig that can be used during forward passes
    for i in range(len(coarsened_pyg)):
        # Preprocess each level of the hierarchy
        for idx, level_graph in enumerate(coarsened_pyg[i]):
            # Preprocess this single graph
            preprocessed_level = preprocessing.preprocess_dataset_dr([level_graph], config)
            # Replace the graph in the hierarchy with the preprocessed version
            coarsened_pyg[i][idx] = preprocessed_level[0]
            # Mark coarsening level
            coarsened_pyg[i][idx].coarsening_level = len(coarsened_pyg[i]) - idx - 1
    
    # Return the final (most coarsened) level as the main dataset
    preprocessed_dataset = [coarsened[-1] for coarsened in coarsened_pyg]
    for i in range(len(preprocessed_dataset)):
        preprocessed_dataset[i].index = i
        
    return preprocessed_dataset, coarsened_pyg, coarsening_matrices


class CoReGDDataModule(pl.LightningDataModule):

    """
    PyTorch Lightning DataModule for CoRe-DR datasets.

    Handles:
    - CLIP dataset loading and preprocessing
    - Train/val/test split
    - Graph coarsening (optional)
    - DataLoader creation
    - Node budget batching
    - Multi-dataset support
    """
    
    def __init__(self, config):
        """
        Initialize the DataModule.
        
        Args:
            config: ExperimentConfig object
        """
        super().__init__()
        self.config = config
        
        # Will be populated in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Coarsening data structures
        self.coarsened_graphs = None
        self.coarsening_matrices = None
        self.val_coarsened_graphs = None
        self.val_coarsening_matrices = None
        self.test_coarsened_graphs = None
        self.test_coarsening_matrices = None
        
    def prepare_data(self):
        """Download data if needed (called only on 1 GPU/TPU)."""
        # Ensure base images are downloaded for CLIP embedding computation
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        transform = transforms.ToTensor()
        datasets.MNIST(
            root=self.config.dimensionality_reduction.clip_cache_dir,
            train=True,
            download=True,
            transform=transform
        )
        datasets.MNIST(
            root=self.config.dimensionality_reduction.clip_cache_dir,
            train=False,
            download=True,
            transform=transform
        )
    
    def setup(self, stage=None):
        """
        Setup datasets for train/val/test.

        Args:
            stage: 'fit', 'validate', 'test', or None
        """
        # Set random seeds
        seed = self.config.training.run_number
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Load CLIP dataset for dimensionality reduction
        dataset_name = self.config.dataset.dataset

        # Build kwargs for dataset loader
        kwargs = {
            'subset_sizes': self.config.dimensionality_reduction.subset_sizes,
            'knn_k': self.config.dimensionality_reduction.knn_k,
            'n_samples_per_size': self.config.dimensionality_reduction.n_samples_per_size,
            'val_test_subset_size': self.config.dimensionality_reduction.val_test_subset_size,
            'edge_weight_method': getattr(self.config.dimensionality_reduction, 'edge_weight_method', 'umap'),
            'tsne_perplexity': getattr(self.config.dimensionality_reduction, 'tsne_perplexity', 10.0),
        }

        # Add val/test dataset configuration if specified
        if self.config.dimensionality_reduction.val_dataset:
            kwargs['val_dataset'] = self.config.dimensionality_reduction.val_dataset
        if self.config.dimensionality_reduction.test_dataset:
            kwargs['test_dataset'] = self.config.dimensionality_reduction.test_dataset
        if self.config.dimensionality_reduction.val_subset_size:
            kwargs['val_subset_size'] = self.config.dimensionality_reduction.val_subset_size
        if self.config.dimensionality_reduction.test_subset_size:
            kwargs['test_subset_size'] = self.config.dimensionality_reduction.test_subset_size

        # Add dataset-specific kwargs and load
        if dataset_name == 'multi_clip':
            kwargs['train_datasets'] = self.config.dataset.train_datasets
            kwargs['use_dynamic_dataset_gpu'] = self.config.dimensionality_reduction.use_dynamic_dataset_gpu
            kwargs['laion_num_chunks'] = self.config.dimensionality_reduction.laion_num_chunks
            kwargs['clip_model'] = self.config.dimensionality_reduction.clip_model
            train_set, val_set, test_set = load_multi_clip_dataset(
                **kwargs
            )
        else:
            # Single CLIP dataset (mnist_clip, cifar10_clip, laion_clip, etc.)
            if dataset_name == 'laion_clip':
                kwargs['num_chunks'] = self.config.dimensionality_reduction.laion_num_chunks
            else:
                kwargs['clip_model'] = self.config.dimensionality_reduction.clip_model
                kwargs['use_dynamic_dataset_gpu'] = self.config.dimensionality_reduction.use_dynamic_dataset_gpu
            train_set, val_set, test_set = load_clip_dataset(
                dataset_name=dataset_name,
                **kwargs
            )

        # Convert config to namespace
        config_namespace = config_to_namespace(self.config)

        # Apply coarsening if enabled (includes preprocessing the final level)
        if self.config.coarsening.coarsen:
            if stage in (None, 'fit'):
                train_set, self.coarsened_graphs, self.coarsening_matrices = \
                    create_coarsened_dataset_dr(config_namespace, train_set)
                val_set, self.val_coarsened_graphs, self.val_coarsening_matrices = \
                    create_coarsened_dataset_dr(config_namespace, val_set)

            if stage in (None, 'test'):
                test_set, self.test_coarsened_graphs, self.test_coarsening_matrices = \
                    create_coarsened_dataset_dr(config_namespace, test_set)
        else:
            # No coarsening: just preprocess the raw graphs
            if stage in (None, 'fit'):
                train_set = preprocess_dataset_dr(train_set, config_namespace)
                val_set = preprocess_dataset_dr(val_set, config_namespace)

            if stage in (None, 'test'):
                test_set = preprocess_dataset_dr(test_set, config_namespace)

        # Store datasets
        if stage in (None, 'fit'):
            self.train_dataset = train_set
            self.val_dataset = val_set

        if stage in (None, 'test'):
            self.test_dataset = test_set
    
    def train_dataloader(self):
        """Create training DataLoader."""
        # GPU tensors cannot be pinned, so pin_memory is always False for DR
        use_pin_memory = not self.config.dimensionality_reduction.use_dynamic_dataset_gpu

        # Check if node budget batching is enabled
        if self.config.training.use_node_budget_batching:
            from gnn_dr.datasets.node_budget_sampler import create_node_budget_sampler_for_clip_dataset
            
            batch_sampler = create_node_budget_sampler_for_clip_dataset(
                dataset=self.train_dataset,
                max_nodes_per_batch=self.config.training.max_nodes_per_batch,
                max_graphs_per_batch=self.config.training.max_graphs_per_batch,
                min_batch_size=self.config.training.min_batch_size,
                shuffle=True,
            )
            
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=0,  # Geometric data doesn't play well with multiprocessing
                pin_memory=use_pin_memory
            )
        else:
            # Standard fixed batch_size DataLoader
            return DataLoader(
                self.train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                num_workers=0,  # Geometric data doesn't play well with multiprocessing
                pin_memory=use_pin_memory
            )
    
    def val_dataloader(self):
        """Create validation DataLoader."""
        use_pin_memory = not self.config.dimensionality_reduction.use_dynamic_dataset_gpu
        return DataLoader(
            self.val_dataset,
            batch_size=1,  # Validation is done graph-by-graph
            shuffle=False,
            num_workers=0,
            pin_memory=use_pin_memory
        )
    
    def test_dataloader(self):
        """Create test DataLoader."""
        use_pin_memory = not self.config.dimensionality_reduction.use_dynamic_dataset_gpu
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # Test is done graph-by-graph
            shuffle=False,
            num_workers=0,
            pin_memory=use_pin_memory
        )
