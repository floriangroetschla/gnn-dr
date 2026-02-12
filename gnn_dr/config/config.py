"""
Configuration dataclasses for GNN-DR.

This module defines structured configuration using Python dataclasses,
compatible with Hydra for YAML-based configuration management.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Architecture
    conv: Literal['gru', 'gat', 'gin'] = 'gru'
    hidden_dimension: int = 64
    hidden_state_factor: float = 4.0
    mlp_depth: int = 2
    out_dim: int = 2
    
    # Features
    use_beacons: bool = True
    num_beacons: int = 2
    encoding_size_per_beacon: int = 8
    laplace_eigvec: int = 8
    random_projection_dim: int = 0  # Gaussian Random Projection dimensions (0 = disabled)
    pca_dim: int = 0  # PCA dimensions (0 = disabled) - per-graph PCA on CLIP embeddings
    random_in_channels: int = 1
    clip_in_channels: int = 0  # CLIP embedding dimensions (0 = disabled)
    
    # Network operations
    aggregation: Literal['add', 'mean', 'max'] = 'add'
    normalization: Literal['LayerNorm', 'BatchNorm', 'None'] = 'LayerNorm'
    dropout: float = 0.0
    
    # Skip connections
    skip_previous: bool = False
    skip_input: bool = False
    
    # Rewiring
    rewiring: Literal['knn', 'radius', 'neg_sample', 'none'] = 'knn'
    knn_k: int = 8
    alt_freq: int = 2
    
    # Negative sampling rewiring
    neg_sample_multiplier: float = 1.0
    neg_sample_force_undirected: bool = True
    
    # Precomputed rewiring (for reducing inference overhead)
    # When True, rewired edges are computed once during preprocessing and stored in data object
    # Supported for: 'neg_sample' rewiring strategy
    rewiring_precompute: bool = False
    
    # Edge attributes (UMAP weights in GNN)
    use_edge_attr: bool = True  # Whether to use edge weights in GNN message passing


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    
    # Optimization
    lr: float = 0.0002
    weight_decay: float = 0.0
    epochs: int = 200
    batch_size: int = 32
    
    # Iteration control
    iter_mean: float = 5.0
    iter_var: float = 1.0
    
    # Scheduler
    scheduler: Literal['Plateau', 'CosineAnnealing', 'None'] = 'Plateau'
    
    # Loss
    use_entropy_loss: bool = False
    use_l1: bool = False
    l1_weight: float = 0.0
    
    # Reproducibility
    run_number: int = 1
    randomize_between_epochs: bool = True
    
    # Dynamic batching by node budget (for variable-sized graphs)
    # When enabled, batches are formed by node count rather than fixed graph count
    use_node_budget_batching: bool = False  # Disabled by default
    max_nodes_per_batch: int = 10000  # Maximum total nodes in a batch
    max_graphs_per_batch: Optional[int] = None  # Maximum graphs per batch (None = no limit)
    min_batch_size: int = 1  # Minimum graphs per batch (prevents batch_size=0)


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    
    dataset: Literal[
        'mnist', 'cifar10',
        'mnist_clip', 'cifar10_clip', 'cifar100_clip', 'fashion_mnist_clip',
        'laion_clip', 'multi_clip'
    ] = 'mnist_clip'
    
    # Multi-dataset training configuration (for multi_clip dataset)
    # List of datasets with sampling weights, e.g.:
    # [{'name': 'mnist_clip', 'weight': 1.0}, {'name': 'laion_clip', 'weight': 1.0}]
    train_datasets: Optional[list] = None


@dataclass
class CoarseningConfig:
    """Graph coarsening configuration."""
    
    coarsen: bool = False
    coarsen_prob: float = 0.5
    coarsen_noise: float = 0.01
    coarsen_k: int = 5
    coarsen_r: float = 0.8
    coarsen_algo: Literal['heavy_edge', 'variation_edges', 'variation_cliques'] = 'heavy_edge'
    coarsen_min_size: int = 50


@dataclass
class ReplayBufferConfig:
    """Replay buffer configuration."""
    
    use_replay_buffer: bool = True
    replay_buffer_size: int = 4096
    replay_train_replacement_prob: float = 0.5
    replay_buffer_replacement_prob: float = 1.0
    num_replay_batches: int = 8
    use_gpu_replay_buffer: bool = True  # Keep replay buffer on GPU (faster, uses more GPU memory)
    
    # Gradient accumulation mode
    # When False (default): Each fresh/replay batch does its own backward + optimizer.step()
    # When True: Gradients are accumulated across fresh + all replay batches, single optimizer.step() at end
    accumulate_replay_gradients: bool = False


@dataclass
class ValidationConfig:
    """Validation and visualization configuration."""
    
    # Evaluation
    num_layers: int = 10  # Number of layers for validation/test
    
    # Visualization
    visualize: bool = False  # Enable graph visualization
    num_graphs_to_visualize: int = 3  # How many validation graphs to visualize
    visualization_interval: int = 5  # Visualize every N epochs
    
    # Visualization style
    node_size: int = 50
    edge_width: float = 0.5
    figure_size: int = 10


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    
    # Logging level
    log_level: Literal['off', 'basic', 'detailed', 'full'] = 'basic'
    
    # What to log
    log_replay_buffer: bool = True
    log_loss_components: bool = True
    log_gradients: bool = False
    log_dataset_stats: bool = False
    log_validation_per_graph: bool = False
    
    # Frequency
    log_every_n_steps: int = 50  # Only log detailed metrics every N steps
    
    # Verbosity
    log_to_console: bool = True
    
    # DR-specific metrics logging
    dr_metrics_interval: int = 1  # Compute DR metrics every N epochs
    dr_compute_trustworthiness: bool = True
    dr_compute_continuity: bool = True
    dr_compute_knn_recall: bool = True
    dr_compute_distance_correlation: bool = True
    dr_compute_silhouette: bool = True
    dr_compute_umap_loss: bool = False  # Expensive, usually disabled
    dr_generate_scatter_plots: bool = True
    dr_scatter_plot_interval: int = 5
    dr_track_embedding_stats: bool = True
    
    # Batch size logging (useful for dynamic batching)
    log_batch_graph_sizes: bool = True  # Log graph size statistics per batch to WandB
    
    # tf-projection-qm metrics (comprehensive DR quality metrics)
    # See: https://github.com/mespadoto/proj-quant-eval
    tfpqm_enabled: bool = True  # Enable tf-projection-qm metrics
    tfpqm_k: int = 15  # k neighbors for local structure metrics
    tfpqm_n_classes: Optional[int] = None  # Number of classes (auto-detected if None)
    tfpqm_selected_metrics: Optional[list] = None  # None = all metrics, or list of metric names
    # Available metrics: Trustworthiness, Continuity, ClassAwareTrustworthiness, ClassAwareContinuity,
    # NormalizedStress, ScaleNormalizedStress, Jaccard, NeighborhoodHit, FalseNeighbors, TrueNeighbors,
    # MRREData, MRREProj, PearsonCorrelation, ShepardGoodness, AverageLocalError, DistanceConsistency, Procrustes


@dataclass
class EvalDatasetConfig:
    """Configuration for a single evaluation dataset."""
    
    name: str = 'mnist_clip'  # Dataset name (e.g., 'mnist_clip', 'cifar10_clip', 'laion_clip')
    label: Optional[str] = None  # Short label for logging (defaults to name without '_clip')
    sizes: Optional[list] = None  # Override eval_sizes for this dataset (None = use global)
    enabled: bool = True  # Whether to evaluate on this dataset


@dataclass
class MultiSizeEvaluationConfig:
    """Multi-size evaluation configuration for DR tasks."""
    
    enabled: bool = False
    eval_sizes: list = field(default_factory=lambda: [100, 500, 1000, 5000, 10000])
    eval_intervals: Optional[dict] = None  # Can be None for auto-defaults
    round_counts: list = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])
    multi_round_interval: int = 10
    multi_round_sizes: Optional[list] = None  # None = all sizes
    generate_scatter_plots: bool = True
    viz_sizes: Optional[list] = None  # None = all sizes
    viz_interval: int = 10
    
    # Use test split for evaluation (recommended for proper OOD evaluation)
    # If True, metrics are computed on held-out test data (e.g., MNIST's 10k test images)
    # If False, metrics are computed on training data (not recommended)
    # Note: LAION always uses train split as it has no labeled test set
    eval_use_test_split: bool = True
    
    # Evaluation datasets configuration
    # List of datasets to evaluate on, each with optional per-dataset sizes
    # Example: [{'name': 'mnist_clip', 'label': 'mnist'}, {'name': 'cifar10_clip'}]
    eval_datasets: Optional[list] = None  # None = auto-detect from training dataset


@dataclass
class DimensionalityReductionConfig:
    """Dimensionality reduction task configuration."""
    
    # Task mode
    enabled: bool = False
    
    # CLIP configuration
    clip_model: str = 'openai/CLIP-vit-base-patch32'
    clip_cache_dir: str = 'data/MNIST'
    normalize_clip_embeddings: bool = True  # Normalize embeddings to unit length when loading
    
    # Graph construction
    knn_k: int = 15
    subset_sizes: list = field(default_factory=lambda: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    val_test_subset_size: int = None  # Legacy: If None, use full dataset; otherwise random subset of this size
    
    # Validation/test dataset configuration (new, more flexible)
    val_dataset: Optional[str] = None  # Validation dataset name (None = same as training)
    test_dataset: Optional[str] = None  # Test dataset name (None = same as training)
    val_subset_size: Optional[int] = None  # Size of validation subset (None = full dataset)
    test_subset_size: Optional[int] = None  # Size of test subset (None = full dataset)
    
    # Loss configuration - deprecated (kept for backward compatibility)
    umap_neg_sample_rate: float = 5.0
    umap_repulsion_weight: float = 1.0
    umap_use_efficient: bool = True
    
    # New UMAP loss type and parameters
    umap_loss_type: str = 'umap_efficient'  # 'umap' or 'umap_efficient'
    umap_edge_sample_rate: float = 1.0
    umap_min_dist: float = 0.1
    umap_spread: float = 1.0
    umap_a: Optional[float] = None
    umap_b: Optional[float] = None
    
    # UMAP sampling parameters (matching UMAP's actual optimization)
    umap_num_positive_samples: int = 1000  # Number of positive edges to sample per step
    umap_num_negatives_per_edge: int = 5  # Fixed negatives per positive edge
    umap_gauge_fix: bool = False  # Disabled by default - distorts distances
    umap_embedding_reg_weight: float = 0.0  # Legacy - ignored
    
    # UMAP loss weighting strategy
    # False (default): Sample edges proportionally to weights, take simple mean (sampling-based approximation)
    # True: Sample edges uniformly, multiply loss by edge weights (closer to standard UMAP objective)
    umap_use_weighted_loss: bool = False

    # Edge weight method: 'umap' (default, fuzzy simplicial set) or 'tsne' (perplexity-based)
    edge_weight_method: str = 'umap'
    # t-SNE perplexity (only used when edge_weight_method='tsne')
    # Must be less than knn_k (max achievable perplexity = knn_k)
    tsne_perplexity: float = 10.0

    # Dynamic dataset settings
    use_dynamic_dataset: bool = True
    n_samples_per_size: int = 10
    
    # GPU-optimized dynamic dataset (new)
    use_dynamic_dataset_gpu: bool = False
    
    # LAION-400M specific settings
    laion_data_dir: str = 'data/laion_embeddings'  # Directory to cache LAION embeddings
    laion_num_chunks: int = 10  # Number of 1M-embedding chunks to load (0-410)
    laion_subset_sizes: list = field(default_factory=lambda: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000])


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    coarsening: CoarseningConfig = field(default_factory=CoarseningConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dimensionality_reduction: DimensionalityReductionConfig = field(default_factory=DimensionalityReductionConfig)
    multi_size_evaluation: MultiSizeEvaluationConfig = field(default_factory=MultiSizeEvaluationConfig)
    
    # System - Single GPU (legacy)
    device: int = 0  # Single GPU device ID (used when devices is None)
    verbose: bool = True
    use_cupy: bool = False
    
    # Multi-GPU configuration
    devices: Optional[list] = None  # List of GPU IDs, e.g., [0,1,2,3]. None = use single 'device'
    strategy: str = 'auto'  # 'auto', 'ddp', 'ddp_spawn', 'fsdp', etc.
    num_nodes: int = 1  # Number of nodes for multi-node training
    
    # Experiment tracking
    model_name: str = 'GNN-DR'
    wandb_project_name: str = 'GNN-DR'
    store_models: bool = True
    
    def to_flat_dict(self):
        """
        Convert nested config to flat dictionary for backward compatibility.
        
        Automatically iterates through all dataclass fields to ensure
        no config sections are accidentally missed. This is future-proof:
        any new config sections added will automatically be included.
        
        Returns:
            dict: Flat dictionary with all config values
        """
        from dataclasses import fields
        
        flat = {}
        
        # Iterate through all fields in the ExperimentConfig dataclass
        for field in fields(self):
            field_value = getattr(self, field.name)
            
            # If it's a dataclass (nested config), flatten it
            if hasattr(field_value, '__dataclass_fields__'):
                for nested_field in fields(field_value):
                    nested_value = getattr(field_value, nested_field.name)
                    flat[nested_field.name] = nested_value
            # Otherwise, it's a top-level field
            else:
                flat[field.name] = field_value
        
        return flat
    
    @classmethod
    def from_flat_dict(cls, flat_dict):
        """
        Create ExperimentConfig from flat dictionary (backward compatibility).
        
        Args:
            flat_dict: Flat dictionary with all config values
            
        Returns:
            ExperimentConfig: Structured configuration object
        """
        # Extract model config
        model_config = ModelConfig(
            conv=flat_dict.get('conv', 'gru'),
            hidden_dimension=flat_dict.get('hidden_dimension', 64),
            hidden_state_factor=flat_dict.get('hidden_state_factor', 4.0),
            mlp_depth=flat_dict.get('mlp_depth', 2),
            out_dim=flat_dict.get('out_dim', 2),
            use_beacons=flat_dict.get('use_beacons', True),
            num_beacons=flat_dict.get('num_beacons', 2),
            encoding_size_per_beacon=flat_dict.get('encoding_size_per_beacon', 8),
            laplace_eigvec=flat_dict.get('laplace_eigvec', 8),
            random_projection_dim=flat_dict.get('random_projection_dim', 0),
            pca_dim=flat_dict.get('pca_dim', 0),
            random_in_channels=flat_dict.get('random_in_channels', 1),
            clip_in_channels=flat_dict.get('clip_in_channels', 0),
            aggregation=flat_dict.get('aggregation', 'add'),
            normalization=flat_dict.get('normalization', 'LayerNorm'),
            dropout=flat_dict.get('dropout', 0.0),
            skip_previous=flat_dict.get('skip_previous', False),
            skip_input=flat_dict.get('skip_input', False),
            rewiring=flat_dict.get('rewiring', 'knn'),
            knn_k=flat_dict.get('knn_k', 8),
            alt_freq=flat_dict.get('alt_freq', 2),
            neg_sample_multiplier=flat_dict.get('neg_sample_multiplier', 1.0),
            neg_sample_force_undirected=flat_dict.get('neg_sample_force_undirected', True),
            rewiring_precompute=flat_dict.get('rewiring_precompute', False),
            use_edge_attr=flat_dict.get('use_edge_attr', True),
        )
        
        # Extract training config
        training_config = TrainingConfig(
            lr=flat_dict.get('lr', 0.0002),
            weight_decay=flat_dict.get('weight_decay', 0.0),
            epochs=flat_dict.get('epochs', 200),
            batch_size=flat_dict.get('batch_size', 32),
            iter_mean=flat_dict.get('iter_mean', 5.0),
            iter_var=flat_dict.get('iter_var', 1.0),
            scheduler=flat_dict.get('scheduler', 'Plateau'),
            use_entropy_loss=flat_dict.get('use_entropy_loss', False),
            use_l1=flat_dict.get('use_l1', False),
            l1_weight=flat_dict.get('l1_weight', 0.0),
            run_number=flat_dict.get('run_number', 1),
            randomize_between_epochs=flat_dict.get('randomize_between_epochs', True),
            use_node_budget_batching=flat_dict.get('use_node_budget_batching', False),
            max_nodes_per_batch=flat_dict.get('max_nodes_per_batch', 10000),
            max_graphs_per_batch=flat_dict.get('max_graphs_per_batch', None),
            min_batch_size=flat_dict.get('min_batch_size', 1),
        )
        
        # Extract dataset config
        dataset_config = DatasetConfig(
            dataset=flat_dict.get('dataset', 'mnist_clip'),
            train_datasets=flat_dict.get('train_datasets', None),
        )
        
        # Extract coarsening config
        coarsening_config = CoarseningConfig(
            coarsen=flat_dict.get('coarsen', False),
            coarsen_prob=flat_dict.get('coarsen_prob', 0.5),
            coarsen_noise=flat_dict.get('coarsen_noise', 0.01),
            coarsen_k=flat_dict.get('coarsen_k', 5),
            coarsen_r=flat_dict.get('coarsen_r', 0.8),
            coarsen_algo=flat_dict.get('coarsen_algo', 'heavy_edge'),
            coarsen_min_size=flat_dict.get('coarsen_min_size', 50),
        )
        
        # Extract replay buffer config
        replay_buffer_config = ReplayBufferConfig(
            use_replay_buffer=flat_dict.get('use_replay_buffer', True),
            replay_buffer_size=flat_dict.get('replay_buffer_size', 4096),
            replay_train_replacement_prob=flat_dict.get('replay_train_replacement_prob', 0.5),
            replay_buffer_replacement_prob=flat_dict.get('replay_buffer_replacement_prob', 1.0),
            num_replay_batches=flat_dict.get('num_replay_batches', 8),
            use_gpu_replay_buffer=flat_dict.get('use_gpu_replay_buffer', True),
            accumulate_replay_gradients=flat_dict.get('accumulate_replay_gradients', False),
        )
        
        # Extract validation config
        validation_config = ValidationConfig(
            num_layers=flat_dict.get('num_layers', 10),
            visualize=flat_dict.get('visualize', False),
            num_graphs_to_visualize=flat_dict.get('num_graphs_to_visualize', 3),
            visualization_interval=flat_dict.get('visualization_interval', 5),
            node_size=flat_dict.get('node_size', 50),
            edge_width=flat_dict.get('edge_width', 0.5),
            figure_size=flat_dict.get('figure_size', 10),
        )
        
        # Extract logging config
        logging_config = LoggingConfig(
            log_level=flat_dict.get('log_level', 'basic'),
            log_replay_buffer=flat_dict.get('log_replay_buffer', True),
            log_loss_components=flat_dict.get('log_loss_components', True),
            log_gradients=flat_dict.get('log_gradients', False),
            log_dataset_stats=flat_dict.get('log_dataset_stats', False),
            log_validation_per_graph=flat_dict.get('log_validation_per_graph', False),
            log_every_n_steps=flat_dict.get('log_every_n_steps', 50),
            log_to_console=flat_dict.get('log_to_console', True),
            dr_metrics_interval=flat_dict.get('dr_metrics_interval', 1),
            dr_compute_trustworthiness=flat_dict.get('dr_compute_trustworthiness', True),
            dr_compute_continuity=flat_dict.get('dr_compute_continuity', True),
            dr_compute_knn_recall=flat_dict.get('dr_compute_knn_recall', True),
            dr_compute_distance_correlation=flat_dict.get('dr_compute_distance_correlation', True),
            dr_compute_silhouette=flat_dict.get('dr_compute_silhouette', True),
            dr_compute_umap_loss=flat_dict.get('dr_compute_umap_loss', False),
            dr_generate_scatter_plots=flat_dict.get('dr_generate_scatter_plots', True),
            dr_scatter_plot_interval=flat_dict.get('dr_scatter_plot_interval', 5),
            dr_track_embedding_stats=flat_dict.get('dr_track_embedding_stats', True),
            log_batch_graph_sizes=flat_dict.get('log_batch_graph_sizes', True),
            tfpqm_enabled=flat_dict.get('tfpqm_enabled', True),
            tfpqm_k=flat_dict.get('tfpqm_k', 15),
            tfpqm_n_classes=flat_dict.get('tfpqm_n_classes', None),
            tfpqm_selected_metrics=flat_dict.get('tfpqm_selected_metrics', None),
        )
        
        # Extract dimensionality reduction config
        dr_config = DimensionalityReductionConfig(
            enabled=flat_dict.get('enabled', False),
            clip_model=flat_dict.get('clip_model', 'openai/CLIP-vit-base-patch32'),
            clip_cache_dir=flat_dict.get('clip_cache_dir', 'data/MNIST'),
            normalize_clip_embeddings=flat_dict.get('normalize_clip_embeddings', True),
            knn_k=flat_dict.get('knn_k', 15),
            subset_sizes=flat_dict.get('subset_sizes', [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]),
            val_test_subset_size=flat_dict.get('val_test_subset_size', None),
            val_dataset=flat_dict.get('val_dataset', None),
            test_dataset=flat_dict.get('test_dataset', None),
            val_subset_size=flat_dict.get('val_subset_size', None),
            test_subset_size=flat_dict.get('test_subset_size', None),
            umap_neg_sample_rate=flat_dict.get('umap_neg_sample_rate', 5.0),
            umap_repulsion_weight=flat_dict.get('umap_repulsion_weight', 1.0),
            umap_use_efficient=flat_dict.get('umap_use_efficient', True),
            umap_loss_type=flat_dict.get('umap_loss_type', 'umap_efficient'),
            umap_edge_sample_rate=flat_dict.get('umap_edge_sample_rate', 1.0),
            umap_min_dist=flat_dict.get('umap_min_dist', 0.1),
            umap_spread=flat_dict.get('umap_spread', 1.0),
            umap_a=flat_dict.get('umap_a', None),
            umap_b=flat_dict.get('umap_b', None),
            umap_num_positive_samples=flat_dict.get('umap_num_positive_samples', 1000),
            umap_num_negatives_per_edge=flat_dict.get('umap_num_negatives_per_edge', 5),
            umap_gauge_fix=flat_dict.get('umap_gauge_fix', False),
            umap_embedding_reg_weight=flat_dict.get('umap_embedding_reg_weight', 0.0),
            umap_use_weighted_loss=flat_dict.get('umap_use_weighted_loss', False),
            edge_weight_method=flat_dict.get('edge_weight_method', 'umap'),
            tsne_perplexity=flat_dict.get('tsne_perplexity', 10.0),
            use_dynamic_dataset=flat_dict.get('use_dynamic_dataset', True),
            n_samples_per_size=flat_dict.get('n_samples_per_size', 10),
            use_dynamic_dataset_gpu=flat_dict.get('use_dynamic_dataset_gpu', False),
            laion_data_dir=flat_dict.get('laion_data_dir', 'data/laion_embeddings'),
            laion_num_chunks=flat_dict.get('laion_num_chunks', 10),
            laion_subset_sizes=flat_dict.get('laion_subset_sizes', [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]),
        )
        
        # Extract multi-size evaluation config
        multi_size_eval_config = MultiSizeEvaluationConfig(
            enabled=flat_dict.get('enabled', False),
            eval_sizes=flat_dict.get('eval_sizes', [100, 500, 1000, 5000, 10000]),
            eval_intervals=flat_dict.get('eval_intervals', None),
            round_counts=flat_dict.get('round_counts', [1, 2, 5, 10, 20, 50]),
            multi_round_interval=flat_dict.get('multi_round_interval', 10),
            multi_round_sizes=flat_dict.get('multi_round_sizes', None),
            generate_scatter_plots=flat_dict.get('generate_scatter_plots', True),
            viz_sizes=flat_dict.get('viz_sizes', None),
            viz_interval=flat_dict.get('viz_interval', 10),
            eval_use_test_split=flat_dict.get('eval_use_test_split', True),
            eval_datasets=flat_dict.get('eval_datasets', None),
        )
        
        # Create experiment config
        return cls(
            model=model_config,
            training=training_config,
            dataset=dataset_config,
            coarsening=coarsening_config,
            replay_buffer=replay_buffer_config,
            validation=validation_config,
            logging=logging_config,
            dimensionality_reduction=dr_config,
            multi_size_evaluation=multi_size_eval_config,
            device=flat_dict.get('device', 0),
            verbose=flat_dict.get('verbose', True),
            use_cupy=flat_dict.get('use_cupy', False),
            devices=flat_dict.get('devices', None),
            strategy=flat_dict.get('strategy', 'auto'),
            num_nodes=flat_dict.get('num_nodes', 1),
            model_name=flat_dict.get('model_name', 'GNN-DR'),
            wandb_project_name=flat_dict.get('wandb_project_name', 'GNN-DR'),
            store_models=flat_dict.get('store_models', True),
        )
