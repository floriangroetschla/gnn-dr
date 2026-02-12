"""
PyTorch Lightning callback for multi-size graph evaluation with multi-round testing.

Evaluates DR quality on multiple fixed-size graphs from configurable datasets
with varying GNN round counts. Implements hierarchical W&B logging for easy 
comparison across sizes, rounds, and datasets.

Now integrates tf-projection-qm for comprehensive DR quality metrics across
all graph sizes and datasets.
"""

import time

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
import wandb
import matplotlib.pyplot as plt

from gnn_dr.baselines.metrics import DRMetrics
from gnn_dr.utils.visualization import visualize_graph_layout

# Try to import tf-projection-qm
try:
    from tensorflow_projection_qm.metrics import run_all_metrics as tfpqm_run_all_metrics
    from tensorflow_projection_qm.metrics import _ALL_METRICS
    TFPQM_AVAILABLE = True
    TFPQM_METRIC_NAMES = [m.name for m in _ALL_METRICS]
except ImportError:
    TFPQM_AVAILABLE = False
    TFPQM_METRIC_NAMES = []


class MultiSizeEvaluationCallback(Callback):
    """
    Evaluate DR quality on multiple fixed-size graphs with multi-round testing.
    
    Features:
    - Evaluates on graphs of different sizes (e.g., 100, 500, 1000, 5000, 10000)
    - Tests multiple GNN round counts (e.g., 1, 2, 5, 10, 20, 50)
    - Supports multiple evaluation datasets (configurable via eval_datasets)
    - Computes all DR metrics for each size/round/dataset combination
    - Generates scatter plot visualizations
    - Logs everything to W&B with hierarchical organization
    
    W&B Organization (using {metric}_{dataset}/size_{N} format):
        trustworthiness_mnist/size_1000
        knn_recall_cifar10/size_5000
        trustworthiness_mnist/size_1000_rounds_10
        visualizations/{dataset}/size_{N}/scatter
    
    Evaluation Strategy:
    - Small graphs (≤500): Every epoch
    - Medium graphs (1000-2000): Every 5 epochs
    - Large graphs (≥5000): Every 10 epochs
    - Multi-round eval: Configurable interval (default: every 10 epochs)
    
    Configuration Example (in config YAML):
        multi_size_evaluation:
          enabled: true
          eval_sizes: [100, 500, 1000, 5000, 10000]
          eval_datasets:
            - name: mnist_clip
              label: mnist
            - name: cifar10_clip
              label: cifar10
              sizes: [500, 1000, 10000]  # Override sizes for this dataset
            - name: laion_clip
              label: laion
    """
    
    def __init__(
        self,
        config,
        eval_sizes: List[int] = [100, 500, 1000, 5000, 10000],
        eval_intervals: Optional[Dict[int, int]] = None,
        round_counts: List[int] = [1, 2, 5, 10, 20, 50],
        multi_round_interval: int = 10,
        multi_round_sizes: Optional[List[int]] = None,
        generate_visualizations: bool = True,
        viz_sizes: Optional[List[int]] = None,
        viz_interval: int = 10,
        k_neighbors: int = 15,
        eval_datasets: Optional[List[Dict[str, Any]]] = None,
        eval_use_test_split: bool = True,
    ):
        """
        Initialize multi-size evaluation callback.
        
        Args:
            config: ExperimentConfig with DR settings
            eval_sizes: Default list of graph sizes to evaluate on
            eval_intervals: Dict mapping size -> eval frequency (epochs)
            round_counts: List of GNN round counts to test
            multi_round_interval: Evaluate multi-round every N epochs
            multi_round_sizes: Sizes to test with multi-round (None = all sizes)
            generate_visualizations: Whether to generate scatter plots
            viz_sizes: Sizes to visualize (None = all sizes)
            viz_interval: Generate plots every N epochs
            k_neighbors: Number of neighbors for metrics
            eval_datasets: List of dataset configs, each with 'name', optional 'label', 
                          optional 'sizes' (overrides eval_sizes for this dataset)
            eval_use_test_split: If True, use test split for evaluation (recommended).
                                 If False, use train split. LAION always uses train.
        """
        super().__init__()
        self.config = config
        self.default_eval_sizes = sorted(eval_sizes)
        self.round_counts = sorted(round_counts)
        self.multi_round_interval = multi_round_interval
        self.k_neighbors = k_neighbors
        
        # Default eval intervals: small=1, medium=5, large=10
        if eval_intervals is None:
            self.eval_intervals = {}
            for size in eval_sizes:
                if size <= 500:
                    self.eval_intervals[size] = 1
                elif size <= 2000:
                    self.eval_intervals[size] = 5
                else:
                    self.eval_intervals[size] = 10
        else:
            self.eval_intervals = eval_intervals
        
        # Which sizes to test with multi-round
        self.multi_round_sizes = multi_round_sizes if multi_round_sizes is not None else eval_sizes
        
        # Visualization settings
        self.generate_visualizations = generate_visualizations
        self.viz_sizes = viz_sizes if viz_sizes is not None else eval_sizes
        self.viz_interval = viz_interval
        
        # Metrics computer
        self.metrics_computer = DRMetrics(k_neighbors=k_neighbors)
        
        # Use test split for evaluation (recommended for proper out-of-distribution eval)
        self.eval_use_test_split = eval_use_test_split
        
        # Parse eval_datasets configuration
        self.eval_datasets_config = self._parse_eval_datasets_config(eval_datasets)
        
        # Storage for eval graphs: {dataset_label: {size: graph}}
        self.eval_graphs = {}
        
        # Cache high-dim distances for efficiency: {dataset_label: {size: dists}}
        self.cached_high_dim_dists = {}
        
        # Track metrics history: {dataset_label: {size: {epoch: metrics}}}
        self.metrics_history = {}
    
    def _parse_eval_datasets_config(
        self, 
        eval_datasets: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Parse and validate eval_datasets configuration.
        
        Args:
            eval_datasets: Raw config from YAML, or None for auto-detect
            
        Returns:
            List of parsed dataset configs with 'name', 'label', 'sizes'
        """
        if eval_datasets is None:
            # Auto-detect based on training dataset
            dataset_name = self.config.dataset.dataset
            
            if dataset_name == 'laion_clip':
                # LAION: evaluate on both MNIST (cross-domain) and LAION (in-domain)
                return [
                    {'name': 'mnist_clip', 'label': 'mnist', 'sizes': self.default_eval_sizes},
                    {'name': 'laion_clip', 'label': 'laion', 'sizes': self.default_eval_sizes},
                ]
            elif dataset_name in ('mnist_clip', 'cifar10_clip', 'cifar100_clip', 'fashion_mnist_clip'):
                # Single dataset: evaluate on same dataset
                label = dataset_name.replace('_clip', '')
                return [
                    {'name': dataset_name, 'label': label, 'sizes': self.default_eval_sizes},
                ]
            elif dataset_name == 'multi_clip':
                # Multi-dataset training: default to MNIST evaluation
                return [
                    {'name': 'mnist_clip', 'label': 'mnist', 'sizes': self.default_eval_sizes},
                ]
            else:
                # Non-CLIP dataset, no multi-size eval
                return []
        
        # Parse provided config
        parsed = []
        for ds_config in eval_datasets:
            if isinstance(ds_config, str):
                # Simple string format: just the dataset name
                ds_config = {'name': ds_config}
            
            name = ds_config.get('name')
            if not name:
                warnings.warn(f"Skipping eval dataset config without 'name': {ds_config}")
                continue
            
            # Default label: remove '_clip' suffix
            label = ds_config.get('label', name.replace('_clip', ''))
            
            # Per-dataset sizes or use global default
            sizes = ds_config.get('sizes', self.default_eval_sizes)
            
            # Skip if disabled
            if not ds_config.get('enabled', True):
                continue
            
            parsed.append({
                'name': name,
                'label': label,
                'sizes': sorted(sizes),
            })
        
        return parsed
    
    def setup(self, trainer, pl_module, stage=None):
        """
        Prepare fixed evaluation graphs at the start of training.
        
        Creates fixed subgraphs from each configured dataset for consistent evaluation.
        """
        print(f"\n[Multi-Size Eval DEBUG] setup() called with stage={stage}")
        
        if stage != 'fit':
            print(f"[Multi-Size Eval DEBUG] Skipping setup - stage is not 'fit' (got: {stage})")
            return
        
        # Only setup for DR tasks
        print(f"[Multi-Size Eval DEBUG] Checking if DR task...")
        print(f"[Multi-Size Eval DEBUG]   hasattr(pl_module, 'is_dr_task') = {hasattr(pl_module, 'is_dr_task')}")
        if hasattr(pl_module, 'is_dr_task'):
            print(f"[Multi-Size Eval DEBUG]   pl_module.is_dr_task = {pl_module.is_dr_task}")
        
        if not hasattr(pl_module, 'is_dr_task') or not pl_module.is_dr_task:
            print(f"[Multi-Size Eval DEBUG] Skipping setup - not a DR task")
            return
        
        if not self.eval_datasets_config:
            print(f"[Multi-Size Eval DEBUG] No eval datasets configured, skipping")
            return
        
        print(f"[Multi-Size Eval DEBUG] This is a DR task, proceeding with setup...")
        print(f"[Multi-Size Eval DEBUG] Configured eval datasets: {[d['label'] for d in self.eval_datasets_config]}")
        
        try:
            from gnn_dr.network.preprocessing import preprocess_dataset_dr
            from gnn_dr.config import config_to_namespace
            from gnn_dr.datasets.torchvision_clip import get_registered_clip_dataset
            
            config_namespace = config_to_namespace(self.config)
            device = 'cuda' if self.config.device >= 0 else 'cpu'
            
            # Load each configured dataset
            for ds_config in self.eval_datasets_config:
                name = ds_config['name']
                label = ds_config['label']
                sizes = ds_config['sizes']
                
                print(f"[Multi-Size Eval] Creating {label.upper()} evaluation graphs...")
                
                try:
                    # Get the dataset class
                    dataset_cls = get_registered_clip_dataset(name)
                    
                    # Determine which split to use for evaluation
                    # LAION has no test split, so always use train for LAION
                    if name == 'laion_clip':
                        use_train_split = True
                    else:
                        # Use test split by default (recommended for proper OOD evaluation)
                        use_train_split = not self.eval_use_test_split
                    
                    split_name = "train" if use_train_split else "test"
                    print(f"[Multi-Size Eval] Using {split_name} split for {label}")
                    
                    # Create dataset instance with appropriate parameters
                    ds_params = {
                        'subset_sizes': sizes,
                        'knn_k': self.config.dimensionality_reduction.knn_k,
                        'n_samples_per_size': 1,
                        'device': device,
                        'train': use_train_split,  # Use test split for proper OOD evaluation
                    }
                    
                    # Add dataset-specific parameters
                    if name == 'laion_clip':
                        ds_params['root'] = self.config.dimensionality_reduction.laion_data_dir
                        ds_params['num_chunks'] = self.config.dimensionality_reduction.laion_num_chunks
                    else:
                        ds_params['root'] = self.config.dimensionality_reduction.clip_cache_dir
                        ds_params['clip_model'] = self.config.dimensionality_reduction.clip_model
                    
                    dataset_helper = dataset_cls(**ds_params)
                    
                    # Initialize storage for this dataset
                    self.eval_graphs[label] = {}
                    self.cached_high_dim_dists[label] = {}
                    self.metrics_history[label] = {size: {} for size in sizes}
                    
                    # Create evaluation graphs for each size
                    for size in sizes:
                        subgraph = dataset_helper.get_random_subset_graph(size)
                        preprocessed = preprocess_dataset_dr([subgraph], config_namespace)[0]
                        self.eval_graphs[label][size] = preprocessed
                        
                        # Cache high-dim distances
                        if hasattr(subgraph, 'clip_embedding'):
                            high_dim_np = subgraph.clip_embedding.cpu().numpy()
                            self.cached_high_dim_dists[label][size] = self.metrics_computer._compute_distances(
                                high_dim_np,
                                metric=self.metrics_computer.metric
                            )
                    
                    print(f"[Multi-Size Eval] Prepared {label.upper()} eval graphs: {list(self.eval_graphs[label].keys())}")
                    
                except Exception as e:
                    warnings.warn(f"Failed to create eval graphs for {label}: {e}")
                    continue
                
        except Exception as e:
            warnings.warn(f"Failed to setup multi-size evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Evaluate on multiple graph sizes with optional multi-round testing.
        Evaluates all configured datasets.
        """
        # Only process if this is a DR task
        if not hasattr(pl_module, 'is_dr_task') or not pl_module.is_dr_task:
            return
        
        # Skip during sanity check
        if trainer.sanity_checking:
            return
        
        # Check if any eval graphs are ready
        if len(self.eval_graphs) == 0:
            return
        
        epoch = trainer.current_epoch
        
        # Evaluate each dataset
        for ds_config in self.eval_datasets_config:
            label = ds_config['label']
            sizes = ds_config['sizes']
            
            if label not in self.eval_graphs:
                continue
            
            # Evaluate each size
            for size in sizes:
                # Check if we should evaluate this size this epoch
                eval_interval = self.eval_intervals.get(size, 5)
                if epoch % eval_interval != 0:
                    continue
                
                if size not in self.eval_graphs[label]:
                    continue
                
                # Standard evaluation with fixed rounds
                self._evaluate_size(trainer, pl_module, size, label, fixed_rounds=10)
                
                # Multi-round evaluation (less frequent)
                if size in self.multi_round_sizes and (epoch % self.multi_round_interval == 0):
                    self._evaluate_multi_round(trainer, pl_module, size, label)
                
                # Visualization (less frequent)
                if (self.generate_visualizations and 
                    size in self.viz_sizes and 
                    (epoch % self.viz_interval == 0)):
                    self._generate_visualization(trainer, pl_module, size, label)
    
    def _time_forward_pass(
        self, 
        pl_module, 
        graph, 
        rounds: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Time the forward pass and return predictions + inference time.
        
        Args:
            pl_module: Lightning module
            graph: Graph to run inference on
            rounds: Number of GNN rounds
            
        Returns:
            Tuple of (predictions, inference_time_ms)
        """
        device = pl_module.device
        
        # Sync before timing to ensure any pending GPU ops complete
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            predictions, _ = pl_module(graph, rounds, return_layers=True, encode=True)
        
        # Sync after to ensure forward pass is complete
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        return predictions, elapsed_ms
    
    def _evaluate_size(
        self, 
        trainer, 
        pl_module, 
        size: int, 
        dataset_label: str, 
        fixed_rounds: int = 10
    ):
        """
        Evaluate a single graph size with fixed number of rounds.
        
        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            size: Graph size to evaluate
            dataset_label: Dataset label (e.g., 'mnist', 'cifar10')
            fixed_rounds: Number of GNN rounds to use
        """
        try:
            graph = self.eval_graphs[dataset_label][size].to(pl_module.device)
            
            # Run model with fixed rounds and measure inference time
            predictions, inference_time_ms = self._time_forward_pass(pl_module, graph, fixed_rounds)
            
            # Compute metrics
            metrics = self._compute_metrics(graph, predictions, size, dataset_label)
            
            # Add inference time to metrics
            metrics['inference_time_ms'] = inference_time_ms
            
            # Log to W&B with format: {metric}_{dataset}/size_{N}
            if trainer.logger is not None:
                log_dict = {f'{k}_{dataset_label}/size_{size}': v for k, v in metrics.items()}
                trainer.logger.log_metrics(log_dict, step=trainer.global_step)
            
            # Store metrics
            self.metrics_history[dataset_label][size][trainer.current_epoch] = metrics
            
            if trainer.is_global_zero and self.config.verbose:
                print(f"[Eval {dataset_label.upper()} Size {size}] Trust: {metrics.get('trustworthiness', 0):.3f}, k-NN: {metrics.get('knn_recall', 0):.3f}, Time: {inference_time_ms:.1f}ms")
                
        except Exception as e:
            warnings.warn(f"Error evaluating {dataset_label} size {size}: {e}")
    
    def _evaluate_multi_round(
        self, 
        trainer, 
        pl_module, 
        size: int, 
        dataset_label: str
    ):
        """
        Evaluate a single graph size with multiple round counts.
        
        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            size: Graph size to evaluate
            dataset_label: Dataset label
        """
        try:
            graph = self.eval_graphs[dataset_label][size].to(pl_module.device)
            
            for rounds in self.round_counts:
                # Run model with specific round count and measure inference time
                predictions, inference_time_ms = self._time_forward_pass(pl_module, graph, rounds)
                
                # Compute metrics
                metrics = self._compute_metrics(graph, predictions, size, dataset_label, use_cached_dists=True)
                
                # Add inference time to metrics
                metrics['inference_time_ms'] = inference_time_ms
                
                # Log to W&B with format: {metric}_{dataset}/size_{N}_rounds_{R}
                if trainer.logger is not None:
                    log_dict = {f'{k}_{dataset_label}/size_{size}_rounds_{rounds}': v for k, v in metrics.items()}
                    trainer.logger.log_metrics(log_dict, step=trainer.global_step)
            
            if trainer.is_global_zero and self.config.verbose:
                print(f"[Multi-Round {dataset_label.upper()} Size {size}] Evaluated rounds: {self.round_counts}")
                
        except Exception as e:
            warnings.warn(f"Error in multi-round evaluation for {dataset_label} size {size}: {e}")
    
    def _compute_metrics(
        self, 
        graph, 
        predictions, 
        size: int, 
        dataset_label: str, 
        use_cached_dists: bool = True
    ) -> Dict[str, float]:
        """
        Compute all DR metrics for a graph, including tf-projection-qm metrics.
        
        Args:
            graph: Graph data
            predictions: Model predictions
            size: Graph size (for cache lookup)
            dataset_label: Dataset label (for cache lookup)
            use_cached_dists: Whether to use cached high-dim distances
            
        Returns:
            Dictionary of metrics
        """
        pred_np = predictions.detach().cpu().numpy()
        
        # Get high-dim embeddings
        if hasattr(graph, 'clip_embedding'):
            high_dim_np = graph.clip_embedding.detach().cpu().numpy()
        elif hasattr(graph, 'x'):
            high_dim_np = graph.x.detach().cpu().numpy()
        else:
            return {}
        
        # Get labels if available
        labels_np = None
        if hasattr(graph, 'y') and graph.y is not None:
            labels_np = graph.y.detach().cpu().numpy()
        
        metrics = {}
        
        try:
            # Compute legacy metrics
            metrics['trustworthiness'] = self.metrics_computer.trustworthiness(high_dim_np, pred_np)
            metrics['continuity'] = self.metrics_computer.continuity(high_dim_np, pred_np)
            metrics['knn_recall'] = self.metrics_computer.knn_recall(high_dim_np, pred_np)
            metrics['distance_correlation'] = self.metrics_computer.distance_correlation(high_dim_np, pred_np)
            
            # Silhouette score if labels available
            if labels_np is not None:
                metrics['silhouette_score'] = self.metrics_computer.silhouette(pred_np, labels_np)
            
            # Embedding statistics
            norms = np.linalg.norm(pred_np, axis=1)
            metrics['embedding_mean_norm'] = float(np.mean(norms))
            metrics['embedding_std_norm'] = float(np.std(norms))
            
        except Exception as e:
            warnings.warn(f"Error computing legacy metrics: {e}")
        
        # Compute tf-projection-qm metrics if available and enabled
        tfpqm_metrics = self._compute_tfpqm_metrics(high_dim_np, pred_np, labels_np)
        if tfpqm_metrics:
            # Add with 'tfpqm_' prefix to distinguish from legacy metrics
            for name, value in tfpqm_metrics.items():
                metrics[f'tfpqm_{name}'] = value
        
        return metrics
    
    def _compute_tfpqm_metrics(
        self,
        high_dim_np: np.ndarray,
        pred_np: np.ndarray,
        labels_np: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute tf-projection-qm metrics for comprehensive DR quality assessment.
        
        Uses the tf-projection-qm package which provides many standard DR evaluation metrics.
        See: https://github.com/mespadoto/proj-quant-eval
        
        Args:
            high_dim_np: High-dimensional embeddings (N, D) as numpy array
            pred_np: Low-dimensional projections (N, d) as numpy array
            labels_np: Optional class labels as numpy array
            
        Returns:
            Dictionary with metric names and values
        """
        if not TFPQM_AVAILABLE:
            return {}
        
        # Check if tf-projection-qm is enabled in config
        logging_config = getattr(self.config, 'logging', None)
        if logging_config is None:
            # Default: enabled
            tfpqm_enabled = True
            k = self.k_neighbors
            n_classes = None
            selected_metrics = None
        else:
            tfpqm_enabled = getattr(logging_config, 'tfpqm_enabled', True)
            k = getattr(logging_config, 'tfpqm_k', self.k_neighbors)
            n_classes = getattr(logging_config, 'tfpqm_n_classes', None)
            selected_metrics = getattr(logging_config, 'tfpqm_selected_metrics', None)
        
        if not tfpqm_enabled:
            return {}
        
        try:
            # Auto-detect n_classes if not specified
            if n_classes is None and labels_np is not None:
                n_classes = len(np.unique(labels_np))
            
            # Run all metrics using tf-projection-qm
            # The function accepts numpy arrays and handles the TensorFlow conversion internally
            all_metrics = tfpqm_run_all_metrics(
                X=high_dim_np,
                X_2d=pred_np,
                y=labels_np,
                k=k,
                n_classes=n_classes,
                as_numpy=True  # Return numpy values instead of TF tensors
            )
            
            # Filter metrics if specific ones are selected
            if selected_metrics is not None and len(selected_metrics) > 0:
                all_metrics = {
                    name: value 
                    for name, value in all_metrics.items() 
                    if name in selected_metrics
                }
            
            # Convert to float and return
            result = {}
            for name, value in all_metrics.items():
                try:
                    # Handle various numpy types
                    if isinstance(value, np.ndarray):
                        if value.size == 1:
                            result[name] = float(value.item())
                        else:
                            # For array metrics, take mean
                            result[name] = float(np.mean(value))
                    else:
                        result[name] = float(value)
                except (TypeError, ValueError) as e:
                    warnings.warn(f"Could not convert tf-projection-qm metric {name}: {e}")
                    continue
            
            return result
            
        except Exception as e:
            warnings.warn(f"Error computing tf-projection-qm metrics: {e}")
            return {}
    
    def _generate_visualization(
        self, 
        trainer, 
        pl_module, 
        size: int, 
        dataset_label: str
    ):
        """
        Generate scatter plot visualization for a specific size.
        
        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            size: Graph size to visualize
            dataset_label: Dataset label
        """
        try:
            graph = self.eval_graphs[dataset_label][size].to(pl_module.device)
            
            # Run model
            with torch.no_grad():
                predictions, _ = pl_module(graph, 10, return_layers=True, encode=True)
            
            # Get metrics for title
            metrics = self.metrics_history[dataset_label][size].get(trainer.current_epoch, {})
            title_parts = [f'{dataset_label.upper()} Size {size}', f'Epoch {trainer.current_epoch}']
            if 'trustworthiness' in metrics:
                title_parts.append(f'Trust: {metrics["trustworthiness"]:.3f}')
            title = ' | '.join(title_parts)
            
            # Use existing visualization function
            fig = visualize_graph_layout(
                graph,
                predictions,
                title=title,
                config=self.config.validation if hasattr(self.config, 'validation') else None
            )
            
            if fig is not None and trainer.logger is not None:
                trainer.logger.experiment.log({
                    f"visualizations/{dataset_label}/size_{size}/scatter": wandb.Image(fig),
                    "epoch": trainer.current_epoch
                })
                plt.close(fig)
                
        except Exception as e:
            warnings.warn(f"Error generating visualization for {dataset_label} size {size}: {e}")
    
    def on_train_end(self, trainer, pl_module):
        """
        Log summary metrics at end of training.
        """
        if not hasattr(pl_module, 'is_dr_task') or not pl_module.is_dr_task:
            return
        
        try:
            # Compute summary statistics across sizes for each dataset
            all_summary = {}
            
            for ds_config in self.eval_datasets_config:
                label = ds_config['label']
                sizes = ds_config['sizes']
                
                if label not in self.metrics_history:
                    continue
                
                final_metrics = {}
                
                for size in sizes:
                    if size not in self.metrics_history[label]:
                        continue
                    if len(self.metrics_history[label][size]) == 0:
                        continue
                    
                    # Get final epoch metrics
                    epochs = sorted(self.metrics_history[label][size].keys())
                    if len(epochs) > 0:
                        final_epoch_metrics = self.metrics_history[label][size][epochs[-1]]
                        for metric_name, value in final_epoch_metrics.items():
                            if metric_name not in final_metrics:
                                final_metrics[metric_name] = []
                            final_metrics[metric_name].append(value)
                
                # Average metrics across sizes for this dataset
                if final_metrics:
                    for metric_name, values in final_metrics.items():
                        all_summary[f'eval_summary_{label}/avg_{metric_name}'] = np.mean(values)
                        all_summary[f'eval_summary_{label}/std_{metric_name}'] = np.std(values)
            
            # Log to W&B
            if trainer.logger is not None and len(all_summary) > 0:
                trainer.logger.log_metrics(all_summary)
                
                if trainer.is_global_zero and self.config.verbose:
                    print("\n" + "="*70)
                    print("Multi-Size Evaluation Summary")
                    print("="*70)
                    for key, value in sorted(all_summary.items()):
                        print(f"  {key:50s}: {value:.4f}")
                    print("="*70 + "\n")
                    
        except Exception as e:
            warnings.warn(f"Error computing summary metrics: {e}")
