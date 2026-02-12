"""
PyTorch Lightning callback for dimensionality reduction evaluation metrics.

Logs UMAP loss, trustworthiness, continuity, and other DR quality metrics during validation.
Enhanced with scatter plot generation, embedding statistics, and configurable options.

Now integrates tf-projection-qm for comprehensive DR quality metrics:
- Trustworthiness, Continuity (and class-aware versions)
- NormalizedStress, ScaleNormalizedStress
- Jaccard, NeighborhoodHit
- FalseNeighbors, TrueNeighbors
- MRREData, MRREProj
- PearsonCorrelation, ShepardGoodness
- AverageLocalError, DistanceConsistency, Procrustes
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np
from typing import Optional, Dict, List
import warnings
import wandb
import matplotlib.pyplot as plt

from gnn_dr.baselines.metrics import DRMetrics
from gnn_dr.utils.logging import get_logger
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
    warnings.warn(
        "tf-projection-qm not available. Install with: pip install tf-projection-qm\n"
        "tf-projection-qm metrics will be disabled."
    )


class DRMetricsCallback(Callback):
    """
    Evaluate dimensionality reduction quality metrics during training.
    
    This callback computes DR-specific metrics during validation:
    - UMAP loss (reconstruction of learned structure)
    - Trustworthiness (local structure preservation)
    - Continuity (inverse of trustworthiness)
    - k-NN recall (fraction of k-NN preserved)
    - Distance correlation (global structure preservation)
    - Silhouette score (cluster separation, if labels available)
    - Embedding statistics (norms, variance, etc.)
    - Scatter plot visualizations (reuses existing visualization code)
    
    Features:
    - Configurable evaluation frequency
    - Individual metric toggles
    - Scatter plot generation with class coloring
    - Embedding statistics tracking
    - Hierarchical W&B logging
    """
    
    def __init__(
        self,
        config=None,
        k_neighbors: int = 15,
        compute_every_n_epochs: int = 1,
        compute_trustworthiness: bool = True,
        compute_continuity: bool = True,
        compute_knn_recall: bool = True,
        compute_distance_correlation: bool = True,
        compute_silhouette: bool = True,
        compute_umap_loss: bool = False,  # Expensive, disabled by default
        generate_scatter_plots: bool = True,
        scatter_plot_interval: int = 5,
        track_embedding_stats: bool = True,
    ):
        """
        Initialize DR metrics callback.
        
        Args:
            config: ExperimentConfig with DR settings
            k_neighbors: Number of neighbors for local structure metrics
            compute_every_n_epochs: Compute metrics every N epochs
            compute_trustworthiness: Whether to compute trustworthiness
            compute_continuity: Whether to compute continuity
            compute_knn_recall: Whether to compute k-NN recall
            compute_distance_correlation: Whether to compute distance correlation
            compute_silhouette: Whether to compute silhouette score
            compute_umap_loss: Whether to compute UMAP loss (expensive)
            generate_scatter_plots: Whether to generate scatter plot visualizations
            scatter_plot_interval: Generate plots every N epochs
            track_embedding_stats: Whether to track embedding statistics
        """
        super().__init__()
        self.config = config
        self.k_neighbors = k_neighbors
        self.compute_every_n_epochs = compute_every_n_epochs
        
        # Metric computation flags
        self.compute_trustworthiness = compute_trustworthiness
        self.compute_continuity = compute_continuity
        self.compute_knn_recall = compute_knn_recall
        self.compute_distance_correlation = compute_distance_correlation
        self.compute_silhouette = compute_silhouette
        self.compute_umap_loss = compute_umap_loss
        
        # Visualization flags
        self.generate_scatter_plots = generate_scatter_plots
        self.scatter_plot_interval = scatter_plot_interval
        self.track_embedding_stats = track_embedding_stats
        
        # Metrics computer
        self.metrics_computer = DRMetrics(k_neighbors=k_neighbors)
        
        # Get logger
        self.logger_obj = get_logger()
        
        # Track metrics across epochs
        self.epoch_metrics = {}
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called after validation epoch completes.
        
        Computes and logs DR evaluation metrics.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module
        """
        # Only process if this is a DR task
        if not hasattr(pl_module, 'is_dr_task') or not pl_module.is_dr_task:
            return
        
        # Check if we should evaluate this epoch
        if trainer.current_epoch % self.compute_every_n_epochs != 0:
            return
        
        # Skip during sanity check
        if trainer.sanity_checking:
            return
        
        try:
            # Get validation data
            val_dataloader = trainer.val_dataloaders
            if val_dataloader is None:
                return
            
            # Get the first (and typically only) validation batch
            # which contains the full MNIST dataset
            val_batch = next(iter(val_dataloader))
            
            # Move to device
            val_batch = val_batch.to(pl_module.device)
            
            # Get model predictions
            # Note: encode=True is needed to encode CLIP embeddings to hidden dimension
            with torch.no_grad():
                predictions, _ = pl_module(val_batch, 10, return_layers=True, encode=True)
            
            # Convert to numpy
            pred_np = predictions.detach().cpu().numpy()
            
            # Get high-dim embeddings (CLIP embeddings)
            if hasattr(val_batch, 'clip_embedding'):
                high_dim_np = val_batch.clip_embedding.detach().cpu().numpy()
            elif hasattr(val_batch, 'x'):
                high_dim_np = val_batch.x.detach().cpu().numpy()
            else:
                warnings.warn("Cannot find embeddings in batch for DR metrics")
                return
            
            # Compute metrics based on configuration
            metrics_dict = {}
            
            # Trustworthiness
            if self.compute_trustworthiness:
                try:
                    metrics_dict['trustworthiness'] = self.metrics_computer.trustworthiness(
                        high_dim_np, pred_np
                    )
                except Exception as e:
                    warnings.warn(f"Error computing trustworthiness: {e}")
                    metrics_dict['trustworthiness'] = 0.0
            
            # Continuity
            if self.compute_continuity:
                try:
                    metrics_dict['continuity'] = self.metrics_computer.continuity(
                        high_dim_np, pred_np
                    )
                except Exception as e:
                    warnings.warn(f"Error computing continuity: {e}")
                    metrics_dict['continuity'] = 0.0
            
            # k-NN recall
            if self.compute_knn_recall:
                try:
                    metrics_dict['knn_recall'] = self.metrics_computer.knn_recall(
                        high_dim_np, pred_np
                    )
                except Exception as e:
                    warnings.warn(f"Error computing knn_recall: {e}")
                    metrics_dict['knn_recall'] = 0.0
            
            # Distance correlation
            if self.compute_distance_correlation:
                try:
                    metrics_dict['distance_correlation'] = self.metrics_computer.distance_correlation(
                        high_dim_np, pred_np
                    )
                except Exception as e:
                    warnings.warn(f"Error computing distance_correlation: {e}")
                    metrics_dict['distance_correlation'] = 0.0
            
            # Silhouette score (if labels available)
            if self.compute_silhouette and hasattr(val_batch, 'y') and val_batch.y is not None:
                try:
                    labels = val_batch.y.detach().cpu().numpy()
                    metrics_dict['silhouette_score'] = self.metrics_computer.silhouette(
                        pred_np, labels
                    )
                except Exception as e:
                    warnings.warn(f"Error computing silhouette: {e}")
            
            # UMAP loss (expensive, usually disabled)
            if self.compute_umap_loss:
                try:
                    umap_kwargs = {}
                    if hasattr(self.config, 'dimensionality_reduction'):
                        dr_config = self.config.dimensionality_reduction
                        umap_kwargs['k_neighbors'] = getattr(dr_config, 'knn_k', 15)
                        umap_kwargs['neg_sample_rate'] = getattr(dr_config, 'umap_neg_sample_rate', 0.3)
                        umap_kwargs['repulsion_weight'] = getattr(dr_config, 'umap_repulsion_weight', 1.0)
                    
                    metrics_dict['umap_loss'] = self.metrics_computer.umap_loss(
                        high_dim_np, pred_np, **umap_kwargs
                    )
                except Exception as e:
                    warnings.warn(f"Error computing UMAP loss: {e}")
            
            # Compute tf-projection-qm metrics if available and enabled
            tfpqm_metrics = self._compute_tfpqm_metrics(high_dim_np, pred_np, val_batch)
            if tfpqm_metrics:
                # Add with 'tfpqm/' prefix to distinguish from existing metrics
                for name, value in tfpqm_metrics.items():
                    metrics_dict[f'tfpqm/{name}'] = value
            
            # Track embedding statistics
            if self.track_embedding_stats:
                embedding_stats = self._compute_embedding_stats(predictions)
                metrics_dict.update(embedding_stats)
            
            # Generate scatter plot visualization using existing visualization code
            if self.generate_scatter_plots and (trainer.current_epoch % self.scatter_plot_interval == 0):
                try:
                    # Create title with metrics
                    title_parts = [f'Epoch {trainer.current_epoch}']
                    if 'trustworthiness' in metrics_dict:
                        title_parts.append(f'Trust: {metrics_dict["trustworthiness"]:.3f}')
                    if 'knn_recall' in metrics_dict:
                        title_parts.append(f'k-NN: {metrics_dict["knn_recall"]:.3f}')
                    title = ' | '.join(title_parts)
                    
                    # Use existing visualize_graph_layout function
                    fig = visualize_graph_layout(
                        val_batch,
                        predictions,
                        title=title,
                        config=self.config.validation if hasattr(self.config, 'validation') else None
                    )
                    
                    if fig is not None and trainer.logger is not None:
                        trainer.logger.experiment.log({
                            "dr_visualizations/scatter": wandb.Image(fig),
                            "epoch": trainer.current_epoch
                        })
                        plt.close(fig)
                except Exception as e:
                    warnings.warn(f"Failed to generate scatter plot: {e}")
            
            # Log metrics to W&B with metric name as top level
            if trainer.logger is not None:
                # For single validation graph, just use metric name directly
                trainer.logger.log_metrics(metrics_dict, step=trainer.global_step)
            
            # Store for later analysis
            self.epoch_metrics[trainer.current_epoch] = metrics_dict
            
            # Print to console if verbose
            if hasattr(self.config, 'verbose') and self.config.verbose:
                print("\n" + "="*60)
                print(f"Epoch {trainer.current_epoch} - DR Evaluation Metrics")
                print("="*60)
                for metric_name, metric_value in metrics_dict.items():
                    if isinstance(metric_value, float):
                        print(f"  {metric_name:30s}: {metric_value:.4f}")
                    else:
                        print(f"  {metric_name:30s}: {metric_value}")
                print("="*60 + "\n")
        
        except Exception as e:
            warnings.warn(f"Error in DRMetricsCallback: {e}")
    
    def _compute_embedding_stats(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics about the low-dimensional embeddings.
        
        Args:
            embeddings: Low-dimensional embeddings (N, d)
            
        Returns:
            Dictionary with embedding statistics
        """
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # Compute norms
        norms = np.linalg.norm(embeddings_np, axis=1)
        
        stats = {
            'embedding_mean_norm': float(np.mean(norms)),
            'embedding_std_norm': float(np.std(norms)),
            'embedding_max_norm': float(np.max(norms)),
            'embedding_min_norm': float(np.min(norms)),
            'embedding_mean_x': float(np.mean(embeddings_np[:, 0])),
            'embedding_mean_y': float(np.mean(embeddings_np[:, 1])),
            'embedding_std_x': float(np.std(embeddings_np[:, 0])),
            'embedding_std_y': float(np.std(embeddings_np[:, 1])),
        }
        
        return stats
    
    def _compute_tfpqm_metrics(
        self, 
        high_dim_np: np.ndarray, 
        pred_np: np.ndarray, 
        val_batch
    ) -> Dict[str, float]:
        """
        Compute tf-projection-qm metrics for comprehensive DR quality assessment.
        
        Uses the tf-projection-qm package which provides many standard DR evaluation metrics.
        See: https://github.com/mespadoto/proj-quant-eval
        
        Args:
            high_dim_np: High-dimensional embeddings (N, D) as numpy array
            pred_np: Low-dimensional projections (N, d) as numpy array
            val_batch: Validation batch (for labels if available)
            
        Returns:
            Dictionary with metric names and values
        """
        if not TFPQM_AVAILABLE:
            return {}
        
        # Check if tf-projection-qm is enabled in config
        logging_config = getattr(self.config, 'logging', None)
        if logging_config is None:
            return {}
        
        if not getattr(logging_config, 'tfpqm_enabled', True):
            return {}
        
        try:
            # Get configuration parameters
            k = getattr(logging_config, 'tfpqm_k', 15)
            n_classes = getattr(logging_config, 'tfpqm_n_classes', None)
            selected_metrics = getattr(logging_config, 'tfpqm_selected_metrics', None)
            
            # Get labels if available
            labels_np = None
            if hasattr(val_batch, 'y') and val_batch.y is not None:
                labels_np = val_batch.y.detach().cpu().numpy()
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
            import traceback
            traceback.print_exc()
            return {}
    
    def get_epoch_metrics(self, epoch: int) -> Optional[dict]:
        """
        Get metrics computed for a specific epoch.
        
        Args:
            epoch: Epoch number
            
        Returns:
            Dictionary of metrics for that epoch, or None if not available
        """
        return self.epoch_metrics.get(epoch)
    
    def get_all_metrics(self) -> dict:
        """
        Get all collected metrics across all epochs.
        
        Returns:
            Dictionary mapping epoch number to metrics dict
        """
        return self.epoch_metrics
