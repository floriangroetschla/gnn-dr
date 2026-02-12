"""
Centralized logging utilities for W&B integration.

Provides structured logging functions for training metrics with configuration-driven verbosity.
"""

import wandb
import torch
import numpy as np
from typing import Dict, Optional, Any, List
from collections import defaultdict


class MetricsLogger:
    """
    Centralized metrics logger for W&B.
    
    Handles structured logging with verbosity control and metric grouping.
    """
    
    def __init__(self, log_level: str = 'basic', log_every_n_steps: int = 50):
        """
        Initialize metrics logger.
        
        Args:
            log_level: 'off', 'basic', 'detailed', or 'full'
            log_every_n_steps: Only log detailed metrics every N steps (avoid spam)
        """
        self.log_level = log_level
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
        self.metric_buffer = defaultdict(list)
    
    def log_training_step(self, metrics: Dict[str, float], step: int = None):
        """
        Log training step metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (used for filtering)
        """
        if self.log_level == 'off':
            return
        
        if step is not None:
            self.step_count = step
        
        # Always log basic metrics
        # Note: Don't pass step= to wandb.log() - let PyTorch Lightning handle step tracking
        if self.log_level in ['basic', 'detailed', 'full']:
            wandb.log(metrics)
    
    def log_replay_buffer_stats(self, stats: Dict[str, Any]):
        """
        Log replay buffer statistics.
        
        Args:
            stats: Dictionary with keys like 'buffer_size', 'occupancy_rate', 'items_added', etc.
        """
        if self.log_level == 'off':
            return
        
        if self.log_level in ['basic', 'detailed', 'full']:
            # Prefix all with 'replay_buffer/' for dashboard organization
            prefixed = {f'replay_buffer/{k}': v for k, v in stats.items()}
            # Don't pass step - let PyTorch Lightning handle step tracking
            wandb.log(prefixed)
    
    def log_loss_components(self, components: Dict[str, float]):
        """
        Log loss component breakdown.
        
        Args:
            components: Dictionary with component names like 'attraction', 'repulsion', etc.
        """
        if self.log_level == 'off':
            return
        
        if self.log_level in ['detailed', 'full']:  # More verbose
            prefixed = {f'loss_components/{k}': v for k, v in components.items()}
            # Don't pass step - let PyTorch Lightning handle step tracking
            wandb.log(prefixed)
    
    def log_gradient_stats(self, stats: Dict[str, float]):
        """
        Log gradient statistics.
        
        Args:
            stats: Dictionary with keys like 'mean_norm', 'max_norm', 'std_norm'
        """
        if self.log_level in ['detailed', 'full']:
            prefixed = {f'gradients/{k}': v for k, v in stats.items()}
            # Don't pass step - let PyTorch Lightning handle step tracking
            wandb.log(prefixed)
    
    def log_dataset_stats(self, stats: Dict[str, Any]):
        """
        Log dataset/batch statistics.
        
        Args:
            stats: Dictionary with graph properties
        """
        if self.log_level == 'full':  # Most verbose
            prefixed = {f'dataset/{k}': v for k, v in stats.items()}
            # Don't pass step - let PyTorch Lightning handle step tracking
            wandb.log(prefixed)
    
    def log_custom(self, category: str, metrics: Dict[str, float]):
        """
        Log custom metrics under a category.
        
        Args:
            category: Category name (will be used as prefix)
            metrics: Dictionary of metrics
        """
        if self.log_level == 'off':
            return
        
        prefixed = {f'{category}/{k}': v for k, v in metrics.items()}
        # Don't pass step - let PyTorch Lightning handle step tracking
        wandb.log(prefixed)
    
    def should_log_detailed(self) -> bool:
        """Check if we should log detailed metrics (not every step)."""
        return (self.step_count % self.log_every_n_steps) == 0


def compute_gradient_stats(model: torch.nn.Module) -> Dict[str, float]:
    """
    Compute gradient statistics across all model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with gradient stats
    """
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.data.cpu().numpy().flatten())
    
    if not grads:
        return {
            'mean_norm': 0.0,
            'max_norm': 0.0,
            'min_norm': 0.0,
            'std_norm': 0.0,
        }
    
    all_grads = np.concatenate(grads)
    grad_norms = np.abs(all_grads)
    
    return {
        'mean_norm': float(np.mean(grad_norms)),
        'max_norm': float(np.max(grad_norms)),
        'min_norm': float(np.min(grad_norms)),
        'std_norm': float(np.std(grad_norms)),
    }


def compute_gradient_stats_per_layer(model: torch.nn.Module) -> Dict[str, float]:
    """
    Compute gradient statistics per layer (useful for detecting vanishing gradients).
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with per-layer gradient stats
    """
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm().item()
            stats[f'grad_norm/{name}'] = grad_norm
    
    return stats


def compute_weight_stats(model: torch.nn.Module) -> Dict[str, float]:
    """
    Compute weight statistics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with weight stats
    """
    stats = {}
    for name, param in model.named_parameters():
        weight_norm = param.data.norm().item()
        stats[f'weight_norm/{name}'] = weight_norm
    
    return stats


def log_tensor_stats(tensor: torch.Tensor, name: str, logger: Optional[MetricsLogger] = None) -> Dict[str, float]:
    """
    Compute and optionally log statistics for a tensor.
    
    Args:
        tensor: Tensor to analyze
        name: Name for the tensor
        logger: Optional MetricsLogger to log to
        
    Returns:
        Dictionary with tensor stats
    """
    tensor_np = tensor.detach().cpu().numpy().flatten()
    
    stats = {
        f'{name}/mean': float(np.mean(tensor_np)),
        f'{name}/std': float(np.std(tensor_np)),
        f'{name}/min': float(np.min(tensor_np)),
        f'{name}/max': float(np.max(tensor_np)),
    }
    
    if logger is not None:
        logger.log_custom('tensors', stats)
    
    return stats


class ReplayBufferMetrics:
    """Track metrics for replay buffer."""
    
    def __init__(self):
        """Initialize replay buffer metrics tracker."""
        self.items_added = 0
        self.items_replaced = 0
        self.total_samples = 0
        self.replacement_counts = defaultdict(int)
    
    def record_add(self, num_items: int, num_replaced: int):
        """
        Record items added to buffer.
        
        Args:
            num_items: Number of items added
            num_replaced: Number of items that were replaced (vs appended)
        """
        self.items_added += num_items
        self.items_replaced += num_replaced
        self.total_samples += num_items
    
    def record_sample(self, batch_size: int):
        """
        Record batch sampled from buffer.
        
        Args:
            batch_size: Size of sampled batch
        """
        self.total_samples += batch_size
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get current statistics.
        
        Returns:
            Dictionary with metrics
        """
        return {
            'items_added': float(self.items_added),
            'items_replaced': float(self.items_replaced),
            'total_samples': float(self.total_samples),
            'replacement_rate': float(self.items_replaced / max(1, self.total_samples)),
        }
    
    def reset(self):
        """Reset metrics for new epoch."""
        self.items_added = 0
        self.items_replaced = 0
        self.total_samples = 0


# Global logger instance
_global_logger: Optional[MetricsLogger] = None


def get_logger() -> MetricsLogger:
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = MetricsLogger()
    return _global_logger


def set_logger(logger: MetricsLogger):
    """Set global logger instance."""
    global _global_logger
    _global_logger = logger
