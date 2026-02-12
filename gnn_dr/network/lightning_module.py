"""
PyTorch Lightning module for CoRe-DR training.
"""

import torch
import pytorch_lightning as pl
import random
import numpy as np
from typing import Optional
from torch_geometric.loader import DataLoader
import wandb

from .model import CoReGD, get_model
from .losses import UMAPLoss
from .replay_buffer import ReplayBuffer, GPUReplayBuffer
from . import preprocessing
from gnn_dr.utils.constants import (
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_GRADIENT_CLIP_VALUE
)
from gnn_dr.config import config_to_namespace
from gnn_dr.utils.logging import (
    MetricsLogger,
    ReplayBufferMetrics,
    compute_gradient_stats,
    get_logger,
    set_logger,
)


class CoReGDLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for CoRe-DR (dimensionality reduction).

    Handles training, validation, and testing with:
    - UMAP loss (with t-SNE kernel auto-detection via a=1, b=1)
    - Replay buffer management
    - Graph coarsening
    - Dynamic iteration scheduling
    - Replay buffer sampling (functionally equivalent to original)
    """
    
    def __init__(self, config):
        """
        Initialize the Lightning module.
        
        Args:
            config: ExperimentConfig object with all hyperparameters
        """
        super().__init__()
        
        # Enable manual optimization for replay buffer training
        self.automatic_optimization = False
        
        # Save hyperparameters
        self.save_hyperparameters(config.to_flat_dict())
        self.config = config
        self.is_dr_task = True
        
        # Create model (get_model expects a namespace-like object with flat attributes)
        config_namespace = config_to_namespace(config)
        self.model = get_model(config_namespace)
        
        # Loss functions (UMAP loss for dimensionality reduction)
        # Get the weighting strategy from config
        # False (default): Sample edges proportionally to weights, take simple mean
        # True: Sample edges uniformly, multiply loss by edge weights (closer to standard UMAP)
        use_weighted_loss = getattr(config.dimensionality_reduction, 'umap_use_weighted_loss', False)

        # Determine kernel parameters a, b
        # If edge_weight_method='tsne' and a,b not explicitly set, default to
        # Student-t kernel (a=1, b=1) instead of UMAP's fitted curve
        a = config.dimensionality_reduction.umap_a
        b = config.dimensionality_reduction.umap_b
        edge_weight_method = getattr(config.dimensionality_reduction, 'edge_weight_method', 'umap')
        if edge_weight_method == 'tsne' and a is None and b is None:
            a, b = 1.0, 1.0  # Student-t kernel: q = 1/(1 + d²)

        # Use UMAP loss with configurable weighting strategy
        self.train_loss_fn = UMAPLoss(
            num_positive_samples=getattr(config.dimensionality_reduction, 'umap_num_positive_samples', 1000),
            num_negatives_per_edge=getattr(config.dimensionality_reduction, 'umap_num_negatives_per_edge', 5),
            repulsion_weight=config.dimensionality_reduction.umap_repulsion_weight,
            min_dist=config.dimensionality_reduction.umap_min_dist,
            spread=config.dimensionality_reduction.umap_spread,
            a=a,
            b=b,
            metric='euclidean',  # Must match graph construction metric
            use_weighted_loss=use_weighted_loss,  # Configurable weighting strategy
        )
        self.val_loss_fn = UMAPLoss(
            num_positive_samples=getattr(config.dimensionality_reduction, 'umap_num_positive_samples', 1000),
            num_negatives_per_edge=getattr(config.dimensionality_reduction, 'umap_num_negatives_per_edge', 5),
            repulsion_weight=config.dimensionality_reduction.umap_repulsion_weight,
            min_dist=config.dimensionality_reduction.umap_min_dist,
            spread=config.dimensionality_reduction.umap_spread,
            a=a,
            b=b,
            metric='euclidean',  # Must match graph construction metric
            use_weighted_loss=use_weighted_loss,  # Configurable weighting strategy
        )
        
        # Layer distribution for variable iterations
        self.layer_dist = torch.distributions.normal.Normal(
            config.training.iter_mean,
            max(config.training.iter_var, 1e-6)
        )
        
        # Replay buffer (clean implementation, will be initialized in on_fit_start)
        self.replay_buffer = None
        
        # Track best validation loss
        self.best_val_loss = float('inf')
        
    def forward(self, batch, iterations, **kwargs):
        """Forward pass through model."""
        return self.model(batch, iterations, **kwargs)
    
    def encode(self, batch):
        """Encode input features."""
        return self.model.encode(batch)
    
    def on_fit_start(self):
        """Initialize replay buffer at start of training."""
        # Log distributed training info
        self._log_distributed_info()
        
        if self.config.replay_buffer.use_replay_buffer:
            # Check if GPU replay buffer is enabled
            use_gpu_buffer = getattr(self.config.replay_buffer, 'use_gpu_replay_buffer', False)
            
            if use_gpu_buffer:
                # Create GPU-resident replay buffer (faster, no CPU-GPU transfers)
                self.replay_buffer = GPUReplayBuffer(
                    capacity=self.config.replay_buffer.replay_buffer_size,
                    batch_size=self.config.training.batch_size,
                    device=self.device
                )
            else:
                # Create CPU replay buffer (backward compatible)
                self.replay_buffer = ReplayBuffer(
                    capacity=self.config.replay_buffer.replay_buffer_size,
                    batch_size=self.config.training.batch_size
                )
            
            # Get initial embeddings from training set
            self.model.eval()
            with torch.no_grad():
                # Get train loader from datamodule
                train_dataset = self.trainer.datamodule.train_dataset
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.config.training.batch_size,
                    shuffle=False  # Don't need to shuffle for initialization
                )
                
                embeddings_list = []
                for batch in train_loader:
                    batch = batch.to(self.device)
                    batch_clone = batch.clone()
                    embeddings = self.model.encode(batch)
                    batch_clone.x = embeddings
                    
                    if use_gpu_buffer:
                        # Keep on GPU for GPU replay buffer
                        embeddings_list.extend(batch_clone.detach().to_data_list())
                    else:
                        # Move to CPU for standard replay buffer
                        embeddings_list.extend(batch_clone.detach().cpu().to_data_list())
                    
                    if len(embeddings_list) >= self.config.replay_buffer.replay_buffer_size:
                        break
                
                # Initialize replay buffer with embeddings
                self.replay_buffer.initialize_from_embeddings(
                    embeddings_list[:self.config.replay_buffer.replay_buffer_size]
                )
            
            self.model.train()
    
    def on_train_epoch_start(self):
        """Reset eigenvectors at start of each epoch if needed (Issue #2 fix)."""
        config_namespace = config_to_namespace(self.config)
        if config_namespace.randomize_between_epochs and config_namespace.laplace_eigvec > 0:
            # Reset eigenvectors in training dataset
            train_dataset = self.trainer.datamodule.train_dataset
            self.trainer.datamodule.train_dataset = preprocessing.reset_eigvecs(
                train_dataset, 
                config_namespace
            )
    
    def training_step(self, batch, batch_idx):
        """
        Training step with replay buffer sampling and coarsening.
        
        Two modes based on config.replay_buffer.accumulate_replay_gradients:
        
        When False (default):
            1. Train on dataset batch → backward → step
            2. For each replay batch → backward → step
            (1 + num_replay_batches optimizer steps)
        
        When True:
            1. Zero gradients
            2. Accumulate gradients from dataset batch
            3. Accumulate gradients from all replay batches
            4. Single optimizer step at the end
            (1 optimizer step with accumulated gradients)
        
        Args:
            batch: Batch of graph data
            batch_idx: Index of current batch
            
        Returns:
            Loss value
        """
        opt = self.optimizers()
        accumulate = getattr(self.config.replay_buffer, 'accumulate_replay_gradients', False)
        
        if accumulate and self.config.replay_buffer.use_replay_buffer:
            # Accumulated gradient mode: single optimizer step at the end
            loss = self._train_accumulated(batch, opt)
        else:
            # Default mode: separate optimizer steps per batch
            loss = self._train_separate_steps(batch, opt)
        
        # Log the dataset batch loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, 
                batch_size=batch.num_graphs)
        
        # Log batch graph size statistics (useful for dynamic batching monitoring)
        if self.config.logging.log_batch_graph_sizes:
            self._log_batch_graph_sizes(batch)
        
        # Manual optimization requires returning None
        return None
    
    def _train_separate_steps(self, batch, opt):
        """
        Train with separate optimizer steps for each batch (original behavior).
        
        Args:
            batch: Fresh dataset batch
            opt: Optimizer
            
        Returns:
            Loss value from fresh batch
        """
        # Step 1: Train on dataset batch
        loss, fresh_layers = self._train_single_batch(batch, opt, encode=True, 
                                       replacement_prob=self.config.replay_buffer.replay_train_replacement_prob)
        
        # Step 2: Train on replay buffer batches
        if self.config.replay_buffer.use_replay_buffer and self.replay_buffer is not None:
            for replay_idx in range(self.config.replay_buffer.num_replay_batches):
                # Sample batch with depth/staleness tracking
                sample_result = self.replay_buffer.sample_batch(self.device)
                
                # Handle both GPU (with tracking) and CPU (without tracking) replay buffers
                if isinstance(sample_result, tuple):
                    replay_batch, sampled_depths, sampled_stalenesses = sample_result
                else:
                    replay_batch = sample_result
                    sampled_depths = None
                    sampled_stalenesses = None
                
                _, replay_layers = self._train_single_batch(
                    replay_batch, opt, encode=False,
                    replacement_prob=self.config.replay_buffer.replay_buffer_replacement_prob,
                    previous_depths=sampled_depths
                )
                
                # Log sampled batch statistics for first replay batch
                if replay_idx == 0 and sampled_depths is not None:
                    self._log_replay_sample_stats(sampled_depths, sampled_stalenesses)
            
            # Increment global step after processing all batches
            if hasattr(self.replay_buffer, 'step'):
                self.replay_buffer.step()
        
        return loss
    
    def _train_accumulated(self, batch, opt):
        """
        Train with accumulated gradients: single optimizer step at the end.
        
        Gradients are accumulated from:
        - 1 fresh dataset batch
        - N replay buffer batches
        
        Then a single optimizer.step() is performed.
        
        Args:
            batch: Fresh dataset batch
            opt: Optimizer
            
        Returns:
            Loss value from fresh batch
        """
        opt.zero_grad()
        
        num_batches = 1 + self.config.replay_buffer.num_replay_batches
        
        # Step 1: Forward + backward on fresh batch (no optimizer step)
        fresh_loss = self._forward_backward_only(
            batch, encode=True,
            replacement_prob=self.config.replay_buffer.replay_train_replacement_prob,
            loss_scale=1.0 / num_batches  # Scale for averaging
        )
        
        # Step 2: Forward + backward on replay batches (no optimizer step)
        if self.replay_buffer is not None:
            for replay_idx in range(self.config.replay_buffer.num_replay_batches):
                # Sample batch with depth/staleness tracking
                sample_result = self.replay_buffer.sample_batch(self.device)
                
                # Handle both GPU (with tracking) and CPU (without tracking) replay buffers
                if isinstance(sample_result, tuple):
                    replay_batch, sampled_depths, sampled_stalenesses = sample_result
                else:
                    replay_batch = sample_result
                    sampled_depths = None
                    sampled_stalenesses = None
                
                self._forward_backward_only(
                    replay_batch, encode=False,
                    replacement_prob=self.config.replay_buffer.replay_buffer_replacement_prob,
                    loss_scale=1.0 / num_batches,  # Scale for averaging
                    previous_depths=sampled_depths
                )
                
                # Log sampled batch statistics for first replay batch
                if replay_idx == 0 and sampled_depths is not None:
                    self._log_replay_sample_stats(sampled_depths, sampled_stalenesses)
            
            # Increment global step after processing all batches
            if hasattr(self.replay_buffer, 'step'):
                self.replay_buffer.step()
        
        # Step 3: Gradient clipping and single optimizer step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), DEFAULT_GRADIENT_CLIP_NORM)
        torch.nn.utils.clip_grad_value_(self.model.parameters(), DEFAULT_GRADIENT_CLIP_VALUE)
        opt.step()
        
        return fresh_loss
    
    def _forward_backward_only(self, batch, encode, replacement_prob, loss_scale=1.0, previous_depths=None):
        """
        Forward and backward pass without optimizer step.
        
        Used for gradient accumulation mode.
        
        Args:
            batch: Batch to process
            encode: Whether to encode (True for fresh, False for replay)
            replacement_prob: Probability of replacing items in replay buffer
            loss_scale: Scale factor for loss (for gradient averaging)
            previous_depths: Previous iteration depths for replay batches (None for fresh)
            
        Returns:
            Loss value (unscaled)
        """
        # Sample number of layers
        layers = max(int(self.layer_dist.sample().item() + 0.5), 1)
        
        # Detach input
        batch.x = batch.x.detach()
        
        # Reset randomized features if needed
        if self.config.training.randomize_between_epochs and encode:
            config_namespace = config_to_namespace(self.config)
            batch = preprocessing.reset_randomized_features_batch(batch, config_namespace)
        
        # Forward pass
        if self.config.coarsening.coarsen and random.random() <= self.config.coarsening.coarsen_prob:
            loss, states = self._forward_with_coarsening(batch, layers, encode)
        else:
            loss, states = self._forward_standard(batch, layers, encode)
        
        # Backward pass with scaled loss (for gradient averaging)
        scaled_loss = loss * loss_scale
        self.manual_backward(scaled_loss)
        
        # Update replay buffer with iteration tracking
        if self.config.replay_buffer.use_replay_buffer:
            self._update_replay_buffer_with_prob(
                batch, states[-1], replacement_prob,
                num_iterations=layers, previous_depths=previous_depths
            )
        
        return loss.item()
    
    def _train_single_batch(self, batch, optimizer, encode, replacement_prob, previous_depths=None):
        """
        Train on a single batch (either dataset or replay buffer).
        
        Args:
            batch: Batch to train on
            optimizer: Optimizer to use
            encode: Whether to encode (True for dataset, False for replay)
            replacement_prob: Probability of replacing items in replay buffer
            previous_depths: Previous iteration depths for replay batches (None for fresh)
            
        Returns:
            Tuple of (loss_value, num_layers) for tracking
        """
        optimizer.zero_grad()
        
        # Sample number of layers
        layers = max(int(self.layer_dist.sample().item() + 0.5), 1)
        
        # Detach input
        batch.x = batch.x.detach()
        
        # Reset randomized features if needed
        if self.config.training.randomize_between_epochs and encode:
            config_namespace = config_to_namespace(self.config)
            batch = preprocessing.reset_randomized_features_batch(batch, config_namespace)
        
        # Forward pass (with or without coarsening)
        if self.config.coarsening.coarsen and random.random() <= self.config.coarsening.coarsen_prob:
            loss, states = self._forward_with_coarsening(batch, layers, encode)
        else:
            loss, states = self._forward_standard(batch, layers, encode)
        
        # Backward pass
        self.manual_backward(loss)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), DEFAULT_GRADIENT_CLIP_NORM)
        torch.nn.utils.clip_grad_value_(self.model.parameters(), DEFAULT_GRADIENT_CLIP_VALUE)
        
        # Optimizer step
        optimizer.step()
        
        # Update replay buffer with iteration tracking
        if self.config.replay_buffer.use_replay_buffer:
            self._update_replay_buffer_with_prob(
                batch, states[-1], replacement_prob, 
                num_iterations=layers, previous_depths=previous_depths
            )
        
        return loss.item(), layers
    
    def _forward_standard(self, batch, layers, encode):
        """Standard forward pass without coarsening."""
        pred, states = self(batch, layers, return_layers=True, encode=encode)
        loss = self.train_loss_fn(pred, batch)
        return loss, states
    
    def _forward_with_coarsening(self, batch, layers, encode):
        """Forward pass with graph coarsening (matches original implementation)."""
        # Split iterations before and after coarsening
        layers_before = max(int((self.layer_dist.sample().item() / 2) + 0.5), 1)
        layers_after = max(int((self.layer_dist.sample().item() / 2) + 0.5), 1)
        
        # First part: before coarsening
        pred, states = self(batch, layers_before, return_layers=True, encode=encode)
        batch.x = states[-1]
        
        # Coarsen graphs (following original logic exactly)
        graphs = batch.to_data_list()
        for i in range(len(graphs)):
            if graphs[i].coarsening_level < len(self.trainer.datamodule.coarsening_matrices[graphs[i].index]):
                graphs[i] = self._go_to_coarser_graph(
                    graphs[i], 
                    graphs[i].x, 
                    self.trainer.datamodule.coarsened_graphs,
                    self.trainer.datamodule.coarsening_matrices
                )
        
        # Rebatch (PyG will handle creating the correct batch attribute)
        import torch_geometric.data
        batch = torch_geometric.data.Batch.from_data_list(graphs)
        
        # Second part: after coarsening
        pred, states = self(batch, layers_after, return_layers=True, encode=False)
        loss = self.train_loss_fn(pred, batch)
        
        return loss, states
    
    def _go_to_coarser_graph(self, graph, last_embeddings, coarsened_graphs, coarsening_matrices, set_batch_attr=False):
        """
        Move to a coarser graph level.

        Creates a NEW Data object instead of mutating the existing one.
        This ensures PyG's internal storage is set up correctly with num_nodes.

        Uses clip_embedding from coarsened graphs and reconstructs x_orig.

        Args:
            graph: Graph to coarsen
            last_embeddings: Embeddings from previous level
            coarsened_graphs: List of coarsened graph structures
            coarsening_matrices: Coarsening transformation matrices
            set_batch_attr: If True, set graph.batch attribute (needed for validation/test)
        """
        new_level = graph.coarsening_level + 1

        # Transform embeddings to coarser level
        embeddings_finer = torch.transpose(
            torch.sparse.mm(
                torch.transpose(last_embeddings, 0, 1),
                coarsening_matrices[graph.index][-new_level].to(self.device)
            ),
            0, 1
        )
        
        # Add noise if configured
        if self.config.coarsening.coarsen_noise > 0:
            noise = torch.tensor(
                np.random.normal(0, self.config.coarsening.coarsen_noise, embeddings_finer.size()),
                dtype=torch.float,
                device=self.device
            )
            embeddings_finer = embeddings_finer + noise
        
        # Create a NEW Data object by explicitly copying ALL TENSOR attributes
        # This ensures PyG's internal storage is initialized correctly
        from torch_geometric.data import Data
        
        # Get all attributes from the original graph, but exclude int attributes
        # that should be set separately
        data_dict = {}
        for key in graph.keys():
            # Skip int attributes - they'll be set after Data creation
            if key in ['num_nodes', 'coarsening_level', 'index']:
                continue
            data_dict[key] = graph[key]
        
        # Update with coarsening-specific tensor values
        data_dict['x'] = embeddings_finer
        data_dict['num_nodes'] = embeddings_finer.size(0)
        data_dict['edge_index'] = coarsened_graphs[graph.index][-new_level-1].edge_index.to(self.device)
        
        # Use preprocessed coarse graph (which has x_orig already computed)
        coarse_graph = coarsened_graphs[graph.index][-new_level-1]
        if hasattr(coarse_graph, 'clip_embedding') and coarse_graph.clip_embedding is not None:
            data_dict['clip_embedding'] = coarse_graph.clip_embedding.to(self.device)
        if hasattr(coarse_graph, 'x_orig') and coarse_graph.x_orig is not None:
            data_dict['x_orig'] = coarse_graph.x_orig.to(self.device)
        
        # Set batch attribute for single-graph processing (validation/test)
        if set_batch_attr:
            data_dict['batch'] = torch.zeros(embeddings_finer.shape[0], device=self.device, dtype=torch.int64)
        
        # Create new Data object with tensor attributes only
        new_graph = Data(**data_dict)
        
        # Set int attributes AFTER Data creation
        # PyG will infer num_nodes from x.shape[0] automatically
        new_graph.coarsening_level = new_level
        new_graph.index = graph.index
        
        return new_graph
    
    def _update_replay_buffer_with_prob(self, batch, final_state, replacement_prob, 
                                        num_iterations=0, previous_depths=None):
        """
        Update replay buffer with new embeddings.
        
        Uses clean ReplayBuffer implementation that stores only essential data
        and avoids stale attribute issues.
        
        Args:
            batch: Current batch
            final_state: Final embeddings from forward pass
            replacement_prob: Probability of replacing items in buffer
            num_iterations: Number of GNN iterations applied in this forward pass
            previous_depths: Previous iteration depths for replay batches (None for fresh)
        """
        # Update batch with final embeddings
        batch.x = final_state.detach()
        
        # Use ReplayBuffer's add() method with iteration tracking for GPU buffer
        if hasattr(self.replay_buffer, 'add') and hasattr(self.replay_buffer.add, '__code__'):
            # Check if add() accepts the new parameters (GPU replay buffer)
            if 'num_iterations' in self.replay_buffer.add.__code__.co_varnames:
                self.replay_buffer.add(
                    batch, 
                    replacement_prob=replacement_prob,
                    num_iterations=num_iterations,
                    previous_depths=previous_depths
                )
            else:
                # CPU replay buffer (no tracking)
                self.replay_buffer.add(batch, replacement_prob=replacement_prob)
        else:
            self.replay_buffer.add(batch, replacement_prob=replacement_prob)
    
    def _inference_with_coarsening(self, batch, layer_num, coarsened_graphs, coarsening_matrices):
        """
        Run inference with optional coarsening.

        Shared inference logic used by validation, test, and visualization.
        
        Args:
            batch: Input graph batch
            layer_num: Number of layers for forward pass
            coarsened_graphs: List of coarsened graph hierarchies (or None)
            coarsening_matrices: List of coarsening transformation matrices (or None)
            
        Returns:
            Tuple of (predictions, final_batch) where final_batch may be coarsened
        """
        # Forward pass
        pred, states = self(batch, layer_num, return_layers=True)
        
        # Apply coarsening if enabled
        if self.config.coarsening.coarsen and coarsened_graphs is not None and coarsening_matrices is not None:
            for i in range(1, len(coarsening_matrices[batch.index]) + 1):
                batch = self._go_to_coarser_graph(
                    batch, states[-1], 
                    coarsened_graphs,
                    coarsening_matrices,
                    set_batch_attr=True  # Set batch attribute for single-graph processing
                )
                pred, states = self(batch, layer_num, encode=False, return_layers=True)
        
        return pred, batch
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Use configurable number of layers
        layer_num = self.config.validation.num_layers
        
        # Forward pass
        pred, states = self(batch, layer_num, return_layers=True)
        
        if self.config.coarsening.coarsen and \
           hasattr(self.trainer.datamodule, 'val_coarsening_matrices'):
            for i in range(1, len(self.trainer.datamodule.val_coarsening_matrices[batch.index]) + 1):
                batch = self._go_to_coarser_graph(
                    batch, states[-1], 
                    self.trainer.datamodule.val_coarsened_graphs,
                    self.trainer.datamodule.val_coarsening_matrices,
                    set_batch_attr=True  # Set batch attribute for single-graph processing
                )
                pred, states = self(batch, layer_num, encode=False, return_layers=True)
        
        # Compute losses
        loss = self.val_loss_fn(pred, batch)
        
        # Log metrics (validation is per-graph, so batch_size=1)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step (same as validation)."""
        # Use configurable number of layers
        layer_num = self.config.validation.num_layers
        
        pred, states = self(batch, layer_num, return_layers=True)
        
        if self.config.coarsening.coarsen and \
           hasattr(self.trainer.datamodule, 'test_coarsening_matrices'):
            for i in range(1, len(self.trainer.datamodule.test_coarsening_matrices[batch.index]) + 1):
                batch = self._go_to_coarser_graph(
                    batch, states[-1],
                    self.trainer.datamodule.test_coarsened_graphs,
                    self.trainer.datamodule.test_coarsening_matrices,
                    set_batch_attr=True  # Set batch attribute for single-graph processing
                )
                pred, states = self(batch, layer_num, encode=False, return_layers=True)
        
        loss = self.val_loss_fn(pred, batch)
        
        # Log metrics (test is per-graph, so batch_size=1)
        self.log('test_loss', loss, batch_size=1)

        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay
        )
        
        # Configure scheduler based on dataset
        if self.config.dataset.dataset in ['suitesparse', 'delaunay']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.8,
                patience=20,
                threshold=2,
                threshold_mode='abs',
                min_lr=0.00000001
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.8,
                patience=20,
                threshold=2,
                threshold_mode='abs',
                min_lr=0.00000001
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }
    
    def on_train_epoch_end(self):
        """Step the learning rate scheduler at end of epoch (for manual optimization)."""
        sch = self.lr_schedulers()
        
        # ReduceLROnPlateau needs the validation loss
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # Get the validation loss from logged metrics
            val_loss = self.trainer.callback_metrics.get('val_loss')
            if val_loss is not None:
                sch.step(val_loss)
        
        # Log replay buffer statistics at end of epoch (for GPU buffer with tracking)
        self._log_replay_buffer_statistics()
    
    def _log_batch_graph_sizes(self, batch):
        """
        Log batch graph size statistics to WandB.
        
        Useful for monitoring dynamic batching behavior and understanding
        the distribution of graph sizes during training.
        
        Args:
            batch: PyG Batch object containing multiple graphs
        """
        # Get per-graph sizes from the batch
        # batch.ptr contains the cumulative node counts [0, n1, n1+n2, ...]
        if hasattr(batch, 'ptr') and batch.ptr is not None:
            ptr = batch.ptr
            graph_sizes = (ptr[1:] - ptr[:-1]).cpu().tolist()
        else:
            # Fallback: if no ptr, estimate from num_nodes / num_graphs
            graph_sizes = [batch.num_nodes // max(batch.num_graphs, 1)] * batch.num_graphs
        
        # Compute statistics
        num_graphs = batch.num_graphs
        total_nodes = batch.num_nodes
        avg_size = total_nodes / max(num_graphs, 1)
        
        if len(graph_sizes) > 0:
            max_size = max(graph_sizes)
            min_size = min(graph_sizes)
        else:
            max_size = 0
            min_size = 0
        
        # Log to WandB
        self.log('train/batch_num_graphs', float(num_graphs), on_step=True, on_epoch=False)
        self.log('train/batch_total_nodes', float(total_nodes), on_step=True, on_epoch=False)
        self.log('train/batch_avg_graph_size', avg_size, on_step=True, on_epoch=False)
        self.log('train/batch_max_graph_size', float(max_size), on_step=True, on_epoch=False)
        self.log('train/batch_min_graph_size', float(min_size), on_step=True, on_epoch=False)
    
    def _log_distributed_info(self):
        """
        Log distributed training information at the start of training.
        
        Useful for verifying multi-GPU setup is working correctly.
        Note: self.log() is not allowed in on_fit_start, so we only use print().
        """
        world_size = self.trainer.world_size
        global_rank = self.trainer.global_rank
        local_rank = self.trainer.local_rank
        
        # Always log on rank 0
        if global_rank == 0:
            print("\n" + "=" * 60)
            print("DISTRIBUTED TRAINING CONFIGURATION")
            print("=" * 60)
            print(f"  World size (total GPUs): {world_size}")
            print(f"  Num nodes: {self.trainer.num_nodes}")
            print(f"  Strategy: {self.trainer.strategy.__class__.__name__}")
            if world_size > 1:
                print(f"  Mode: Multi-GPU (DDP)")
            else:
                print(f"  Mode: Single-GPU")
            print("=" * 60 + "\n")
        
        # Log from each rank (for verification)
        if world_size > 1:
            print(f"[Rank {global_rank}/{world_size-1}] Device: {self.device}, "
                  f"Local rank: {local_rank}")
    
    def _log_replay_sample_stats(self, sampled_depths, sampled_stalenesses):
        """
        Log statistics for a sampled replay batch.
        
        Called once per training step (for the first replay batch only to avoid overhead).
        
        Args:
            sampled_depths: List of iteration depths for sampled graphs
            sampled_stalenesses: List of stalenesses for sampled graphs
        """
        if not self.config.logging.log_replay_buffer:
            return
        
        if sampled_depths is None or len(sampled_depths) == 0:
            return
        
        # Use the replay buffer's helper if available
        if hasattr(self.replay_buffer, 'get_sampled_batch_statistics'):
            stats = self.replay_buffer.get_sampled_batch_statistics(
                sampled_depths, sampled_stalenesses
            )
        else:
            # Manual computation
            depths_arr = np.array(sampled_depths)
            stalenesses_arr = np.array(sampled_stalenesses)
            stats = {
                'sampled_depth_mean': float(np.mean(depths_arr)),
                'sampled_depth_max': int(np.max(depths_arr)),
                'sampled_staleness_mean': float(np.mean(stalenesses_arr)),
                'sampled_staleness_max': int(np.max(stalenesses_arr)),
            }
        
        # Log sampled batch statistics (on_step=True for per-step tracking)
        self.log('replay/sampled_depth_mean', stats['sampled_depth_mean'], 
                 on_step=True, on_epoch=False)
        self.log('replay/sampled_depth_max', float(stats['sampled_depth_max']), 
                 on_step=True, on_epoch=False)
        self.log('replay/sampled_staleness_mean', stats['sampled_staleness_mean'], 
                 on_step=True, on_epoch=False)
        self.log('replay/sampled_staleness_max', float(stats['sampled_staleness_max']), 
                 on_step=True, on_epoch=False)
    
    def _log_replay_buffer_statistics(self):
        """
        Log buffer-wide statistics at the end of each epoch.
        
        Only logs if using GPU replay buffer with tracking enabled.
        """
        if not self.config.replay_buffer.use_replay_buffer:
            return
        
        if self.replay_buffer is None:
            return
        
        if not self.config.logging.log_replay_buffer:
            return
        
        # Check if buffer has tracking statistics
        if not hasattr(self.replay_buffer, 'get_buffer_statistics'):
            return
        
        stats = self.replay_buffer.get_buffer_statistics()
        
        # Log buffer-wide statistics (on_epoch=True for per-epoch tracking)
        self.log('replay/buffer_size', float(stats['buffer_size']), 
                 on_step=False, on_epoch=True)
        self.log('replay/buffer_depth_mean', stats['depth_mean'], 
                 on_step=False, on_epoch=True)
        self.log('replay/buffer_depth_std', stats['depth_std'], 
                 on_step=False, on_epoch=True)
        self.log('replay/buffer_depth_min', float(stats['depth_min']), 
                 on_step=False, on_epoch=True)
        self.log('replay/buffer_depth_max', float(stats['depth_max']), 
                 on_step=False, on_epoch=True)
        self.log('replay/buffer_staleness_mean', stats['staleness_mean'], 
                 on_step=False, on_epoch=True)
        self.log('replay/buffer_staleness_std', stats['staleness_std'], 
                 on_step=False, on_epoch=True)
        self.log('replay/buffer_staleness_min', float(stats['staleness_min']), 
                 on_step=False, on_epoch=True)
        self.log('replay/buffer_staleness_max', float(stats['staleness_max']), 
                 on_step=False, on_epoch=True)
        self.log('replay/buffer_memory_mb', stats['memory_mb'], 
                 on_step=False, on_epoch=True)
        self.log('replay/global_step', float(stats['global_step']), 
                 on_step=False, on_epoch=True)
