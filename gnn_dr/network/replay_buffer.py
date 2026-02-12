"""
Clean replay buffer implementation for CoRe-GD.

Stores complete graph objects and reconstructs them with proper tensor filtering.
Supports both CPU and GPU-resident storage for optimal performance.

Tracking metrics:
- iteration_depth: Total number of GNN iterations the embedding has undergone
- staleness: Number of training steps since the embedding was last updated
- creation_step: Global training step when the embedding was created/updated
"""

import torch
import random
import numpy as np
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Optional, Union, Dict
from gnn_dr.utils.logging import ReplayBufferMetrics


class ReplayBuffer:
    """
    Replay buffer that stores complete graphs and reconstructs them cleanly.
    
    This approach stores whole graph objects, then carefully reconstructs them
    with only tensor attributes to avoid PyG collate issues.
    
    Attributes:
        capacity: Maximum number of graphs to store
        batch_size: Batch size for sampling
        buffer: List of stored graph objects
    """
    
    def __init__(self, capacity: int, batch_size: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of graphs to store
            batch_size: Number of graphs per batch when sampling
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer: List[Data] = []
        
        # Tracking metrics
        self.items_added = 0  # Total items added (initialization + training)
        self.items_replaced = 0  # Items that were replaced (vs appended)
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)
    
    def add(self, batch: Batch, replacement_prob: float = 1.0):
        """
        Add graphs from batch to replay buffer.
        
        Stores complete cloned graphs for simplicity.
        
        Args:
            batch: Batch of graphs to add (with final embeddings in batch.x)
            replacement_prob: Probability of replacing existing items
        """
        # Extract individual graphs
        graphs = batch.detach().cpu().to_data_list()
        
        for graph in graphs:
            # Only store with probability replacement_prob
            if random.random() > replacement_prob:
                continue
            
            self.items_added += 1
            
            # Store the complete cloned graph
            if len(self.buffer) < self.capacity:
                self.buffer.append(graph.clone())
            else:
                # Random replacement
                idx = random.randint(0, len(self.buffer) - 1)
                self.buffer[idx] = graph.clone()
                self.items_replaced += 1
    
    def sample_batch(self, device: torch.device) -> Batch:
        """
        Sample a batch of graphs from the replay buffer.
        
        Reconstructs clean Data objects with only tensor attributes,
        ensuring PyG's collate function works correctly.
        
        Args:
            device: Device to move tensors to
            
        Returns:
            Batch of reconstructed graphs
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty replay buffer")
        
        # Sample random indices
        num_to_sample = min(self.batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), num_to_sample)
        
        # Reconstruct clean graphs with ONLY tensor attributes
        graphs = []
        for idx in indices:
            stored_graph = self.buffer[idx]
            
            # Create new Data object with ONLY tensor attributes
            # This prevents PyG's collate from trying to process int attributes
            graph = Data()
            for key in stored_graph.keys():
                value = stored_graph[key]
                # Only add tensor attributes to the Data object
                if torch.is_tensor(value):
                    graph[key] = value.clone()
            
            # Set integer metadata separately (after Data creation)
            # These won't be collated by PyG
            if hasattr(stored_graph, 'coarsening_level'):
                graph.coarsening_level = stored_graph.coarsening_level
            if hasattr(stored_graph, 'index'):
                graph.index = stored_graph.index
            if hasattr(stored_graph, 'y') and stored_graph.y is not None:
                # Preserve class labels if available
                if torch.is_tensor(stored_graph.y):
                    graph.y = stored_graph.y.clone()
                else:
                    graph.y = stored_graph.y

            graph.num_nodes = graph.x.shape[0]
            
            graphs.append(graph)
        
        # Batch graphs and move to device
        batched = Batch.from_data_list(graphs)
        return batched.to(device)
    
    def initialize_from_embeddings(self, embeddings_list: List[Data]):
        """
        Initialize replay buffer with initial embeddings.
        
        Args:
            embeddings_list: List of graphs with encoded features
        """
        self.buffer = []
        self.items_added = 0  # Reset counters at initialization
        self.items_replaced = 0
        
        for graph in embeddings_list[:self.capacity]:
            self.buffer.append(graph.clone())
            self.items_added += 1  # Track initialization items as "added"
    
    def get_stats_and_reset(self) -> dict:
        """
        Get current stats and reset counters for this epoch.
        
        Returns:
            Dictionary with stats for this epoch
        """
        stats = {
            'items_added': self.items_added,
            'items_replaced': self.items_replaced,
        }
        # Don't reset - keep cumulative for overall view
        return stats


class GPUReplayBuffer:
    """
    GPU-resident replay buffer that keeps all data on GPU.
    
    Eliminates CPU-GPU memory transfers during training by storing
    graphs directly on the GPU. Significantly faster for GPU-optimized
    datasets where data is already on GPU.
    
    Key differences from CPU ReplayBuffer:
    - All tensors stay on GPU (no .cpu() calls)
    - No CPU-GPU transfers during add() or sample_batch()
    - Requires more GPU memory but much faster
    
    Tracking metrics per graph:
    - iteration_depth: Total GNN iterations the embedding has undergone
    - creation_step: Global training step when last updated
    
    Attributes:
        capacity: Maximum number of graphs to store
        batch_size: Batch size for sampling
        device: GPU device for storage
        buffer: List of stored graph objects (all tensors on GPU)
        iteration_depths: List of cumulative iteration counts per graph
        creation_steps: List of global step when each graph was last updated
        global_step: Current global training step counter
    """
    
    def __init__(self, capacity: int, batch_size: int, device: Union[str, torch.device] = 'cuda'):
        """
        Initialize GPU replay buffer.
        
        Args:
            capacity: Maximum number of graphs to store
            batch_size: Number of graphs per batch when sampling
            device: GPU device for storage (default: 'cuda')
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.buffer: List[Data] = []
        
        # Tracking metrics (per-item)
        self.iteration_depths: List[int] = []  # Cumulative iterations per graph
        self.creation_steps: List[int] = []    # Global step when last updated
        
        # Global counters
        self.global_step = 0  # Current global training step
        self.items_added = 0
        self.items_replaced = 0
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)
    
    def _ensure_on_device(self, graph: Data) -> Data:
        """
        Ensure all tensor attributes are on the target device.
        
        Args:
            graph: Input graph (may have tensors on any device)
            
        Returns:
            Graph with all tensors on self.device
        """
        new_graph = Data()
        for key in graph.keys():
            value = graph[key]
            if torch.is_tensor(value):
                # Move to target device without cloning if already there
                if value.device == self.device:
                    new_graph[key] = value.detach()
                else:
                    new_graph[key] = value.detach().to(self.device)
            elif key not in ['num_nodes', 'num_edges']:
                # Preserve non-tensor attributes (like coarsening_level, index)
                new_graph[key] = value
        
        # Set num_nodes from x tensor
        if hasattr(new_graph, 'x') and new_graph.x is not None:
            new_graph.num_nodes = new_graph.x.shape[0]
        
        return new_graph
    
    def step(self):
        """
        Increment the global step counter.
        
        Should be called once per training step (after processing fresh + replay batches).
        """
        self.global_step += 1
    
    def add(self, batch: Batch, replacement_prob: float = 1.0, 
            num_iterations: int = 0, previous_depths: Optional[List[int]] = None):
        """
        Add graphs from batch to replay buffer (stays on GPU).
        
        Args:
            batch: Batch of graphs to add (tensors should already be on GPU)
            replacement_prob: Probability of replacing existing items
            num_iterations: Number of GNN iterations applied in this forward pass
            previous_depths: Previous iteration depths for replay batches (None for fresh batches)
        """
        # Extract individual graphs - detach but keep on same device
        graphs = batch.detach().to_data_list()
        
        for i, graph in enumerate(graphs):
            # Only store with probability replacement_prob
            if random.random() > replacement_prob:
                continue
            
            self.items_added += 1
            
            # Ensure graph is on target device
            gpu_graph = self._ensure_on_device(graph)
            
            # Preserve metadata
            if hasattr(graph, 'coarsening_level'):
                gpu_graph.coarsening_level = graph.coarsening_level
            if hasattr(graph, 'index'):
                gpu_graph.index = graph.index
            
            # Calculate cumulative iteration depth
            if previous_depths is not None and i < len(previous_depths):
                new_depth = previous_depths[i] + num_iterations
            else:
                # Fresh batch: starts from 0 (encoder) + num_iterations
                new_depth = num_iterations
            
            # Store the graph
            if len(self.buffer) < self.capacity:
                self.buffer.append(gpu_graph)
                self.iteration_depths.append(new_depth)
                self.creation_steps.append(self.global_step)
            else:
                # Random replacement
                idx = random.randint(0, len(self.buffer) - 1)
                self.buffer[idx] = gpu_graph
                self.iteration_depths[idx] = new_depth
                self.creation_steps[idx] = self.global_step
                self.items_replaced += 1
    
    def sample_batch(self, device: Optional[torch.device] = None) -> Tuple[Batch, List[int], List[int]]:
        """
        Sample a batch of graphs from the replay buffer.
        
        Args:
            device: Target device (ignored - data is already on GPU)
            
        Returns:
            Tuple of:
            - Batch of graphs (all tensors on GPU)
            - List of iteration depths for sampled graphs
            - List of stalenesses (current_step - creation_step) for sampled graphs
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty GPU replay buffer")
        
        # Sample random indices
        num_to_sample = min(self.batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), num_to_sample)
        
        # Collect graphs and metadata
        graphs = []
        sampled_depths = []
        sampled_stalenesses = []
        
        for idx in indices:
            stored_graph = self.buffer[idx]
            
            # Create clean graph with only tensor attributes for batching
            graph = Data()
            for key in stored_graph.keys():
                value = stored_graph[key]
                if torch.is_tensor(value):
                    # Clone to avoid in-place modification issues
                    graph[key] = value.clone()
            
            # Preserve metadata
            if hasattr(stored_graph, 'coarsening_level'):
                graph.coarsening_level = stored_graph.coarsening_level
            if hasattr(stored_graph, 'index'):
                graph.index = stored_graph.index
            
            if hasattr(graph, 'x') and graph.x is not None:
                graph.num_nodes = graph.x.shape[0]
            
            graphs.append(graph)
            
            # Collect tracking metrics for this sample
            sampled_depths.append(self.iteration_depths[idx])
            staleness = self.global_step - self.creation_steps[idx]
            sampled_stalenesses.append(staleness)
        
        # Batch graphs (already on GPU, no device transfer needed)
        return Batch.from_data_list(graphs), sampled_depths, sampled_stalenesses
    
    def initialize_from_embeddings(self, embeddings_list: List[Data]):
        """
        Initialize replay buffer with initial embeddings.
        
        Initial embeddings have iteration_depth=0 (just encoded, no GNN iterations yet)
        and creation_step=0.
        
        Args:
            embeddings_list: List of graphs with encoded features
        """
        self.buffer = []
        self.iteration_depths = []
        self.creation_steps = []
        self.global_step = 0
        self.items_added = 0
        self.items_replaced = 0
        
        for graph in embeddings_list[:self.capacity]:
            # Ensure on target device
            gpu_graph = self._ensure_on_device(graph)
            
            # Preserve metadata
            if hasattr(graph, 'coarsening_level'):
                gpu_graph.coarsening_level = graph.coarsening_level
            if hasattr(graph, 'index'):
                gpu_graph.index = graph.index
            
            self.buffer.append(gpu_graph)
            # Initial embeddings: 0 iterations (just encoded), created at step 0
            self.iteration_depths.append(0)
            self.creation_steps.append(0)
            self.items_added += 1
    
    def get_stats_and_reset(self) -> dict:
        """
        Get current stats.
        
        Returns:
            Dictionary with stats
        """
        return {
            'items_added': self.items_added,
            'items_replaced': self.items_replaced,
            'gpu_resident': True,
        }
    
    def memory_usage_mb(self) -> float:
        """
        Estimate GPU memory usage of the buffer.
        
        Returns:
            Estimated memory usage in MB
        """
        total_bytes = 0
        for graph in self.buffer:
            for key in graph.keys():
                value = graph[key]
                if torch.is_tensor(value):
                    total_bytes += value.element_size() * value.nelement()
        return total_bytes / (1024 * 1024)
    
    def get_buffer_statistics(self) -> Dict[str, float]:
        """
        Compute statistics over the entire buffer for logging.
        
        Returns:
            Dictionary with statistics about iteration depths and staleness
            of all items currently in the buffer.
        """
        if len(self.buffer) == 0:
            return {
                'buffer_size': 0,
                'depth_mean': 0.0,
                'depth_std': 0.0,
                'depth_min': 0,
                'depth_max': 0,
                'staleness_mean': 0.0,
                'staleness_std': 0.0,
                'staleness_min': 0,
                'staleness_max': 0,
                'global_step': self.global_step,
                'memory_mb': 0.0,
            }
        
        # Compute staleness for all items
        stalenesses = [self.global_step - cs for cs in self.creation_steps]
        
        # Convert to numpy for statistics
        depths_arr = np.array(self.iteration_depths)
        stalenesses_arr = np.array(stalenesses)
        
        return {
            'buffer_size': len(self.buffer),
            'depth_mean': float(np.mean(depths_arr)),
            'depth_std': float(np.std(depths_arr)),
            'depth_min': int(np.min(depths_arr)),
            'depth_max': int(np.max(depths_arr)),
            'staleness_mean': float(np.mean(stalenesses_arr)),
            'staleness_std': float(np.std(stalenesses_arr)),
            'staleness_min': int(np.min(stalenesses_arr)),
            'staleness_max': int(np.max(stalenesses_arr)),
            'global_step': self.global_step,
            'memory_mb': self.memory_usage_mb(),
        }
    
    def get_sampled_batch_statistics(
        self, 
        sampled_depths: List[int], 
        sampled_stalenesses: List[int]
    ) -> Dict[str, float]:
        """
        Compute statistics for a sampled batch.
        
        Args:
            sampled_depths: Iteration depths of sampled graphs
            sampled_stalenesses: Stalenesses of sampled graphs
            
        Returns:
            Dictionary with statistics for the sampled batch
        """
        if len(sampled_depths) == 0:
            return {
                'sampled_depth_mean': 0.0,
                'sampled_depth_std': 0.0,
                'sampled_depth_min': 0,
                'sampled_depth_max': 0,
                'sampled_staleness_mean': 0.0,
                'sampled_staleness_std': 0.0,
                'sampled_staleness_min': 0,
                'sampled_staleness_max': 0,
            }
        
        depths_arr = np.array(sampled_depths)
        stalenesses_arr = np.array(sampled_stalenesses)
        
        return {
            'sampled_depth_mean': float(np.mean(depths_arr)),
            'sampled_depth_std': float(np.std(depths_arr)),
            'sampled_depth_min': int(np.min(depths_arr)),
            'sampled_depth_max': int(np.max(depths_arr)),
            'sampled_staleness_mean': float(np.mean(stalenesses_arr)),
            'sampled_staleness_std': float(np.std(stalenesses_arr)),
            'sampled_staleness_min': int(np.min(stalenesses_arr)),
            'sampled_staleness_max': int(np.max(stalenesses_arr)),
        }
    
    def get_depth_histogram(self, num_bins: int = 10) -> Tuple[List[int], List[int]]:
        """
        Get histogram of iteration depths in the buffer.
        
        Args:
            num_bins: Number of histogram bins
            
        Returns:
            Tuple of (bin_edges, counts) for the histogram
        """
        if len(self.iteration_depths) == 0:
            return [], []
        
        counts, bin_edges = np.histogram(self.iteration_depths, bins=num_bins)
        return bin_edges.tolist(), counts.tolist()
    
    def get_staleness_histogram(self, num_bins: int = 10) -> Tuple[List[int], List[int]]:
        """
        Get histogram of staleness values in the buffer.
        
        Args:
            num_bins: Number of histogram bins
            
        Returns:
            Tuple of (bin_edges, counts) for the histogram
        """
        if len(self.creation_steps) == 0:
            return [], []
        
        stalenesses = [self.global_step - cs for cs in self.creation_steps]
        counts, bin_edges = np.histogram(stalenesses, bins=num_bins)
        return bin_edges.tolist(), counts.tolist()
