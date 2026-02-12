"""
Node budget batch sampler for variable-sized graphs.

This module provides a custom batch sampler that forms batches based on
total node count rather than fixed graph count. This helps manage GPU
memory when training with graphs of highly variable sizes.
"""

import random
from typing import Iterator, List, Optional, Sized
from torch.utils.data import Sampler


class NodeBudgetBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups graphs by total node budget.
    
    Instead of having a fixed batch_size, this sampler fills batches until
    the total number of nodes exceeds max_nodes_per_batch (or optionally
    max_graphs_per_batch is reached).
    
    This is useful for:
    - Training on graphs with highly variable sizes
    - Preventing OOM errors when large graphs are batched together
    - More consistent memory usage across batches
    
    Args:
        data_source: Dataset to sample from. Must have a way to determine
                     graph sizes (via __getitem__ returning Data with num_nodes)
        max_nodes_per_batch: Maximum total nodes allowed in a batch
        max_graphs_per_batch: Maximum graphs per batch (None = no limit)
        min_batch_size: Minimum number of graphs per batch
        shuffle: Whether to shuffle indices before batching
        drop_last: Whether to drop the last incomplete batch
        get_num_nodes_fn: Optional function to get node count for an index
                          (for lazy datasets where __getitem__ is expensive)
    
    Example:
        >>> sampler = NodeBudgetBatchSampler(
        ...     data_source=dataset,
        ...     max_nodes_per_batch=50000,
        ...     max_graphs_per_batch=8,
        ...     shuffle=True
        ... )
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(
        self,
        data_source: Sized,
        max_nodes_per_batch: int = 10000,
        max_graphs_per_batch: Optional[int] = None,
        min_batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        get_num_nodes_fn=None,
    ):
        self.data_source = data_source
        self.max_nodes_per_batch = max_nodes_per_batch
        self.max_graphs_per_batch = max_graphs_per_batch
        self.min_batch_size = min_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.get_num_nodes_fn = get_num_nodes_fn
        
        # Cache node counts if possible (for efficient re-iteration)
        self._node_counts_cache = None
        
    def _get_num_nodes(self, idx: int) -> int:
        """Get the number of nodes for graph at index idx."""
        if self.get_num_nodes_fn is not None:
            return self.get_num_nodes_fn(idx)
        
        # Try to get from dataset directly
        data = self.data_source[idx]
        if hasattr(data, 'num_nodes'):
            return data.num_nodes
        elif hasattr(data, 'x'):
            return data.x.shape[0]
        else:
            raise ValueError(
                f"Cannot determine num_nodes for index {idx}. "
                "Provide get_num_nodes_fn or ensure dataset returns Data with num_nodes."
            )
    
    def _cache_node_counts(self) -> List[int]:
        """Cache node counts for all graphs (called once per epoch if shuffling)."""
        if self._node_counts_cache is None:
            # For dynamic datasets, we can't cache since sizes change each access
            # In that case, we estimate from subset_sizes if available
            if hasattr(self.data_source, 'subset_sizes') and hasattr(self.data_source, 'n_samples_per_size'):
                # This is a dynamic CLIP dataset - estimate sizes from config
                sizes = []
                for size in self.data_source.subset_sizes:
                    sizes.extend([size] * self.data_source.n_samples_per_size)
                self._node_counts_cache = sizes
            else:
                # Regular static dataset - cache actual sizes
                self._node_counts_cache = [
                    self._get_num_nodes(i) for i in range(len(self.data_source))
                ]
        return self._node_counts_cache
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches of indices."""
        n = len(self.data_source)
        
        # Create index list
        if self.shuffle:
            indices = list(range(n))
            random.shuffle(indices)
        else:
            indices = list(range(n))
        
        # Get node counts (from cache or dynamically)
        node_counts = self._cache_node_counts()
        
        # Form batches
        batches = []
        current_batch = []
        current_nodes = 0
        
        for idx in indices:
            # Get node count for this graph
            if len(node_counts) > idx:
                num_nodes = node_counts[idx]
            else:
                num_nodes = self._get_num_nodes(idx)
            
            # Check if adding this graph would exceed budget
            would_exceed_nodes = (current_nodes + num_nodes) > self.max_nodes_per_batch
            would_exceed_graphs = (
                self.max_graphs_per_batch is not None and 
                len(current_batch) >= self.max_graphs_per_batch
            )
            
            # If current batch is non-empty and adding would exceed limits
            if current_batch and (would_exceed_nodes or would_exceed_graphs):
                # Yield current batch if it meets minimum size
                if len(current_batch) >= self.min_batch_size:
                    batches.append(current_batch)
                current_batch = [idx]
                current_nodes = num_nodes
            else:
                # Add to current batch
                current_batch.append(idx)
                current_nodes += num_nodes
        
        # Handle last batch
        if current_batch:
            if not self.drop_last or len(current_batch) >= self.min_batch_size:
                batches.append(current_batch)
        
        # Yield batches
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        """
        Return approximate number of batches.
        
        Note: This is an approximation since batch sizes vary.
        The actual number may differ due to shuffling and dynamic graph sizes.
        """
        node_counts = self._cache_node_counts()
        total_nodes = sum(node_counts)
        
        # Estimate based on average batch filling
        estimated_batches = max(1, total_nodes // self.max_nodes_per_batch)
        
        # Account for max_graphs_per_batch constraint
        if self.max_graphs_per_batch is not None:
            max_possible_batches = len(self.data_source) // self.max_graphs_per_batch
            estimated_batches = max(estimated_batches, max_possible_batches)
        
        return estimated_batches


def create_node_budget_sampler_for_clip_dataset(
    dataset,
    max_nodes_per_batch: int,
    max_graphs_per_batch: Optional[int] = None,
    min_batch_size: int = 1,
    shuffle: bool = True,
) -> NodeBudgetBatchSampler:
    """
    Create a NodeBudgetBatchSampler optimized for CLIP dynamic datasets.
    
    For dynamic CLIP datasets, graph sizes are determined by subset_sizes config,
    and each graph is generated on-the-fly. This function creates a sampler
    that knows about the expected sizes without needing to materialize graphs.
    
    Args:
        dataset: CLIP dynamic dataset with subset_sizes and n_samples_per_size
        max_nodes_per_batch: Maximum total nodes per batch
        max_graphs_per_batch: Maximum graphs per batch (None = no limit)
        min_batch_size: Minimum graphs per batch
        shuffle: Whether to shuffle
        
    Returns:
        NodeBudgetBatchSampler configured for the dataset
    """
    def get_num_nodes_for_clip(idx: int) -> int:
        """Get expected node count based on subset_sizes cycling."""
        if hasattr(dataset, 'subset_sizes'):
            size_idx = idx % len(dataset.subset_sizes)
            return dataset.subset_sizes[size_idx]
        else:
            # Fallback to actually fetching (expensive)
            return dataset[idx].num_nodes
    
    return NodeBudgetBatchSampler(
        data_source=dataset,
        max_nodes_per_batch=max_nodes_per_batch,
        max_graphs_per_batch=max_graphs_per_batch,
        min_batch_size=min_batch_size,
        shuffle=shuffle,
        get_num_nodes_fn=get_num_nodes_for_clip,
    )
