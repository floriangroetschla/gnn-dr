"""
Compute edge weights for dimensionality reduction from directed kNN graphs.

Supports two weight computation methods:
- UMAP: Fuzzy simplicial set weights (default)
- t-SNE: Perplexity-calibrated symmetric affinities

Key assumptions:
- local_connectivity = 1.0 for UMAP (match your UMAP baseline to this value)
- Distances d_ij are computed under the same metric as the kNN graph (e.g., Euclidean or cosine)
"""

import logging
import torch
from typing import Tuple, Union, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UMAPWeightDebugInfo:
    """Debug information from UMAP weight computation."""
    rho: torch.Tensor           # [N] local connectivity (min neighbor distance)
    sigma: torch.Tensor         # [N] bandwidth parameters
    k_eff: torch.Tensor         # [N] effective k per node
    target: torch.Tensor        # [N] target connectivity = log2(k_eff)
    p_dir: torch.Tensor         # [E_sorted] directed memberships p_{i->j}
    p_rev: torch.Tensor         # [E_sorted] reverse memberships p_{j->i}
    w_dir: torch.Tensor         # [E_sorted] fuzzy union weights before coalescing
    src_sorted: torch.Tensor    # [E_sorted] source nodes (sorted order)
    dst_sorted: torch.Tensor    # [E_sorted] dest nodes (sorted order)
    d_sorted: torch.Tensor      # [E_sorted] distances (sorted order)


def compute_umap_fuzzy_weights(
    edge_index: torch.Tensor,   # [2, E] directed kNN edges (src, dst)
    d_ij: torch.Tensor,         # [E] distances for each directed edge (same metric as kNN)
    num_nodes: int,
    k: int,
    n_iters: int = 20,
    eps: float = 1e-6,
    return_debug: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, UMAPWeightDebugInfo]]:
    """
    Compute UMAP-style fuzzy simplicial set weights from a directed kNN graph.
    
    This implements the core weight computation algorithm from UMAP:
    1. Compute ρ_i (local connectivity: minimum neighbor distance per node)
    2. Solve for σ_i per node using binary search to match target connectivity log₂(k_eff_i)
    3. Compute directed memberships p_{i→j} = exp(-(d_ij - ρ_i) / σ_i)
    4. Symmetrize using fuzzy union: w_ij = p_ij + p_ji - p_ij * p_ji
    5. Return undirected edges with max aggregation for collisions

    Args:
        edge_index: [2, E] tensor where edge_index[0] = source nodes, edge_index[1] = target nodes
        d_ij: [E] tensor of distances for each directed edge, computed with the same metric
              used for kNN graph construction (e.g., Euclidean or cosine)
        num_nodes: Total number of nodes
        k: Target neighborhood size (controls connectivity target = log₂(k))
           Note: for nodes with fewer than k neighbors, effective_k = min(k, degree_i)
        n_iters: Number of binary search iterations for σ solving
        eps: Numerical stability epsilon

    Returns:
        edge_index_und: [2, E_und] undirected edge indices (u < v, coalesced, possibly converted to bidirectional)
        edge_weight: [E_und] fuzzy union weights in (0, 1]

    Reference:
        McInnes, L., Healy, J., & Melville, J. (2018).
        UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
        arXiv preprint arXiv:1802.03426.
    """
    device = d_ij.device
    src = edge_index[0].to(device)
    dst = edge_index[1].to(device)
    d = d_ij.to(device).float()

    # Sort edges by source for efficient processing
    order = torch.argsort(src)
    src_s = src[order]
    dst_s = dst[order]
    d_s = d[order]

    # Count outgoing edges per node
    counts = torch.bincount(src_s, minlength=num_nodes)  # [N]
    # Keep max_deg as tensor to avoid GPU→CPU sync (.item() forces sync!)
    max_deg_tensor = counts.max() if num_nodes > 0 else torch.tensor(0, device=device)
    K = max(k, 1)

    # Check for empty graph (only sync here if needed for error)
    if num_nodes > 0 and counts.sum().item() == 0:
        raise ValueError(f"No edges found. Cannot compute UMAP weights.")

    # Build padded neighbor-distance matrix D_out: [N, K]
    # Each row represents outgoing edges from that node
    # Pad with inf if node has fewer than K neighbors
    D_out = torch.full((num_nodes, K), float("inf"), device=device, dtype=d_s.dtype)

    # Compute position within each node's edge group
    cumsum = torch.cumsum(counts, dim=0)  # [N] cumulative end indices
    starts = torch.cat([torch.zeros(1, device=device, dtype=cumsum.dtype), cumsum[:-1]])
    idx_in_group = torch.arange(src_s.numel(), device=device) - starts[src_s].to(device)

    # Keep only first K edges per node
    mask_k = idx_in_group < K
    src_k = src_s[mask_k]
    d_k = d_s[mask_k]
    idx_k = idx_in_group[mask_k].long()

    D_out[src_k, idx_k] = d_k

    # ---- ρ_i: local connectivity (minimum distance) ----
    # This is UMAP's behavior with local_connectivity=1.0 (default)
    rho = torch.min(D_out, dim=1).values  # [N]
    rho = torch.where(torch.isfinite(rho), rho, torch.zeros_like(rho))

    # ---- σ_i: binary search to match target sum ----
    # Compute per-node effective k to handle nodes with < k neighbors
    # This ensures connectivity target is achievable for all nodes
    k_eff = torch.minimum(counts, torch.tensor(K, device=device, dtype=counts.dtype))  # [N]
    k_eff = torch.clamp(k_eff, min=1)
    target = torch.log2(k_eff.float())  # [N] per-node target connectivity
    
    # delta = max(0, d_ij - ρ_i)
    delta = torch.clamp(D_out - rho[:, None], min=0.0)

    # Initialize binary search bounds
    lo = torch.full((num_nodes,), eps, device=device, dtype=D_out.dtype)
    hi = torch.ones((num_nodes,), device=device, dtype=D_out.dtype)

    def sum_at(sigma_vec: torch.Tensor) -> torch.Tensor:
        """Compute sum of exponentials for a given sigma vector.
        
        Monotonicity: as sigma increases, exp(-delta/sigma) increases, so sum increases.
        """
        # exp(-inf) = 0, so padded entries with inf distances contribute 0
        s = torch.exp(-delta / (sigma_vec[:, None] + eps))
        return s.sum(dim=1)  # [N]

    # Expand upper bound until it brackets the target
    # Fixed iterations to avoid GPU→CPU sync from torch.any() check
    s_hi = sum_at(hi)
    for _ in range(10):  # Fixed iterations, no syncs!
        need_grow = s_hi < target
        hi = torch.where(need_grow, hi * 2.0, hi)
        s_hi = sum_at(hi)

    # Binary search for sigma
    # Monotonicity: larger sigma -> larger sum
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        s_mid = sum_at(mid)

        # If s_mid >= target, sigma is too large -> shrink upper bound
        # If s_mid < target, sigma is too small -> raise lower bound
        hi = torch.where(s_mid >= target, mid, hi)
        lo = torch.where(s_mid < target, mid, lo)

    sigma = hi  # [N]

    # ---- Directed memberships p_{i→j} ----
    delta_e = torch.clamp(d_s - rho[src_s], min=0.0)
    p = torch.exp(-delta_e / (sigma[src_s] + eps))
    p = torch.clamp(p, min=eps, max=1.0)

    # ---- Fuzzy union: w_{ij} = p_ij + p_ji - p_ij * p_ji ----
    # Create a searchable map of directed edges
    N = int(num_nodes)
    keys = src_s.long() * N + dst_s.long()
    sort_keys, perm = torch.sort(keys)
    p_sorted = p[perm]

    # Look up reverse edges
    rev_keys = dst_s.long() * N + src_s.long()
    pos = torch.searchsorted(sort_keys, rev_keys)
    # Clamp pos to valid range for indexing - values at numel() will fail the has check anyway
    pos_clamped = torch.clamp(pos, max=sort_keys.numel() - 1)
    has = (pos < sort_keys.numel()) & (sort_keys[pos_clamped] == rev_keys)
    p_rev = torch.zeros_like(p)
    p_rev[has] = p_sorted[pos_clamped[has]]

    # Fuzzy union formula
    w_dir = p + p_rev - p * p_rev
    w_dir = torch.clamp(w_dir, min=eps, max=1.0)

    # ---- Produce undirected edge list with coalescing (max) ----
    u = torch.minimum(src_s, dst_s).long()
    v = torch.maximum(src_s, dst_s).long()
    und_keys = u * N + v

    # Coalesce: take max weight for duplicate undirected edges
    uniq_keys, inv = torch.unique(und_keys, return_inverse=True)
    w_und = torch.full((uniq_keys.numel(),), 0.0, device=device, dtype=w_dir.dtype)

    if hasattr(w_und, "scatter_reduce_"):
        # PyTorch >= 2.0
        w_und.scatter_reduce_(0, inv, w_dir, reduce="amax", include_self=True)
    else:
        # Fallback for older PyTorch
        for i in range(w_dir.numel()):
            idx = int(inv[i])
            w_und[idx] = torch.maximum(w_und[idx], w_dir[i])

    edge_u = (uniq_keys // N).long()
    edge_v = (uniq_keys % N).long()

    # Remove self-loops
    mask = edge_u != edge_v
    edge_u = edge_u[mask]
    edge_v = edge_v[mask]
    w_und = w_und[mask]

    edge_index_und = torch.stack([edge_u, edge_v], dim=0)
    
    # Note: If downstream code expects bidirectional edges, convert here:
    # edge_index_bidir = torch.cat([edge_index_und, edge_index_und.flip(0)], dim=1)
    # w_bidir = torch.cat([w_und, w_und])
    # return edge_index_bidir, w_bidir
    
    if return_debug:
        debug_info = UMAPWeightDebugInfo(
            rho=rho,
            sigma=sigma,
            k_eff=k_eff,
            target=target,
            p_dir=p,
            p_rev=p_rev,
            w_dir=w_dir,
            src_sorted=src_s,
            dst_sorted=dst_s,
            d_sorted=d_s,
        )
        return edge_index_und, w_und, debug_info
    
    return edge_index_und, w_und


def compute_tsne_weights(
    edge_index: torch.Tensor,   # [2, E] directed kNN edges (src, dst)
    d_ij: torch.Tensor,         # [E] distances for each directed edge
    num_nodes: int,
    k: int,
    perplexity: float = 30.0,
    n_iters: int = 50,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute t-SNE-style perplexity-calibrated symmetric affinities from a directed kNN graph.

    Algorithm:
    1. Build padded squared-distance matrix D_sq [N, K] from directed kNN edges
    2. Binary search for β_i = 1/(2σ_i²) per node to match target entropy log(perplexity)
       - Conditional: p_{j|i} = exp(-β_i * d_ij²) / Σ_l exp(-β_i * d_il²)
       - Entropy: H_i = -Σ_j p_{j|i} log₂(p_{j|i})
       - Target: H_i = log₂(perplexity)
    3. Symmetrize: p_ij = (p_{j|i} + p_{i|j}) / (2n)
    4. Return undirected edges with symmetric weights (same format as compute_umap_fuzzy_weights)

    Args:
        edge_index: [2, E] directed kNN edges (src, dst)
        d_ij: [E] distances for each directed edge
        num_nodes: Total number of nodes
        k: Number of neighbors in the kNN graph
        perplexity: Target perplexity (typical: 5-50, default: 30)
        n_iters: Number of binary search iterations for β solving
        eps: Numerical stability epsilon

    Returns:
        edge_index_und: [2, E_und] undirected edge indices (u < v, coalesced)
        edge_weight: [E_und] symmetric affinities

    Reference:
        van der Maaten, L. & Hinton, G. (2008).
        Visualizing Data using t-SNE. JMLR, 9, 2579-2605.
    """
    # Validate perplexity vs k: max achievable perplexity with k neighbors is k
    if perplexity >= k:
        logger.warning(
            f"t-SNE perplexity ({perplexity}) >= k ({k}). "
            f"Max achievable entropy with {k} neighbors is log2({k})={torch.log2(torch.tensor(float(k))).item():.2f} bits, "
            f"but target is log2({perplexity})={torch.log2(torch.tensor(perplexity)).item():.2f} bits. "
            f"Clamping effective perplexity to {k - 1}. Consider using perplexity < k."
        )
        perplexity = min(perplexity, k - 1.0)

    device = d_ij.device
    src = edge_index[0].to(device)
    dst = edge_index[1].to(device)
    d = d_ij.to(device).float()

    # Sort edges by source for efficient processing
    order = torch.argsort(src)
    src_s = src[order]
    dst_s = dst[order]
    d_s = d[order]

    # Count outgoing edges per node
    counts = torch.bincount(src_s, minlength=num_nodes)  # [N]
    K = max(k, 1)

    if num_nodes > 0 and counts.sum().item() == 0:
        raise ValueError("No edges found. Cannot compute t-SNE weights.")

    # Build padded squared-distance matrix D_sq: [N, K]
    D_sq = torch.full((num_nodes, K), float("inf"), device=device, dtype=d_s.dtype)

    cumsum = torch.cumsum(counts, dim=0)
    starts = torch.cat([torch.zeros(1, device=device, dtype=cumsum.dtype), cumsum[:-1]])
    idx_in_group = torch.arange(src_s.numel(), device=device) - starts[src_s].to(device)

    mask_k = idx_in_group < K
    src_k = src_s[mask_k]
    d_k = d_s[mask_k]
    idx_k = idx_in_group[mask_k].long()

    # Store squared distances (t-SNE uses d² in the Gaussian kernel)
    D_sq[src_k, idx_k] = d_k * d_k

    # Target entropy: log2(perplexity)
    target_H = torch.tensor(torch.log2(torch.tensor(perplexity)), device=device, dtype=D_sq.dtype)

    # ---- Binary search for β_i = 1/(2σ_i²) ----
    # p_{j|i} = exp(-β_i * d_ij²) / Z_i, where Z_i = Σ_l exp(-β_i * d_il²)
    # Entropy H_i = log2(Z_i) + β_i * Σ_j p_{j|i} * d_ij² / ln(2)
    # Monotonicity: as β increases, distribution sharpens, entropy decreases

    lo = torch.full((num_nodes,), eps, device=device, dtype=D_sq.dtype)
    hi = torch.ones((num_nodes,), device=device, dtype=D_sq.dtype)

    def entropy_at(beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute entropy and conditional probs for given beta values.
        Returns (entropy [N], conditional_probs [N, K]).
        """
        # exp(-beta * d²), with inf -> 0
        logits = -beta[:, None] * D_sq  # [N, K]
        # For numerical stability, subtract max per row
        logits_max = logits.max(dim=1, keepdim=True).values
        logits_max = torch.where(torch.isfinite(logits_max), logits_max, torch.zeros_like(logits_max))
        exp_val = torch.exp(logits - logits_max)  # [N, K]
        # Zero out inf-distance entries
        exp_val = torch.where(torch.isfinite(D_sq), exp_val, torch.zeros_like(exp_val))

        Z = exp_val.sum(dim=1, keepdim=True) + eps  # [N, 1]
        p_cond = exp_val / Z  # [N, K] conditional probabilities

        # Entropy in bits: H = -Σ p log2(p) = log2(Z) - logits_max/ln2 + β * Σ p*d² / ln2
        # Simpler: H = -Σ p * log2(p)
        log2_p = torch.log2(p_cond + eps)
        H = -(p_cond * log2_p).sum(dim=1)  # [N]
        # Zero out contribution from padded entries (p=0 for inf distances)

        return H, p_cond

    # Expand upper bound until it brackets the target
    H_hi, _ = entropy_at(hi)
    for _ in range(20):
        # If entropy is too high (distribution too uniform), beta is too small -> increase hi
        # If entropy is too low, beta is too large -> but hi is the upper bound for beta, so we grow it
        # Actually: higher beta = sharper distribution = lower entropy
        # We want H ≈ target_H. If H_hi > target_H, beta is too small (entropy too high).
        need_grow = H_hi > target_H
        hi = torch.where(need_grow, hi * 2.0, hi)
        H_hi, _ = entropy_at(hi)

    # Also ensure lo is small enough (entropy at lo should be >= target)
    H_lo, _ = entropy_at(lo)
    for _ in range(10):
        need_shrink = H_lo < target_H
        lo = torch.where(need_shrink, lo * 0.5, lo)
        H_lo, _ = entropy_at(lo)

    # Binary search: find beta where entropy = target_H
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        H_mid, _ = entropy_at(mid)

        # If H_mid > target_H, entropy is too high -> beta too small -> raise lower bound
        # If H_mid < target_H, entropy is too low -> beta too large -> lower upper bound
        lo = torch.where(H_mid > target_H, mid, lo)
        hi = torch.where(H_mid <= target_H, mid, hi)

    beta = 0.5 * (lo + hi)  # [N]

    # ---- Compute final conditional probabilities p_{j|i} ----
    _, p_cond = entropy_at(beta)  # [N, K]

    # ---- Map conditional probs back to edge list ----
    # p_cond[i, j_idx] is p_{j|i} for the j_idx-th neighbor of node i
    p_dir = torch.zeros(src_s.numel(), device=device, dtype=d_s.dtype)
    p_dir[mask_k] = p_cond[src_k, idx_k]

    # ---- Symmetrize: p_ij = (p_{j|i} + p_{i|j}) / (2n) ----
    N = int(num_nodes)
    keys = src_s.long() * N + dst_s.long()
    sort_keys, perm = torch.sort(keys)
    p_sorted = p_dir[perm]

    # Look up reverse edges
    rev_keys = dst_s.long() * N + src_s.long()
    pos = torch.searchsorted(sort_keys, rev_keys)
    pos_clamped = torch.clamp(pos, max=sort_keys.numel() - 1)
    has = (pos < sort_keys.numel()) & (sort_keys[pos_clamped] == rev_keys)
    p_rev = torch.zeros_like(p_dir)
    p_rev[has] = p_sorted[pos_clamped[has]]

    # Symmetric affinity
    w_dir = (p_dir + p_rev) / (2.0 * N)
    w_dir = torch.clamp(w_dir, min=eps)

    # ---- Produce undirected edge list with coalescing (max) ----
    u = torch.minimum(src_s, dst_s).long()
    v = torch.maximum(src_s, dst_s).long()
    und_keys = u * N + v

    uniq_keys, inv = torch.unique(und_keys, return_inverse=True)
    w_und = torch.full((uniq_keys.numel(),), 0.0, device=device, dtype=w_dir.dtype)

    if hasattr(w_und, "scatter_reduce_"):
        w_und.scatter_reduce_(0, inv, w_dir, reduce="amax", include_self=True)
    else:
        for i in range(w_dir.numel()):
            idx = int(inv[i])
            w_und[idx] = torch.maximum(w_und[idx], w_dir[i])

    edge_u = (uniq_keys // N).long()
    edge_v = (uniq_keys % N).long()

    # Remove self-loops
    mask = edge_u != edge_v
    edge_u = edge_u[mask]
    edge_v = edge_v[mask]
    w_und = w_und[mask]

    edge_index_und = torch.stack([edge_u, edge_v], dim=0)

    return edge_index_und, w_und


def compute_edge_weights(
    method: str,
    edge_index: torch.Tensor,
    d_ij: torch.Tensor,
    num_nodes: int,
    k: int,
    perplexity: float = 30.0,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatch edge weight computation to the appropriate method.

    Args:
        method: 'umap' (fuzzy simplicial set) or 'tsne' (perplexity-based)
        edge_index: [2, E] directed kNN edges
        d_ij: [E] distances
        num_nodes: Total number of nodes
        k: Number of neighbors
        perplexity: t-SNE perplexity (only used when method='tsne')
        **kwargs: Additional arguments passed to the weight function

    Returns:
        edge_index_und: [2, E_und] undirected edges
        edge_weight: [E_und] edge weights
    """
    if method == 'tsne':
        return compute_tsne_weights(
            edge_index=edge_index,
            d_ij=d_ij,
            num_nodes=num_nodes,
            k=k,
            perplexity=perplexity,
        )
    else:  # 'umap' (default)
        return compute_umap_fuzzy_weights(
            edge_index=edge_index,
            d_ij=d_ij,
            num_nodes=num_nodes,
            k=k,
        )


def log_weight_statistics(edge_weight: torch.Tensor, prefix: str = ""):
    """
    Log histogram and statistics of edge weights.
    
    Args:
        edge_weight: [E] tensor of edge weights
        prefix: string prefix for logging
    """
    import numpy as np
    
    weights_np = edge_weight.cpu().numpy()
    
    print(f"\n{prefix}Weight Statistics:")
    print(f"  Count:     {len(weights_np)}")
    print(f"  Mean:      {weights_np.mean():.6f}")
    print(f"  Median:    {np.median(weights_np):.6f}")
    print(f"  Std:       {weights_np.std():.6f}")
    print(f"  Min:       {weights_np.min():.6f}")
    print(f"  Max:       {weights_np.max():.6f}")
    
    # Histogram
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(weights_np, bins=bins)
    
    print(f"\n{prefix}Weight Distribution:")
    for i in range(len(bins)-1):
        pct = 100 * hist[i] / len(weights_np)
        bar = '█' * int(pct / 2)
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]: {hist[i]:6d} ({pct:5.1f}%) {bar}")
