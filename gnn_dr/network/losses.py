"""Loss functions for dimensionality reduction."""

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import curve_fit
from torch_geometric.utils import batched_negative_sampling, remove_self_loops, coalesce


def find_ab_params(spread: float, min_dist: float):
    """
    Fit a, b params for the differentiable curve used in UMAP.
    This is a direct port of umap.umap_.find_ab_params(spread, min_dist).

    Args:
        spread: UMAP spread parameter
        min_dist: UMAP min_dist parameter

    Returns:
        tuple: (a, b) parameters for the curve 1 / (1 + a * x^(2b))
    """
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros_like(xv)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)

    params, _ = curve_fit(curve, xv, yv)
    return float(params[0]), float(params[1])


def pairwise_distance(x: torch.Tensor,
                      y: torch.Tensor,
                      metric: str = "euclidean",
                      eps: float = 1e-8) -> torch.Tensor:
    """
    Compute pairwise distances between rows of x and y.

    Args:
        x, y: Tensors of shape (N, d), matched row-wise.
        metric: "euclidean" or "cosine".
        eps: Numerical stability epsilon.

    Returns:
        Distances of shape (N,).
    """
    if metric == "euclidean":
        return torch.norm(x - y, p=2, dim=1)

    elif metric == "cosine":
        # 1 - cosine similarity
        x_norm = x / (x.norm(dim=1, keepdim=True) + eps)
        y_norm = y / (y.norm(dim=1, keepdim=True) + eps)
        cos_sim = (x_norm * y_norm).sum(dim=1)
        return 1.0 - cos_sim.clamp(-1.0, 1.0)

    else:
        raise ValueError(f"Unsupported metric: {metric}")


class UMAPLoss(nn.Module):
    """
    UMAP-style loss for dimensionality reduction.

    Supports two weighting strategies controlled by `use_weighted_loss`:

    1. Sampling-based approximation (use_weighted_loss=False, default):
       - Positive edges sampled proportional to fuzzy simplicial weights
       - Negative edges sampled uniformly (excluding positive edges and their reverses)
       - Attraction and repulsion computed as simple means
       - Balance enforced via sampling frequency, NOT loss normalization
       - Key principle: "Sampling IS the normalization"

    2. Direct weighted loss (use_weighted_loss=True):
       - Positive edges sampled uniformly
       - Loss is weighted by edge weights: L = -Σ w_ij · log(q_ij) / Σ w_ij
       - Closer to standard UMAP's cross-entropy objective
       - Recommended for comparison with vanilla UMAP

    Args:
        num_positive_samples: Number of positive edges to sample per step (default: 1000)
        num_negatives_per_edge: Number of negatives per sampled positive (default: 5)
        repulsion_weight: Weight for repulsion term (gamma, default: 1.0)
        min_dist: UMAP min_dist parameter for a,b curve fitting (default: 0.1)
        spread: UMAP spread parameter for a,b curve fitting (default: 1.0)
        a, b: Direct curve parameters (if None, computed from min_dist/spread)
        metric: Distance metric in embedding space ("euclidean" or "cosine")
        use_weighted_loss: If True, use direct weighted loss; if False, use sampling-based (default: False)
    """
    def __init__(
        self,
        num_positive_samples: int = 1000,
        num_negatives_per_edge: int = 5,
        repulsion_weight: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float = None,
        b: float = None,
        metric: str = "euclidean",
        use_weighted_loss: bool = False,
        # Legacy parameters (kept for backward compatibility, some ignored)
        edge_sample_rate: float = 1.0,
        gauge_fix: bool = False,
        embedding_reg_weight: float = 0.0,
        neg_sample_rate: float = 5.0,
        k_neighbors: int = 15,
        reduce=None,
        weight_metric: str = "euclidean",
        weight_eps: float = 1e-6,
        weight_min: float = 1e-6,
    ):
        super().__init__()
        self.num_positive_samples = num_positive_samples
        self.num_negatives_per_edge = num_negatives_per_edge
        self.repulsion_weight = repulsion_weight
        self.min_dist = min_dist
        self.spread = spread
        self.metric = metric
        self.gauge_fix = gauge_fix
        self.use_weighted_loss = use_weighted_loss

        # Compute a, b parameters from min_dist/spread if not provided
        if a is None or b is None:
            a_val, b_val = find_ab_params(spread, min_dist)
        else:
            a_val, b_val = a, b

        self.register_buffer("a", torch.tensor(a_val, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(b_val, dtype=torch.float32))

    def forward(self, node_pos, batch, return_components: bool = False):
        """
        Compute UMAP loss with one of two weighting strategies.

        If use_weighted_loss=False (default, sampling-based approximation):
            1. Sample P positive edges proportional to their fuzzy simplicial weights
            2. Sample n_neg negatives per positive (excluding neighbors & reverse edges)
            3. Compute attraction and repulsion as simple means
            4. Balance via sampling frequency, not loss normalization

        If use_weighted_loss=True (direct weighted loss, closer to standard UMAP):
            1. Sample P positive edges uniformly
            2. Compute weighted attraction: L = -Σ w_ij · log(q_ij) / Σ w_ij
            3. Sample negatives and compute repulsion as simple mean
            4. This is an unbiased estimator of the true UMAP cross-entropy objective

        Args:
            node_pos: Predicted node positions (N, d)
            batch: PyG batch object with edge_index and edge_weight
            return_components: If True, return dict with loss components

        Returns:
            Total loss (or dict with 'total', 'attraction', 'repulsion')
        """
        edge_index = batch.edge_index
        n_nodes = node_pos.shape[0]
        device = node_pos.device

        # Use embeddings directly (no gauge-fixing - it distorts distances!)
        z = node_pos

        # Require pre-computed UMAP weights from dataset
        if not hasattr(batch, 'edge_weight') or batch.edge_weight is None:
            raise ValueError(
                "UMAPLoss requires pre-computed edge weights in batch.edge_weight. "
                "Ensure your dataset computes UMAP fuzzy simplicial set weights "
                "using compute_umap_fuzzy_weights() from gnn_dr.utils.umap_weights."
            )

        edge_weight = batch.edge_weight.to(device)

        # Validate alignment
        if edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"Edge weight/index mismatch: {edge_weight.shape[0]} weights "
                f"for {edge_index.shape[1]} edges."
            )

        # ── Positive sampling ────────────────────────────────────────────
        n_edges = edge_index.shape[1]
        P = min(self.num_positive_samples, n_edges)

        # Clamp weights (requires non-negative)
        w = torch.clamp(edge_weight, min=1e-12)

        if self.use_weighted_loss:
            # WEIGHTED LOSS MODE: Sample edges uniformly, weight in loss
            pos_idx = torch.randperm(n_edges, device=device)[:P]
        else:
            # SAMPLING MODE: Sample edges proportional to weights
            pos_idx = torch.multinomial(w, num_samples=P, replacement=True)

        pos_edges = edge_index[:, pos_idx]
        pos_weights = w[pos_idx]

        # ── Attraction loss ──────────────────────────────────────────────
        pos_i = z[pos_edges[0]]
        pos_j = z[pos_edges[1]]
        d_pos = pairwise_distance(pos_i, pos_j, metric=self.metric)

        # UMAP's smooth approximation: q = 1 / (1 + a * d^(2b))
        q_pos = 1.0 / (1.0 + self.a * d_pos.pow(2 * self.b))
        q_pos = torch.clamp(q_pos, min=1e-4, max=1.0)

        if self.use_weighted_loss:
            # WEIGHTED LOSS: -Σ w_ij · log(q_ij) / Σ w_ij
            weighted_log_q = pos_weights * (-torch.log(q_pos))
            attraction_loss = weighted_log_q.sum() / (pos_weights.sum() + 1e-12)
        else:
            # SAMPLING MODE: Simple mean (sampling provides the weighting)
            attraction_loss = -torch.log(q_pos).mean()

        # ── Negative sampling ────────────────────────────────────────────
        # Must exclude: positive edges, reverse edges, self-loops
        if hasattr(batch, "batch") and batch.batch is not None:
            batch_vec = batch.batch.to(device)
        else:
            batch_vec = torch.zeros(n_nodes, dtype=torch.long, device=device)

        n_neg_total = P * self.num_negatives_per_edge

        # Build exclusion set: both (i,j) AND (j,i) to prevent reverse edges
        edge_index_bidir = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_index_bidir, _ = remove_self_loops(edge_index_bidir)
        edge_index_bidir = coalesce(edge_index_bidir, None, n_nodes, n_nodes)[0]

        neg_edge_index = batched_negative_sampling(
            edge_index=edge_index_bidir,
            batch=batch_vec,
            num_neg_samples=n_neg_total,
            method="sparse",
        )

        # ── Repulsion loss ───────────────────────────────────────────────
        if neg_edge_index.shape[1] > 0:
            neg_i = z[neg_edge_index[0]]
            neg_j = z[neg_edge_index[1]]
            d_neg = pairwise_distance(neg_i, neg_j, metric=self.metric)

            q_neg = 1.0 / (1.0 + self.a * d_neg.pow(2 * self.b))
            q_neg = torch.clamp(q_neg, min=0.0, max=1.0 - 1e-4)

            repulsion_loss = -torch.log(1.0 - q_neg).mean()
        else:
            repulsion_loss = z.new_tensor(0.0)

        # ── Total loss ───────────────────────────────────────────────────
        total_loss = attraction_loss + self.repulsion_weight * repulsion_loss

        if return_components:
            return {
                "total": total_loss.item(),
                "attraction": attraction_loss.item(),
                "repulsion": repulsion_loss.item(),
            }

        return total_loss
