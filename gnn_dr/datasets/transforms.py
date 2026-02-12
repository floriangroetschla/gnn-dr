import torch


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Normalize embeddings to unit length (L2 normalization).

    Converts embeddings to unit vectors with norm = 1.
    Useful for CLIP embeddings and other normalized representation spaces.

    Args:
        embeddings: Tensor of shape (N, D) or any shape with embeddings on last dim

    Returns:
        Normalized embeddings of same shape and dtype
    """
    # Compute L2 norm along last dimension
    norms = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
    # Avoid division by zero
    norms = torch.clamp(norms, min=1e-12)
    # Normalize
    return embeddings / norms
