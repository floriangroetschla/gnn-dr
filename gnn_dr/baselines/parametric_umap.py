"""Parametric UMAP baseline for dimensionality reduction evaluation.

Uses the PyTorch parametric_umap package (https://github.com/fcarli/parametric_umap).
Install with: pip install parametric_umap
"""

import numpy as np
import time
from typing import Optional

from .base import NeuralBaselineMethod


class ParametricUMAPBaseline(NeuralBaselineMethod):
    """
    Parametric UMAP baseline using a PyTorch encoder network.

    Learns a neural network to approximate the UMAP embedding, making it
    inductive (can project unseen data without re-fitting).

    Reference: https://github.com/fcarli/parametric_umap
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        n_components: int = 2,
        n_epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 3,
        random_state: Optional[int] = None,
        device: str = 'cpu',
    ):
        super().__init__(name='Parametric UMAP', random_state=random_state, device=device)

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.model = None

        try:
            from parametric_umap import ParametricUMAP
            self._ParametricUMAP = ParametricUMAP
        except ImportError:
            raise ImportError(
                "Parametric UMAP requires the parametric_umap package. "
                "Install with: pip install parametric_umap"
            )

    def fit(self, embeddings: np.ndarray) -> None:
        """Fit parametric UMAP on high-dimensional embeddings."""
        start_time = time.time()

        self.model = self._ParametricUMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            device=self.device,
        )

        self.model.fit(embeddings)

        self._train_time = time.time() - start_time
        self.is_fitted = True

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings using the learned encoder network."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before transform")

        start_time = time.time()
        result = self.model.transform(embeddings)
        self._inference_time = time.time() - start_time

        if np.isnan(result).any():
            n_nan = np.isnan(result).any(axis=1).sum()
            raise RuntimeError(
                f"Parametric UMAP produced NaN for {n_nan}/{len(result)} samples."
            )

        return result

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(embeddings)
        return self.transform(embeddings)

    def get_model_size(self) -> int:
        """Get number of trainable parameters in the encoder."""
        if self.model is not None and hasattr(self.model, 'parameters'):
            try:
                return sum(p.numel() for p in self.model.parameters())
            except Exception:
                return 0
        return 0
