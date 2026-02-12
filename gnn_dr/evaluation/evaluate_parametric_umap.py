"""
Evaluate Parametric UMAP trained on the same datasets as CoRe-DR.

Trains a single Parametric UMAP encoder on the combined training split of the
3 GNN training datasets (MNIST, Fashion-MNIST, CIFAR-10), then evaluates it
on all evaluation datasets (including held-out ones never seen during training).

This mirrors how CoRe-DR is evaluated: one model, trained once, then applied
to unseen data via forward pass.

Usage:
    python -m gnn_dr.evaluation.evaluate_parametric_umap
    python -m gnn_dr.evaluation.evaluate_parametric_umap --output results/pumap.csv
    python -m gnn_dr.evaluation.evaluate_parametric_umap --device cuda --max-samples 10000
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

from gnn_dr.evaluation.run_full_evaluation import (
    load_dataset_embeddings,
    compute_tfpqm_metrics,
)

# Same training datasets as CoRe-DR (default.yaml)
TRAIN_DATASETS = [
    'mnist_clip',
    'fashion_mnist_clip',
    'cifar10_clip',
]

# Same evaluation datasets as CoRe-DR (default.yaml)
EVAL_DATASETS = [
    'mnist_clip',
    'cifar10_clip',
    'fashion_mnist_clip',
    'kmnist_clip',
    'fgvc_aircraft_clip',
    'oxford_pets_clip',
    'food101_clip',
]


def load_training_data(max_samples_per_dataset: int = 10000, device: str = 'cpu'):
    """Load and concatenate training embeddings from all training datasets."""
    all_embeddings = []

    for ds_name in TRAIN_DATASETS:
        print(f"  Loading {ds_name} (train split)...")
        embeddings, labels, n = load_dataset_embeddings(
            ds_name, split='train', max_samples=max_samples_per_dataset, device=device,
        )
        if embeddings is None:
            print(f"    WARNING: Failed to load {ds_name}, skipping")
            continue
        all_embeddings.append(embeddings)
        print(f"    Loaded {n} samples, dim={embeddings.shape[1]}")

    combined = np.concatenate(all_embeddings, axis=0)
    print(f"  Combined training data: {combined.shape[0]} samples, {combined.shape[1]} dims")
    return combined


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Parametric UMAP (trained like CoRe-DR)')
    parser.add_argument('--output', default='results/pumap.csv',
                        help='Output CSV path')
    parser.add_argument('--checkpoint', default='models/pumap.pt',
                        help='Path to save/load model checkpoint')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training, load from --checkpoint instead')
    parser.add_argument('--device', default='cuda',
                        help='Device for Parametric UMAP (cuda or cpu)')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='Max samples per dataset for evaluation')
    parser.add_argument('--n-neighbors', type=int, default=15,
                        help='Number of neighbors (matching CoRe-DR k=15)')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Training epochs for Parametric UMAP')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension of encoder network')
    parser.add_argument('--n-layers', type=int, default=3,
                        help='Number of hidden layers in encoder')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate for encoder training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for encoder training')
    parser.add_argument('--k', type=int, default=15,
                        help='Number of neighbors for metrics')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    import torch
    from parametric_umap import ParametricUMAP

    np.random.seed(args.seed)
    checkpoint_path = Path(args.checkpoint)

    if args.eval_only:
        # --- Load existing checkpoint ---
        print("=" * 60)
        print("STEP 1: Loading trained model from checkpoint")
        print("=" * 60)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model = ckpt['model']
        train_time = ckpt.get('train_time', 0.0)
        print(f"  Loaded from {checkpoint_path} (trained in {train_time:.2f}s)")
    else:
        # --- 1. Load training data ---
        print("=" * 60)
        print("STEP 1: Loading training data")
        print("=" * 60)
        train_embeddings = load_training_data(
            max_samples_per_dataset=args.max_samples, device=args.device,
        )

        # --- 2. Train Parametric UMAP ---
        print("\n" + "=" * 60)
        print("STEP 2: Training Parametric UMAP")
        print("=" * 60)
        print(f"  n_neighbors={args.n_neighbors}, n_epochs={args.n_epochs}, "
              f"hidden_dim={args.hidden_dim}, n_layers={args.n_layers}, "
              f"lr={args.learning_rate}, batch_size={args.batch_size}")

        model = ParametricUMAP(
            n_neighbors=args.n_neighbors,
            n_components=2,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            device=args.device,
        )

        train_start = time.time()
        model.fit(train_embeddings)
        train_time = time.time() - train_start
        print(f"  Training completed in {train_time:.2f}s")

        # Save checkpoint
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model': model,
            'train_time': train_time,
            'args': vars(args),
        }, checkpoint_path)
        print(f"  Checkpoint saved to {checkpoint_path}")

    # --- 3. Evaluate on all datasets ---
    print("\n" + "=" * 60)
    print("STEP 3: Evaluating on all datasets")
    print("=" * 60)

    results = []

    for ds_name in EVAL_DATASETS:
        print(f"\n  Evaluating {ds_name}...")

        # Load test split
        embeddings, labels, n_samples = load_dataset_embeddings(
            ds_name, split='test', max_samples=args.max_samples, device=args.device,
        )
        if embeddings is None:
            print(f"    WARNING: Failed to load {ds_name}, skipping")
            continue

        n_classes = len(np.unique(labels)) if labels is not None else 0
        print(f"    {n_samples} samples, {n_classes} classes")

        # Inference (timed)
        infer_start = time.time()
        projection = model.transform(embeddings)
        inference_time = time.time() - infer_start

        # Check for NaN
        n_nan = np.isnan(projection).any(axis=1).sum()
        if n_nan > 0:
            print(f"    WARNING: {n_nan}/{n_samples} samples have NaN projections")
            continue

        print(f"    Inference: {inference_time:.4f}s")

        # Compute metrics
        metrics = compute_tfpqm_metrics(embeddings, projection, labels, k=args.k)

        row = {
            'dataset': ds_name,
            'method': 'Parametric_UMAP',
            'n_samples': n_samples,
            'n_classes': n_classes,
            'time_seconds': inference_time,
            'inference_time': inference_time,
            'train_time': train_time,
        }
        row.update(metrics)
        results.append(row)

        print(f"    Trustworthiness: {metrics.get('trustworthiness', 'N/A'):.4f}, "
              f"NH: {metrics.get('neighborhood_hit', 'N/A'):.4f}, "
              f"Shepard: {metrics.get('shepard_goodness', 'N/A'):.4f}")

    # --- 4. Save results ---
    print("\n" + "=" * 60)
    print("STEP 4: Saving results")
    print("=" * 60)

    df = pd.DataFrame(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"  Results saved to {args.output}")
    print(f"\n{df.to_string()}")


if __name__ == '__main__':
    main()
