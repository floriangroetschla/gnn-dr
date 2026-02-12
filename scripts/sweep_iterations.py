"""
Sweep over GNN iteration counts and evaluate metrics at each step.

Usage:
    python scripts/sweep_iterations.py \
        --checkpoint models/last.ckpt \
        --config configs/default.yaml \
        --output results/iteration_sweep.csv
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_dr.evaluation.evaluate_model import (
    load_model_from_checkpoint,
    load_dataset_for_model,
    build_knn_graph,
    preprocess_for_inference,
    run_model_inference,
)
from gnn_dr.evaluation.run_full_evaluation import compute_tfpqm_metrics


def main():
    parser = argparse.ArgumentParser(description='Sweep GNN iterations')
    parser.add_argument('--checkpoint', '-c', required=True,
                        help='Path to Lightning checkpoint')
    parser.add_argument('--config', default=None,
                        help='Path to config YAML')
    parser.add_argument('--output', '-o', default='results/iteration_sweep.csv',
                        help='Output CSV path')
    parser.add_argument('--max-iterations', type=int, default=20,
                        help='Maximum number of iterations to evaluate')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='Maximum samples per dataset')
    parser.add_argument('--k', type=int, default=15,
                        help='Number of KNN neighbors')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to evaluate (default: all)')
    args = parser.parse_args()

    default_datasets = [
        'mnist_clip', 'fashion_mnist_clip', 'cifar10_clip',
        'kmnist_clip', 'fgvc_aircraft_clip', 'oxford_pets_clip', 'food101_clip',
    ]
    datasets = args.datasets or default_datasets

    print(f"Loading model from: {args.checkpoint}")
    model, config = load_model_from_checkpoint(
        args.checkpoint, config_path=args.config, device=args.device
    )

    iterations_range = list(range(1, args.max_iterations + 1))
    all_results = []

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        embeddings, labels, actual_size = load_dataset_for_model(
            dataset_name, split='test', max_samples=args.max_samples,
            device=args.device, random_state=42,
        )
        if embeddings is None:
            print(f"  Skipping {dataset_name} (failed to load)")
            continue

        print(f"  Loaded {actual_size} samples")

        # Build graph once (shared across all iteration counts)
        graph = build_knn_graph(embeddings, labels, k=args.k, device=args.device)
        graph = preprocess_for_inference(graph, config)

        embeddings_f32 = embeddings.astype(np.float32)

        for T in iterations_range:
            start = time.time()
            projection = run_model_inference(model, graph, rounds=T)
            elapsed = time.time() - start

            projection_f32 = projection.astype(np.float32)
            metrics = compute_tfpqm_metrics(embeddings_f32, projection_f32, labels, k=args.k)

            row = {
                'dataset': dataset_name,
                'iterations': T,
                'inference_time': elapsed,
                'n_samples': actual_size,
            }
            row.update(metrics)
            all_results.append(row)

            print(f"  T={T:2d}: trust={metrics.get('trustworthiness', 0):.4f}  "
                  f"cont={metrics.get('continuity', 0):.4f}  "
                  f"NH={metrics.get('neighborhood_hit', 0):.4f}  "
                  f"shep={metrics.get('shepard_goodness', 0):.4f}  "
                  f"SNS={metrics.get('scale_normalized_stress', 0):.4f}  "
                  f"({elapsed:.2f}s)")

        # Cleanup
        del embeddings, labels, graph
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
