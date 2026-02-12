"""
Plot embedding quality vs. number of GNN iterations.

Usage:
    python scripts/plot_iterations.py
    python scripts/plot_iterations.py --input results/iteration_sweep.csv
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATASET_SHORT = {
    'mnist_clip': 'MNIST',
    'cifar10_clip': 'CIFAR-10',
    'fashion_mnist_clip': 'F-MNIST',
    'kmnist_clip': 'KMNIST',
    'flowers102_clip': 'Flowers',
    'fgvc_aircraft_clip': 'Aircraft',
    'oxford_pets_clip': 'Pets',
    'food101_clip': 'Food-101',
}

TRAINING_DATASETS = {'mnist_clip', 'fashion_mnist_clip', 'cifar10_clip'}

# Metrics to plot
METRICS = {
    'Trustworthiness': ('trustworthiness', True),     # (col_name, higher_is_better)
    'Continuity': ('continuity', True),
    'Neighborhood Hit': ('neighborhood_hit', True),
    'Shepard Goodness': ('shepard_goodness', True),
    'Scale-Norm. Stress': ('scale_normalized_stress', False),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='results/iteration_sweep.csv')
    parser.add_argument('--output', '-o', default='figures/iteration_sweep.pdf')
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Compute mean across datasets, separately for training and held-out
    metrics_to_plot = list(METRICS.keys())
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.4 * n_metrics / 3, 2.2),
                             squeeze=False)
    axes = axes[0]

    # Average across all datasets
    for ax_idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        col_name, higher_is_better = METRICS[metric_name]

        if col_name not in df.columns:
            ax.set_visible(False)
            continue

        # Plot individual datasets as thin lines
        for ds in sorted(df['dataset'].unique()):
            ds_data = df[df['dataset'] == ds].sort_values('iterations')
            is_train = ds in TRAINING_DATASETS
            color = '#1f77b4' if is_train else '#ff7f0e'
            alpha = 0.2
            ax.plot(ds_data['iterations'], ds_data[col_name],
                    color=color, alpha=alpha, linewidth=0.8)

        # Plot mean for training and held-out
        for group_name, group_ds, color, marker in [
            ('Train', TRAINING_DATASETS, '#1f77b4', 'o'),
            ('Held-out', set(df['dataset'].unique()) - TRAINING_DATASETS, '#ff7f0e', 's'),
        ]:
            group_data = df[df['dataset'].isin(group_ds)]
            if group_data.empty:
                continue
            means = group_data.groupby('iterations')[col_name].mean()
            ax.plot(means.index, means.values, color=color, linewidth=1.8,
                    marker=marker, markersize=3, label=group_name, zorder=5)

        ax.set_xlabel('Iterations $T$', fontsize=8)
        ax.set_title(metric_name, fontsize=8.5, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.set_xlim(0.5, df['iterations'].max() + 0.5)

        # Add vertical line at training mean T=5
        ax.axvline(x=5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

        if ax_idx == 0:
            ax.legend(fontsize=6.5, loc='lower right')

    fig.tight_layout(pad=0.5)
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {args.output}")


if __name__ == '__main__':
    main()
