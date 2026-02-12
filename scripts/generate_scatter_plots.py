"""
Generate scatter plot comparison figure for the paper.

Runs PCA, t-SNE, UMAP, and CoRe-DR on specified datasets and produces a
multi-panel figure (rows=datasets, columns=methods) colored by class label.

Usage:
    python scripts/generate_scatter_plots.py \
        --checkpoint models/last.ckpt \
        --config configs/default.yaml \
        --datasets mnist_clip oxford_pets_clip food101_clip \
        --output figures/scatter_comparison.pdf

    python scripts/generate_scatter_plots.py \
        --checkpoint models/last.ckpt \
        --datasets cifar10_clip kmnist_clip \
        --max-samples 2000 \
        --output figures/scatter_comparison.pdf
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_dr.evaluation.evaluate_model import (
    load_model_from_checkpoint,
    load_dataset_for_model,
    build_knn_graph,
    preprocess_for_inference,
    run_model_inference,
)
from gnn_dr.evaluation.run_full_evaluation import create_baselines


DATASET_SHORT = {
    'mnist_clip': 'MNIST',
    'cifar10_clip': 'CIFAR-10',
    'fashion_mnist_clip': 'Fashion-MNIST',
    'kmnist_clip': 'KMNIST',
    'flowers102_clip': 'Flowers-102',
    'fgvc_aircraft_clip': 'Aircraft',
    'oxford_pets_clip': 'Oxford Pets',
    'food101_clip': 'Food-101',
}

TRAINING_DATASETS = {'mnist_clip', 'fashion_mnist_clip', 'cifar10_clip'}

METHODS_ORDER_BASE = ['PCA', 't-SNE', 'UMAP', 'CoRe-DR']
METHODS_ORDER_WITH_PUMAP = ['PCA', 't-SNE', 'UMAP', 'P-UMAP', 'CoRe-DR']


def run_all_methods(embeddings, labels, model, config, k=15, rounds=10,
                    device='cuda', random_state=42, pumap_model=None):
    """Run all methods on one dataset and return dict of {method: projection (N,2)}."""
    projections = {}

    # --- Baselines (CPU) ---
    baselines = create_baselines(k=k, random_state=random_state)
    for name in ['PCA', 't-SNE', 'UMAP']:
        if name not in baselines:
            continue
        print(f"    Running {name}...")
        start = time.time()
        proj = baselines[name].fit_transform(embeddings)
        elapsed = time.time() - start
        projections[name] = proj
        print(f"      Done in {elapsed:.1f}s")

    # --- Parametric UMAP (pre-trained, inductive) ---
    if pumap_model is not None:
        print(f"    Running P-UMAP...")
        start = time.time()
        proj = pumap_model.transform(embeddings)
        elapsed = time.time() - start
        if not np.isnan(proj).any():
            projections['P-UMAP'] = proj
            print(f"      Done in {elapsed:.1f}s")
        else:
            print(f"      WARNING: NaN in output, skipping")

    # --- CoRe-DR (GPU) ---
    print(f"    Running CoRe-DR...")
    start = time.time()
    graph = build_knn_graph(embeddings, labels, k=k, device=device)
    graph = preprocess_for_inference(graph, config)
    proj = run_model_inference(model, graph, rounds=rounds)
    elapsed = time.time() - start
    projections['CoRe-DR'] = proj
    print(f"      Done in {elapsed:.1f}s")

    return projections


def _build_palette(n_classes):
    """Build a perceptually distinct color palette for n classes."""
    if n_classes <= 10:
        # Tableau-10 inspired, slightly desaturated for scatter plots
        return np.array([
            [0.216, 0.494, 0.722],  # steel blue
            [0.894, 0.102, 0.110],  # crimson
            [0.302, 0.686, 0.290],  # green
            [1.000, 0.498, 0.000],  # orange
            [0.596, 0.306, 0.639],  # purple
            [0.651, 0.337, 0.157],  # brown
            [0.969, 0.506, 0.749],  # pink
            [0.400, 0.761, 0.647],  # teal
            [0.737, 0.741, 0.133],  # olive
            [0.090, 0.745, 0.812],  # cyan
        ])[:n_classes]
    elif n_classes <= 20:
        cmap = plt.get_cmap('tab20')
        return cmap(np.linspace(0, 0.95, n_classes))[:, :3]
    else:
        # Evenly-spaced hue wheel, golden-ratio offset for neighbor contrast
        golden = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_classes)
        hues = (indices / golden) % 1.0
        return np.array([hsv_to_rgb([h, 0.65, 0.82]) for h in hues])


def _normalize_projection(proj):
    """Normalize a 2D projection into [-0.5, 0.5]^2, preserving aspect ratio."""
    centered = proj - proj.mean(axis=0)
    span_x = np.max(centered[:, 0]) - np.min(centered[:, 0])
    span_y = np.max(centered[:, 1]) - np.min(centered[:, 1])
    scale = max(span_x, span_y)
    if scale > 0:
        centered = centered / scale
    return centered


def make_figure(all_data, output_path, methods_order, point_size=1.5, dpi=300):
    """
    Create a publication-quality scatter plot comparison figure.

    Layout is computed analytically in inches — every axis is placed manually
    via fig.add_axes() so that panels are exactly square with no clipping.
    Data is pre-normalized so set_aspect('equal') is never needed.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
    })

    n_rows = len(all_data)
    n_cols = len(methods_order)

    # --- Layout geometry (all in inches) ---
    fig_width = 6.9          # full two-column width

    label_w = 0.50           # left margin for dataset labels
    annot_w = 0.30           # right margin for Training/Held-out text
    header_h = 0.25          # top margin for method titles
    bottom_h = 0.02          # bottom margin
    gap = 0.06               # gap between panels
    sep_gap = 0.18           # extra vertical gap at train/held-out boundary

    # Find the train/held-out boundary row
    first_heldout_row = None
    for i, data in enumerate(all_data):
        if data['dataset'] not in TRAINING_DATASETS:
            first_heldout_row = i
            break

    # Compute panel size: width is constrained by figure width
    panel = (fig_width - label_w - annot_w - (n_cols - 1) * gap) / n_cols

    # Compute figure height from panel size
    n_vgaps = n_rows - 1
    extra = sep_gap if (first_heldout_row is not None and first_heldout_row > 0) else 0
    fig_height = header_h + bottom_h + n_rows * panel + n_vgaps * gap + extra

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')

    # --- Place axes manually ---
    axes = np.empty((n_rows, n_cols), dtype=object)

    for row in range(n_rows):
        for col in range(n_cols):
            # x position: left margin + col * (panel + gap)
            x = label_w + col * (panel + gap)

            # y position: top-down; row 0 is at the top
            y = fig_height - header_h - (row + 1) * panel - row * gap
            # shift down for rows after the separator
            if first_heldout_row is not None and row >= first_heldout_row:
                y -= sep_gap

            # Convert to figure-fraction coordinates
            rect = [x / fig_width, y / fig_height,
                    panel / fig_width, panel / fig_height]
            ax = fig.add_axes(rect)
            axes[row, col] = ax

    # --- Plot data ---
    for row_idx, data in enumerate(all_data):
        ds_name = data['dataset']
        labels = data['labels']
        n_classes = data['n_classes']
        projections = data['projections']

        # Build color map
        unique_labels = np.unique(labels)
        colors = _build_palette(n_classes)
        label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}
        point_colors = np.array([label_to_color[l] for l in labels])

        # Shuffle once per row (consistent order across methods)
        rng = np.random.RandomState(42)
        order = rng.permutation(len(labels))

        for col_idx, method in enumerate(methods_order):
            ax = axes[row_idx, col_idx]

            if method not in projections:
                ax.set_visible(False)
                continue

            # Normalize projection to [-0.5, 0.5]^2 preserving aspect ratio
            proj = _normalize_projection(projections[method])

            ax.scatter(
                proj[order, 0], proj[order, 1],
                c=point_colors[order],
                s=point_size,
                alpha=0.55,
                edgecolors='none',
                rasterized=True,
                linewidths=0,
            )

            # Fixed limits with padding — data is in [-0.5, 0.5]
            pad = 0.54
            ax.set_xlim(-pad, pad)
            ax.set_ylim(-pad, pad)

            # Remove all decorations
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # --- Column headers (method names) above top row ---
    for col_idx, method in enumerate(methods_order):
        ax = axes[0, col_idx]
        title = method if method != 'CoRe-DR' else 'Ours'
        # Place text in figure coords, centered above the panel
        cx = (label_w + col_idx * (panel + gap) + panel / 2) / fig_width
        ty = (fig_height - header_h * 0.35) / fig_height
        fig.text(cx, ty, title, fontsize=8.5, fontweight='bold',
                 ha='center', va='center')

    # --- Row labels (dataset names) to the left of each row ---
    for row_idx, data in enumerate(all_data):
        ax = axes[row_idx, 0]
        ds_short = DATASET_SHORT.get(data['dataset'], data['dataset'])

        # Vertical center of this row in figure coords
        y_top = fig_height - header_h - row_idx * (panel + gap)
        if first_heldout_row is not None and row_idx >= first_heldout_row:
            y_top -= sep_gap
        cy = (y_top - panel / 2) / fig_height
        lx = (label_w * 0.45) / fig_width

        fig.text(lx, cy, ds_short, fontsize=7.5, fontweight='bold',
                 ha='center', va='center', rotation=90)

    # --- Training / Held-out annotations on the right ---
    if first_heldout_row is not None and first_heldout_row > 0:
        rx = (fig_width - annot_w * 0.45) / fig_width

        # Training label — centered over training rows
        train_top = fig_height - header_h
        train_bot = train_top - first_heldout_row * panel - (first_heldout_row - 1) * gap
        fig.text(rx, (train_top + train_bot) / 2 / fig_height, 'Training',
                 fontsize=7, fontstyle='italic', color='#666666',
                 ha='center', va='center', rotation=-90)

        # Held-out label — centered over held-out rows
        n_held = n_rows - first_heldout_row
        held_top = train_bot - sep_gap
        held_bot = held_top - n_held * panel - (n_held - 1) * gap
        fig.text(rx, (held_top + held_bot) / 2 / fig_height, 'Held-out',
                 fontsize=7, fontstyle='italic', color='#666666',
                 ha='center', va='center', rotation=-90)

        # Separator line between training and held-out
        line_y = (train_bot - sep_gap / 2) / fig_height
        line_x0 = label_w / fig_width
        line_x1 = (fig_width - annot_w) / fig_width
        fig.add_artist(plt.Line2D(
            [line_x0, line_x1], [line_y, line_y],
            transform=fig.transFigure,
            color='#aaaaaa', linewidth=0.6, linestyle=(0, (4, 3)),
            clip_on=False,
        ))

    # --- Save ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches='tight' — layout is already exact
    fig.savefig(output_path, dpi=dpi, facecolor='white')
    print(f"\nFigure saved to: {output_path}")

    # Also save NPZ with raw projections for later re-plotting
    npz_path = output_path.with_suffix('.npz')
    save_data = {}
    for data in all_data:
        ds = data['dataset']
        save_data[f'{ds}_labels'] = data['labels']
        for method, proj in data['projections'].items():
            save_data[f'{ds}_{method}'] = proj
    np.savez_compressed(npz_path, **save_data)
    print(f"Raw projections saved to: {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate scatter plot comparison figure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--checkpoint', '-c', required=True,
                        help='Path to CoRe-DR Lightning checkpoint')
    parser.add_argument('--pumap-checkpoint', default=None,
                        help='Path to Parametric UMAP checkpoint (models/pumap.pt)')
    parser.add_argument('--config', default=None,
                        help='Path to config YAML (uses checkpoint config if not provided)')
    parser.add_argument('--datasets', nargs='+',
                        default=['mnist_clip', 'food101_clip'],
                        help='Datasets to include (default: mnist_clip food101_clip)')
    parser.add_argument('--output', '-o', default='paper/scatter_comparison.pdf',
                        help='Output figure path (default: paper/scatter_comparison.pdf)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per dataset (default: all)')
    parser.add_argument('--k', type=int, default=15,
                        help='Number of KNN neighbors (default: 15)')
    parser.add_argument('--rounds', type=int, default=10,
                        help='Number of GNN message passing rounds (default: 10)')
    parser.add_argument('--device', default='cuda',
                        help='Device for GNN inference (default: cuda)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--point-size', type=float, default=1.5,
                        help='Scatter point size (default: 1.5)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI (default: 300)')
    args = parser.parse_args()

    # Load CoRe-DR model
    print(f"Loading CoRe-DR model from: {args.checkpoint}")
    model, config = load_model_from_checkpoint(
        args.checkpoint, config_path=args.config, device=args.device,
    )
    print(f"  Model loaded on {args.device}")

    # Load Parametric UMAP model (optional)
    pumap_model = None
    if args.pumap_checkpoint:
        import torch
        print(f"Loading Parametric UMAP from: {args.pumap_checkpoint}")
        ckpt = torch.load(args.pumap_checkpoint, map_location=args.device, weights_only=False)
        pumap_model = ckpt['model']
        print(f"  P-UMAP loaded (trained in {ckpt.get('train_time', 0):.1f}s)")

    # Process each dataset
    all_data = []
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {DATASET_SHORT.get(dataset_name, dataset_name)}")
        print(f"{'='*60}")

        embeddings, labels, actual_size = load_dataset_for_model(
            dataset_name, split='test', max_samples=args.max_samples,
            device=args.device, random_state=args.random_state,
        )
        if embeddings is None:
            print(f"  Skipping {dataset_name} (failed to load)")
            continue

        n_classes = len(np.unique(labels))
        print(f"  Loaded {actual_size} samples, {n_classes} classes")

        projections = run_all_methods(
            embeddings, labels, model, config,
            k=args.k, rounds=args.rounds, device=args.device,
            random_state=args.random_state, pumap_model=pumap_model,
        )

        all_data.append({
            'dataset': dataset_name,
            'labels': labels,
            'n_classes': n_classes,
            'projections': projections,
        })

        # Cleanup
        import torch
        del embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_data:
        print("ERROR: No datasets were successfully processed!")
        sys.exit(1)

    # Generate figure
    print(f"\n{'='*60}")
    print("Generating figure...")
    print(f"{'='*60}")
    methods = METHODS_ORDER_WITH_PUMAP if pumap_model is not None else METHODS_ORDER_BASE
    make_figure(all_data, args.output, methods_order=methods,
                point_size=args.point_size, dpi=args.dpi)


if __name__ == '__main__':
    main()
