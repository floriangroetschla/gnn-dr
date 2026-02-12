"""
Export data for TikZ figure generation.
This script exports MNIST images, graph data, and UMAP coordinates.

Usage:
    python scripts/export_data_for_tikz.py
    python scripts/export_data_for_tikz.py --output-dir tikz_data
"""
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import os
import sys
from pathlib import Path
import umap

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_dr.utils.umap_weights import compute_umap_fuzzy_weights

CONFIG = {
    'n_umap_samples': 500,
    'n_graph_nodes': 100,
    'k_neighbors': 8,
    'random_seed': 42,
    'output_dir': 'tikz_data'
}

np.random.seed(CONFIG['random_seed'])

def get_data():
    """Load MNIST and compute embeddings."""
    print("Loading MNIST...")
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    n_samples = CONFIG['n_umap_samples']
    n_per_class = n_samples // 10
    selected = []
    counts = {i: 0 for i in range(10)}

    for idx, (_, label) in enumerate(mnist):
        if counts[label] < n_per_class:
            selected.append(idx)
            counts[label] += 1
        if len(selected) >= n_samples:
            break

    print(f"Loading CLIP model for {len(selected)} samples...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    embeddings, labels, images = [], [], []

    print("Computing embeddings...")
    with torch.no_grad():
        for i, idx in enumerate(selected):
            if i % 100 == 0:
                print(f"  {i}/{len(selected)}")
            img_tensor, label = mnist[idx]
            img = transforms.ToPILImage()(img_tensor)
            img_rgb = img.convert('RGB')
            inputs = processor(images=img_rgb, return_tensors="pt")
            features = model.visual_projection(
                model.vision_model(pixel_values=inputs['pixel_values']).pooler_output
            )
            embeddings.append(features.squeeze().numpy())
            labels.append(label)
            images.append(np.array(img))

    return np.array(embeddings), np.array(labels), images


def export_input_panel_data(labels, images, output_dir):
    """Export 9 sample images for input panel."""
    print("\nExporting input panel images...")
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    # Select one image per digit (0-8)
    for digit in range(9):
        idx = np.where(np.array(labels) == (digit % 10))[0][0]
        img = Image.fromarray(images[idx])
        img.save(os.path.join(img_dir, f'digit_{digit}.png'))

    print(f"  Saved 9 images to {img_dir}")


def export_graph_data(embeddings, labels, output_dir):
    """Export graph node positions and edges."""
    print("\nExporting graph construction data...")

    n = CONFIG['n_graph_nodes']
    k_neighbors = CONFIG['k_neighbors']

    # Sample subset
    sample_idx = np.random.choice(len(embeddings), n, replace=False)
    emb_subset = embeddings[sample_idx]
    labels_subset = np.array(labels)[sample_idx]

    # Project to 2D using PCA
    pca = PCA(n_components=2, random_state=CONFIG['random_seed'])
    positions_2d = pca.fit_transform(emb_subset)

    # Normalize positions to [0, 1]
    positions_2d -= positions_2d.min(axis=0)
    positions_2d /= (positions_2d.max(axis=0) + 1e-8)

    # Build kNN graph and compute UMAP weights
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine')
    nn.fit(emb_subset)
    distances, indices = nn.kneighbors(emb_subset)

    # Create directed edge list
    edge_list = []
    distance_list = []
    for i in range(n):
        for j_idx, j in enumerate(indices[i, 1:]):
            edge_list.append([i, j])
            distance_list.append(distances[i, j_idx + 1])

    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    edge_distances = torch.tensor(distance_list, dtype=torch.float32)

    # Compute UMAP weights
    edge_index_und, edge_weights = compute_umap_fuzzy_weights(
        edge_index, edge_distances, num_nodes=n, k=k_neighbors
    )

    # Save node positions
    with open(os.path.join(output_dir, 'graph_nodes.dat'), 'w') as f:
        f.write("# node_id x y label\n")
        for i in range(n):
            f.write(f"{i} {positions_2d[i, 0]:.6f} {positions_2d[i, 1]:.6f} {labels_subset[i]}\n")

    # Save edges with weights
    edge_index_np = edge_index_und.numpy()
    edge_weights_np = edge_weights.numpy()

    with open(os.path.join(output_dir, 'graph_edges.dat'), 'w') as f:
        f.write("# source target weight\n")
        for idx in range(edge_index_np.shape[1]):
            i, j = edge_index_np[0, idx], edge_index_np[1, idx]
            w = edge_weights_np[idx]
            f.write(f"{i} {j} {w:.6f}\n")

    print(f"  Saved {n} nodes and {edge_index_np.shape[1]} edges")


def export_output_data(embeddings, labels, output_dir):
    """Export UMAP scatter plot data."""
    print("\nComputing and exporting UMAP...")

    reducer = umap.UMAP(n_components=2, random_state=CONFIG['random_seed'],
                        n_neighbors=15, min_dist=0.1)
    umap_coords = reducer.fit_transform(embeddings)

    # Normalize
    umap_coords -= umap_coords.min(axis=0)
    umap_coords /= umap_coords.max()

    # Save by digit class for easier TikZ plotting
    for digit in range(10):
        mask = labels == digit
        coords = umap_coords[mask]

        with open(os.path.join(output_dir, f'umap_digit_{digit}.dat'), 'w') as f:
            f.write("# x y\n")
            for i in range(len(coords)):
                f.write(f"{coords[i, 0]:.6f} {coords[i, 1]:.6f}\n")

    print(f"  Saved UMAP coordinates for all digits")


def main():
    parser = argparse.ArgumentParser(description='Export data for TikZ figure generation')
    parser.add_argument('--output-dir', default=CONFIG['output_dir'],
                        help='Output directory (default: tikz_data)')
    args = parser.parse_args()

    CONFIG['output_dir'] = args.output_dir
    output_dir = CONFIG['output_dir']

    print("=" * 60)
    print("Exporting data for TikZ figure generation")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Get data
    embeddings, labels, images = get_data()

    # Export each component
    export_input_panel_data(labels, images, output_dir)
    export_graph_data(embeddings, labels, output_dir)
    export_output_data(embeddings, labels, output_dir)

    print("\n" + "=" * 60)
    print("Done! Data exported to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
