"""
Generate TikZ-based figure with matplotlib components only where necessary.
This script creates a .tex file that can be compiled with pdflatex.

Usage:
    python scripts/generate_figure.py
    python scripts/generate_figure.py --output-dir paper/figure_assets
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import os
import sys
from pathlib import Path
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import umap
import networkx as nx

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_dr.utils.umap_weights import compute_umap_fuzzy_weights

# Configuration
CONFIG = {
    'n_umap_samples': 3000,
    'n_graph_nodes': 60,  # Reduced from 150 for legibility
    'k_neighbors': 15,  # Match config knn_k
    'random_seed': 42,
    'output_dir': 'figure_assets'
}

# Colors
COLORS = {
    0: '#3182bd', 1: '#e6550d', 2: '#31a354', 3: '#de2d26', 4: '#756bb1',
    5: '#8c6d31', 6: '#d53e4f', 7: '#636363', 8: '#969696', 9: '#6baed6',
}

np.random.seed(CONFIG['random_seed'])
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
})


def get_data():
    """Load MNIST and compute CLIP embeddings."""
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

    print(f"Loading CLIP for {len(selected)} samples...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    embeddings, labels, images = [], [], []
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


def export_mnist_images(labels, images, output_dir):
    """Export 9 MNIST images (one per digit 0-8)."""
    print("Exporting MNIST images...")
    for digit in range(9):
        idx = np.where(np.array(labels) == (digit % 10))[0][0]
        img = Image.fromarray(images[idx])
        img.save(os.path.join(output_dir, f'mnist_{digit}.png'))


def generate_graph_panel(embeddings, labels, output_dir):
    """Generate graph construction panel as PDF."""
    print("Generating graph panel...")

    n = CONFIG['n_graph_nodes']
    k_neighbors = CONFIG['k_neighbors']

    # Sample subset
    sample_idx = np.random.choice(len(embeddings), n, replace=False)
    emb_subset = embeddings[sample_idx]
    labels_subset = np.array(labels)[sample_idx]

    # Build kNN graph
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine')
    nn.fit(emb_subset)
    distances, indices = nn.kneighbors(emb_subset)

    # Get UMAP weights
    edge_list = []
    distance_list = []
    for i in range(n):
        for j_idx, j in enumerate(indices[i, 1:]):
            edge_list.append([i, j])
            distance_list.append(distances[i, j_idx + 1])

    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    edge_distances = torch.tensor(distance_list, dtype=torch.float32)
    edge_index_und, edge_weights = compute_umap_fuzzy_weights(
        edge_index, edge_distances, num_nodes=n, k=k_neighbors
    )

    # Create NetworkX graph for spring layout
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edge_index_np = edge_index_und.numpy()
    edge_weights_np = edge_weights.numpy()
    for idx in range(edge_index_np.shape[1]):
        i, j = edge_index_np[0, idx], edge_index_np[1, idx]
        w = edge_weights_np[idx]
        G.add_edge(i, j, weight=w)

    # Use spring layout with weights
    positions_dict = nx.spring_layout(G, k=0.5, iterations=100, seed=CONFIG['random_seed'])
    positions_2d = np.array([positions_dict[i] for i in range(n)])

    # Normalize with padding to avoid cutoff
    padding = 0.05
    positions_2d -= positions_2d.min(axis=0)
    positions_2d /= (positions_2d.max(axis=0) + 1e-8)
    positions_2d = positions_2d * (1 - 2*padding) + padding

    # Sample negative edges (pairs not in k-NN graph)
    pos_edges_set = set()
    for idx in range(edge_index_np.shape[1]):
        i, j = edge_index_np[0, idx], edge_index_np[1, idx]
        pos_edges_set.add((min(i, j), max(i, j)))

    n_neg_edges = min(60, n)
    neg_edges = []
    max_attempts = n_neg_edges * 10
    attempts = 0
    while len(neg_edges) < n_neg_edges and attempts < max_attempts:
        i, j = np.random.randint(0, n, 2)
        if i != j:
            edge = (min(i, j), max(i, j))
            if edge not in pos_edges_set and edge not in neg_edges:
                neg_edges.append((i, j))
        attempts += 1

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw negative edges first (so positive edges are on top)
    for i, j in neg_edges:
        p1, p2 = positions_2d[i], positions_2d[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='#de2d26', linewidth=0.7, alpha=0.5,
                solid_capstyle='round', zorder=0, linestyle='--')

    # Draw all positive edges
    for idx in range(edge_index_np.shape[1]):
        i, j = edge_index_np[0, idx], edge_index_np[1, idx]
        w = edge_weights_np[idx]
        p1, p2 = positions_2d[i], positions_2d[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='#2980b9', linewidth=0.3 + w*0.7, alpha=0.35 + w*0.45,
                solid_capstyle='round', zorder=1)

    # Draw nodes
    for i in range(n):
        x, y = positions_2d[i]
        digit = int(labels_subset[i])
        node = Circle((x, y), 0.018, facecolor=COLORS[digit],
                     edgecolor='#333333', linewidth=0.3, zorder=2, alpha=0.8)
        ax.add_patch(node)

    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(output_dir, 'graph_panel.pdf'),
               bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


def generate_umap_scatter(embeddings, labels, output_dir):
    """Generate UMAP scatter plot as PDF - clean style matching graph panel."""
    print("Generating UMAP scatter plot...")

    reducer = umap.UMAP(n_components=2, random_state=CONFIG['random_seed'],
                        n_neighbors=15, min_dist=0.1)
    umap_coords = reducer.fit_transform(embeddings)
    umap_coords -= umap_coords.min(axis=0)
    umap_coords /= umap_coords.max(axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0), dpi=150)

    for digit in range(10):
        mask = labels == digit
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                  c=COLORS[digit], s=2.5, alpha=0.7, edgecolors='none', rasterized=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.margins(0.02)

    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(output_dir, 'umap_scatter.pdf'),
               bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


def generate_tex_file(output_dir):
    """Generate the main TikZ .tex file."""
    print("Generating .tex file...")

    tex_content = r'''\documentclass{article}
\usepackage[paperwidth=18.5cm,paperheight=9.2cm,margin=0.25cm]{geometry}
\usepackage{tikz}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usetikzlibrary{positioning,calc,arrows.meta,fit,shapes.geometric}
\pagestyle{empty}

% Colors - professional palette for scientific publication
\definecolor{boxcolor}{HTML}{f5f5f5}
\definecolor{edgecolor}{HTML}{2c3e50}
\definecolor{textcolor}{HTML}{212529}
\definecolor{poscolor}{HTML}{2980b9}
\definecolor{negcolor}{HTML}{c0392b}
\definecolor{notecolor}{HTML}{7f8c8d}
\definecolor{color0}{HTML}{2980b9}
\definecolor{color1}{HTML}{e67e22}
\definecolor{color2}{HTML}{27ae60}
\definecolor{color3}{HTML}{c0392b}
\definecolor{color4}{HTML}{8e44ad}
\definecolor{color5}{HTML}{95a5a6}
\definecolor{color6}{HTML}{e74c3c}
\definecolor{color7}{HTML}{34495e}
\definecolor{color8}{HTML}{7f8c8d}
\definecolor{color9}{HTML}{3498db}
\definecolor{preprocessbg}{HTML}{ecf0f1}
\definecolor{nnbg}{HTML}{ecf0f1}
\definecolor{outputbg}{HTML}{ecf0f1}

\begin{document}
\noindent\centering
\begin{tikzpicture}

% ============================================================
% ROW 1: PREPROCESSING (TOP) - Compact design
% ============================================================

% Preprocessing box (full width)
\draw[edgecolor, line width=1.2pt, rounded corners=3pt] (-0.45,8.7) rectangle (17.55,13.5);
\node[anchor=north east, font=\normalsize\bfseries, color=edgecolor] at (17.37,13.35) {Preprocessing};
\node[anchor=north east, font=\scriptsize, color=notecolor] at (17.37,13.0) {(fixed)};

% MNIST grid (3x3) - compact spacing with visible gaps between images
\node at (0.5,12.15) {\includegraphics[width=0.9cm]{''' + output_dir + r'''/mnist_0.png}};
\node at (1.6,12.15) {\includegraphics[width=0.9cm]{''' + output_dir + r'''/mnist_1.png}};
\node at (2.7,12.15) {\includegraphics[width=0.9cm]{''' + output_dir + r'''/mnist_2.png}};
\node at (0.5,11.1) {\includegraphics[width=0.9cm]{''' + output_dir + r'''/mnist_3.png}};
\node at (1.6,11.1) {\includegraphics[width=0.9cm]{''' + output_dir + r'''/mnist_4.png}};
\node at (2.7,11.1) {\includegraphics[width=0.9cm]{''' + output_dir + r'''/mnist_5.png}};
\node at (0.5,10.05) {\includegraphics[width=0.9cm]{''' + output_dir + r'''/mnist_6.png}};
\node at (1.6,10.05) {\includegraphics[width=0.9cm]{''' + output_dir + r'''/mnist_7.png}};
\node at (2.7,10.05) {\includegraphics[width=0.9cm]{''' + output_dir + r'''/mnist_8.png}};

% Input images label (below grid)
\node[font=\small] at (1.6,9.15) {\textbf{Input Images}};

% Arrow to embeddings
\draw[-{Stealth[length=2.5mm]}, line width=1.2pt, color=edgecolor] (3.35,11.1) -- (4.25,11.1);
\node[font=\small\itshape] at (3.8,11.5) {CLIP};

% CLIP Embeddings - mathematical column vectors with bmatrix (taller, more values)
\node at (5.0,11.1) {\footnotesize$\begin{bmatrix} 0.42 \\ \!-0.17 \\ 0.91 \\ \vdots \\ \!-0.58 \\ 0.83 \end{bmatrix}$};
\node[font=\large\bfseries, color=notecolor] at (5.85,11.1) {$\cdots$};
\node at (6.7,11.1) {\footnotesize$\begin{bmatrix} \!-0.31 \\ 0.65 \\ \!-0.04 \\ \vdots \\ 0.72 \\ \!-0.12 \end{bmatrix}$};

% Embedding label
\node[font=\small] at (5.85,9.15) {\textbf{CLIP Embeddings}};

% k-NN boxes - centered at y=11.1
\node[rectangle, rounded corners=2pt, draw=poscolor, fill=poscolor!8,
      line width=1.0pt, minimum width=2.3cm, minimum height=0.7cm, align=center]
      (knn) at (10.0,11.1) {\small\textbf{k-NN Graph} \\ \scriptsize $k=15$, cosine};

\node[rectangle, rounded corners=2pt, draw=poscolor, fill=poscolor!8,
      line width=1.0pt, minimum width=2.2cm, minimum height=0.7cm, align=center]
      (umap) at (10.0,12.4) {\small\textbf{UMAP Weights} \\ \scriptsize fuzzy simplicial set};

\node[rectangle, rounded corners=2pt, draw=negcolor, fill=negcolor!8,
      line width=1.0pt, minimum width=2.2cm, minimum height=0.7cm, align=center]
      (negsample) at (10.0,9.8) {\small\textbf{Neg. Sampling} \\ \scriptsize random pairs};

% Graph visualization - centered at y=11.1 (reduced height for balance)
\node (graphviz) at (15.0,11.1) {\includegraphics[height=3.5cm]{''' + output_dir + r'''/graph_panel.pdf}};

% Weighted graph label (below visualization, aligned with other labels)
\node[font=\small] at (15.0,9.15) {\textbf{Weighted Graph}};

% Arrow from embeddings to k-NN (horizontal)
\draw[-{Stealth[length=2.5mm]}, line width=1.2pt, color=edgecolor] (7.4,11.1) -- (knn.west);

% Process arrows
\draw[-{Stealth[length=2mm]}, line width=0.9pt, color=edgecolor] (knn.north) -- (umap.south);
\draw[-{Stealth[length=2mm]}, line width=0.9pt, color=edgecolor] (knn.south) -- (negsample.north);

% To graph visualization
\draw[-{Stealth[length=2mm]}, line width=0.9pt, color=edgecolor]
      (knn.east) -- (graphviz.west)
      node[pos=0.5, above, font=\scriptsize, color=poscolor] {$\mathcal{E}^+$};

\draw[-{Stealth[length=2mm]}, line width=0.9pt, color=edgecolor]
      (umap.east) -- (graphviz.west |- umap.east)
      node[pos=0.5, above, font=\scriptsize, color=poscolor] {$w_{ij}$};

\draw[-{Stealth[length=2mm]}, line width=0.9pt, color=edgecolor]
      (negsample.east) -- (graphviz.west |- negsample.east)
      node[pos=0.5, above, font=\scriptsize, color=negcolor] {$\mathcal{E}^-$};

% ============================================================
% ROW 2: NEURAL NETWORK (HORIZONTAL) + OUTPUT (BOTTOM)
% ============================================================

% Neural Network box (compact)
\draw[edgecolor, line width=1.2pt, rounded corners=3pt] (-0.45,5.0) rectangle (12.3,8.3);
\node[anchor=north east, font=\normalsize\bfseries, color=edgecolor] at (12.12,8.15) {Neural Network};
\node[anchor=north east, font=\scriptsize, color=notecolor] at (12.12,7.80) {(trainable)};

% Output box (compact)
\draw[edgecolor, line width=1.2pt, rounded corners=3pt] (12.55,5.0) rectangle (17.55,8.3);
\node[anchor=north east, font=\normalsize\bfseries, color=edgecolor] at (17.37,8.15) {Output};
\node[anchor=north east, font=\scriptsize, color=notecolor] at (17.37,7.80) {(optimized)};

% GNN Architecture - horizontal layout
\begin{scope}[
    gnnbox/.style={rectangle, rounded corners=2pt, draw=edgecolor, fill=nnbg,
                   line width=1.0pt, minimum height=1.2cm, align=center},
    posbox/.style={gnnbox, draw=poscolor, fill=poscolor!8, minimum width=2.2cm},
    negbox/.style={gnnbox, draw=negcolor, fill=negcolor!8, minimum width=2.2cm},
    smallbox/.style={gnnbox, minimum width=1.8cm},
    arrow/.style={-{Stealth[length=2.5mm]}, line width=1.0pt, color=edgecolor},
    looparrow/.style={-{Stealth[length=2mm]}, line width=0.9pt, color=notecolor},
    node distance=0.5cm and 0.5cm
]
    % Horizontal chain - centered at y=6.5
    \node[smallbox] (pcafeat) at (0.72,6.5) {\small\textbf{PCA} \\ \scriptsize $\mathbb{R}^{n \times 2}$};

    \node[smallbox, right=of pcafeat] (encoder) {\small\textbf{Encoder} \\ \scriptsize $\mathbb{R}^{n \times 64}$};

    \node[posbox, right=0.7cm of encoder] (convpos) {
        \small\textbf{\color{poscolor}GNN $\mathcal{E}^+$} \\[-1pt]
        \scriptsize\color{poscolor}(attraction)
    };

    \node[negbox, right=of convpos] (convneg) {
        \small\textbf{\color{negcolor}GNN $\mathcal{E}^-$} \\[-1pt]
        \scriptsize\color{negcolor}(repulsion)
    };

    \node[smallbox, right=0.7cm of convneg] (decoder) {\small\textbf{Decoder} \\ \scriptsize $\mathbb{R}^{n \times 2}$};

    % Arrows
    \draw[arrow] (pcafeat) -- (encoder);
    \draw[arrow] (encoder) -- (convpos);
    \draw[arrow] (convpos) -- (convneg);
    \draw[arrow] (convneg) -- (decoder);

    % Dotted box around iterated components
    \node[rectangle, rounded corners=3pt, draw=notecolor!80!black, line width=1.5pt, dashed,
          fit=(convpos)(convneg), inner sep=3mm] (iterbox) {};

    % Iteration loop: splits off convneg->decoder arrow, merges into encoder->convpos arrow
    \coordinate (iter-out) at ($(convneg.east) + (0.45, 0)$);
    \coordinate (iter-in) at ($(convpos.west) + (-0.45, 0)$);
    \draw[looparrow, rounded corners=5pt]
        (iter-out) -- ++(0, 1.1) -| (iter-in);

    % xL label centered above the loop
    \node[color=notecolor, font=\normalsize\bfseries]
        at ($(iter-out)!0.5!(iter-in) + (0, 1.25)$) {$\times L$};
\end{scope}

% Output scatter plot (square, compact)
\node at (14.85,6.5) {\includegraphics[height=2.8cm]{''' + output_dir + r'''/umap_scatter.pdf}};

% Arrow from decoder to output
\draw[-{Stealth[length=2.5mm]}, line width=1.2pt, color=edgecolor]
      (decoder.east) -- (13.0,6.5);

\end{tikzpicture}
\end{document}
'''

    with open('figure_main.tex', 'w') as f:
        f.write(tex_content)


def main():
    parser = argparse.ArgumentParser(description='Generate TikZ-based paper figure')
    parser.add_argument('--output-dir', default=CONFIG['output_dir'],
                        help='Directory for generated assets (default: figure_assets)')
    args = parser.parse_args()

    CONFIG['output_dir'] = args.output_dir
    output_dir = CONFIG['output_dir']

    print("=" * 60)
    print("Generating TikZ-based figure")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Get data
    embeddings, labels, images = get_data()

    # Generate components
    export_mnist_images(labels, images, output_dir)
    generate_graph_panel(embeddings, labels, output_dir)
    generate_umap_scatter(embeddings, labels, output_dir)
    generate_tex_file(output_dir)

    print("\n" + "=" * 60)
    print("Done! Generated:")
    print("  - figure_main.tex (compile with: pdflatex figure_main.tex)")
    print("  - Assets in:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
