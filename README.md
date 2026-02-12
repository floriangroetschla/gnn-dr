# GNN-DR: Graph Neural Network Dimensionality Reduction

GNN-DR learns a single GNN model that maps high-dimensional CLIP embeddings to 2D visualizations.
Unlike t-SNE or UMAP, the trained model generalizes to unseen datasets without per-dataset optimization.

## Features

- **Inductive DR**: Train once, apply to any CLIP-embedded dataset
- **GRU-based GNN**: Iterative message passing with edge-weighted convolutions
- **UMAP-style loss**: Fuzzy simplicial set weights with attraction/repulsion objective
- **Multi-dataset training**: Train on multiple datasets simultaneously with weighted sampling
- **GPU-optimized**: KNN graph construction, edge weight computation, and replay buffer all on GPU
- **Comprehensive evaluation**: 14 DR quality metrics via tf-projection-qm

## Installation

### 1. Create a conda environment

```bash
conda create -n gnn-dr python=3.10 -y
conda activate gnn-dr
```

### 2. Install PyTorch and PyTorch Geometric

PyTorch and PyG extensions (torch-scatter, torch-sparse, torch-cluster) must be installed
with matching versions. Choose **one** of the following:

**CPU only:**

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
```

**CUDA 11.8:**

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
```

**CUDA 12.1:**

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

For other PyTorch/CUDA combinations, see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 3. Install GNN-DR

```bash
# From the gnn-dr/ directory
pip install -e .
```

This installs all remaining dependencies (pytorch-lightning, wandb, transformers, scikit-learn, etc.).

### 4. (Optional) Install evaluation metrics

For comprehensive DR quality metrics using tf-projection-qm:

```bash
pip install -e ".[metrics]"
```

### Verify installation

```bash
python -c "from gnn_dr.network.lightning_module import CoReGDLightningModule; print('OK')"
python scripts/train.py --help
```

## Quick Start

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

This trains on MNIST, CIFAR-10, and Fashion-MNIST simultaneously (multi-dataset mode).
Training progress is logged to [Weights & Biases](https://wandb.ai/).

Override config options from the command line:

```bash
python scripts/train.py --config configs/default.yaml --epochs 100 --lr 0.001 --device 0
```

Resume from a checkpoint:

```bash
python scripts/train.py --config configs/default.yaml --resume_from last
python scripts/train.py --config configs/default.yaml --resume_from models/my_model_best_val.ckpt
```

Key config options (in `configs/default.yaml`):

| Config key | Description | Default |
|---|---|---|
| `dataset.train_datasets` | Datasets to train on | CIFAR-10, MNIST, Fashion-MNIST |
| `dimensionality_reduction.knn_k` | Number of KNN neighbors | 15 |
| `dimensionality_reduction.umap_repulsion_weight` | Repulsion weight (gamma) | 5.0 |
| `model.conv` | Convolution type (`gru` or `gin`) | gru |
| `model.hidden_dimension` | Hidden dimension | 64 |
| `model.num_layers` | Number of GNN iterations | 12 |
| `training.lr` | Learning rate | 0.001 |
| `training.epochs` | Number of training epochs | 200 |

### Evaluating a checkpoint

Evaluate a trained model on multiple datasets with tf-projection-qm metrics:

```bash
python scripts/evaluate.py \
    --checkpoint models/last.ckpt \
    --datasets mnist_clip cifar10_clip food101_clip \
    --output results/gnn_results.csv
```

Options:

```bash
python scripts/evaluate.py --help
```

| Flag | Description | Default |
|---|---|---|
| `--checkpoint` | Path to .ckpt file (required) | - |
| `--config` | Config YAML (uses checkpoint config if omitted) | None |
| `--datasets` | Datasets to evaluate (space-separated) | all |
| `--rounds` | Number of GNN message passing rounds | 10 |
| `--k` | KNN neighbors for graph construction | 15 |
| `--max-samples` | Max samples per dataset | 10000 |
| `--output` | Output CSV path | results/gnn_results.csv |
| `--compile` | Use torch.compile for faster inference | off |

### Running baselines

Run PCA, t-SNE, UMAP, and Parametric UMAP on all datasets:

```bash
python scripts/run_baselines.py --output results/baselines.csv
```

Run on specific datasets:

```bash
python scripts/run_baselines.py \
    --datasets mnist_clip cifar10_clip fashion_mnist_clip \
    --output results/baselines.csv
```

### Training and evaluating Parametric UMAP

Train a Parametric UMAP encoder on the same training data as CoRe-DR (MNIST + Fashion-MNIST + CIFAR-10), then evaluate on all datasets:

```bash
python scripts/evaluate_parametric_umap.py \
    --output results/pumap.csv \
    --checkpoint models/pumap.pt \
    --device cuda
```

Evaluate using a previously trained checkpoint:

```bash
python scripts/evaluate_parametric_umap.py \
    --eval-only \
    --checkpoint models/pumap.pt \
    --output results/pumap.csv
```

### Generating scatter plots

Compare GNN-DR projections against baselines visually:

```bash
python scripts/generate_scatter_plots.py \
    --checkpoint models/last.ckpt \
    --datasets mnist_clip cifar10_clip food101_clip \
    --output figures/scatter_comparison.pdf
```

Include Parametric UMAP in the comparison:

```bash
python scripts/generate_scatter_plots.py \
    --checkpoint models/last.ckpt \
    --pumap-checkpoint models/pumap.pt \
    --datasets mnist_clip fashion_mnist_clip cifar10_clip kmnist_clip food101_clip \
    --output figures/scatter_comparison.pdf
```

### Analyzing results

Generate a comparison table (console + LaTeX):

```bash
python scripts/analyze_results.py \
    --baselines results/baselines.csv \
    --gnn results/gnn_results.csv \
    --pumap results/pumap.csv \
    --latex --latex-out results/table.tex
```

Generate a full-metric appendix table:

```bash
python scripts/analyze_results.py \
    --baselines results/baselines.csv \
    --gnn results/gnn_results.csv \
    --pumap results/pumap.csv \
    --appendix --appendix-out results/appendix_table.tex
```

### Sweeping iteration counts

Evaluate how metrics change with the number of GNN message passing rounds:

```bash
python scripts/sweep_iterations.py \
    --checkpoint models/last.ckpt \
    --max-iterations 20 \
    --output results/iteration_sweep.csv
```

Plot the results:

```bash
python scripts/plot_iterations.py \
    --input results/iteration_sweep.csv \
    --output figures/iteration_sweep.pdf
```

### Generating the paper figure

Generate the main TikZ-based architecture figure (requires CLIP model download):

```bash
python scripts/generate_figure.py --output-dir figure_assets
pdflatex figure_main.tex
```

Export raw data for TikZ plotting:

```bash
python scripts/export_data_for_tikz.py --output-dir tikz_data
python scripts/generate_scatter.py --input-dir tikz_data
```

## Reproducing Paper Results

The following commands reproduce all results, tables, and figures from the paper.
All scripts should be run from the `gnn-dr/` directory. Evaluation requires a GPU
(NVIDIA A6000 or similar) for reasonable runtimes.

### Step 1: Train the model

```bash
python scripts/train.py --config configs/default.yaml --device 0
```

Training runs for 200 epochs on MNIST, CIFAR-10, and Fashion-MNIST. Checkpoints
are saved to `models/`. Training progress is logged to W&B.

### Step 2: Evaluate baselines (PCA, t-SNE, UMAP)

```bash
python scripts/run_baselines.py \
    --datasets mnist_clip fashion_mnist_clip cifar10_clip \
               kmnist_clip fgvc_aircraft_clip oxford_pets_clip food101_clip \
    --output results/baselines.csv
```

### Step 3: Train and evaluate Parametric UMAP

```bash
python scripts/evaluate_parametric_umap.py \
    --output results/pumap.csv \
    --checkpoint models/pumap.pt \
    --device cuda
```

### Step 4: Evaluate the GNN model

```bash
python scripts/evaluate.py \
    --checkpoint models/last.ckpt \
    --datasets mnist_clip fashion_mnist_clip cifar10_clip \
               kmnist_clip fgvc_aircraft_clip oxford_pets_clip food101_clip \
    --output results/gnn_results.csv
```

### Step 5: Generate tables

Main results table (Table 1):

```bash
python scripts/analyze_results.py \
    --baselines results/baselines.csv \
    --gnn results/gnn_results.csv \
    --pumap results/pumap.csv \
    --latex --latex-out results/table.tex
```

Full metrics appendix table:

```bash
python scripts/analyze_results.py \
    --baselines results/baselines.csv \
    --gnn results/gnn_results.csv \
    --pumap results/pumap.csv \
    --appendix --appendix-out results/appendix_table.tex
```

### Step 6: Generate figures

Scatter plot comparison:

```bash
python scripts/generate_scatter_plots.py \
    --checkpoint models/last.ckpt \
    --pumap-checkpoint models/pumap.pt \
    --datasets mnist_clip fashion_mnist_clip cifar10_clip \
               kmnist_clip fgvc_aircraft_clip oxford_pets_clip food101_clip \
    --output figures/scatter_comparison.pdf
```

Iteration sweep plot:

```bash
python scripts/sweep_iterations.py \
    --checkpoint models/last.ckpt \
    --max-iterations 20 \
    --output results/iteration_sweep.csv

python scripts/plot_iterations.py \
    --input results/iteration_sweep.csv \
    --output figures/iteration_sweep.pdf
```

Architecture figure:

```bash
python scripts/generate_figure.py --output-dir figure_assets
pdflatex figure_main.tex
```

## Container (Podman / Docker)

A `Containerfile` is provided for reproducible GPU-accelerated training and evaluation.

### Build the image

```bash
cd gnn-dr/
podman build -t gnn-dr .
```

### Verify the build

```bash
podman run --rm gnn-dr -c "from gnn_dr.network.lightning_module import CoReGDLightningModule; print('OK')"
```

### GPU training

Requires the [NVIDIA Container Toolkit (CDI)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html):

```bash
podman run --rm --device nvidia.com/gpu=all \
    -v ./data:/app/data \
    -v ./models:/app/models \
    -v ./results:/app/results \
    gnn-dr scripts/train.py --config configs/default.yaml --device 0
```

### Evaluate a checkpoint

```bash
podman run --rm --device nvidia.com/gpu=all \
    -v ./data:/app/data \
    -v ./models:/app/models \
    -v ./results:/app/results \
    gnn-dr scripts/evaluate.py \
        --checkpoint models/last.ckpt \
        --datasets mnist_clip cifar10_clip food101_clip \
        --output results/gnn_results.csv
```

### Run baselines

```bash
podman run --rm --device nvidia.com/gpu=all \
    -v ./data:/app/data \
    -v ./results:/app/results \
    gnn-dr scripts/run_baselines.py --output results/baselines.csv
```

> **Note:** Replace `podman` with `docker` if using Docker. For Docker GPU support, use `--gpus all` instead of `--device nvidia.com/gpu=all`.

## Supported Datasets

All datasets use CLIP (ViT-B/32) embeddings, extracted and cached automatically on first use.

**Training (default):** MNIST, CIFAR-10, Fashion-MNIST

**Evaluation:** KMNIST, Flowers-102, FGVC Aircraft, Oxford Pets, Food-101, STL-10, SVHN, EMNIST, CIFAR-100, Caltech-101, Stanford Cars, DTD

Dataset names for CLI arguments: `mnist_clip`, `cifar10_clip`, `fashion_mnist_clip`, `cifar100_clip`, `svhn_clip`, `stl10_clip`, `emnist_clip`, `kmnist_clip`, `flowers102_clip`, `fgvc_aircraft_clip`, `oxford_pets_clip`, `food101_clip`, `dtd_clip`, `caltech101_clip`, `stanford_cars_clip`

## Architecture

1. **Preprocessing**: Extract CLIP embeddings, build KNN graph, compute UMAP fuzzy simplicial set weights
2. **Input features**: PCA projection of CLIP embeddings (configurable)
3. **GNN**: Encoder MLP -> Iterative GRU convolutions with edge weights -> Decoder MLP -> 2D output
4. **Loss**: UMAP cross-entropy with attraction (positive edges) and repulsion (negative sampling)
5. **Training**: Dynamic graph generation with GPU replay buffer for stability

## Project Structure

```
gnn-dr/
├── pyproject.toml                # Package configuration
├── configs/
│   └── default.yaml              # Default training configuration
├── gnn_dr/
│   ├── config/                   # Configuration dataclasses and loading
│   ├── network/                  # Model, loss, training loop, callbacks
│   │   ├── model.py              # CoReGD model with GRU/GIN convolutions
│   │   ├── losses.py             # UMAP attraction/repulsion loss
│   │   ├── lightning_module.py   # PyTorch Lightning training module
│   │   ├── lightning_datamodule.py  # Data loading and preprocessing
│   │   ├── convolutions.py       # GRUEdgeConv, GINEdgeConv layers
│   │   ├── replay_buffer.py      # GPU replay buffer for training stability
│   │   └── *_callback.py         # Evaluation callbacks
│   ├── datasets/                 # CLIP dataset loading and graph construction
│   │   ├── clip_dr_base.py       # Base class for GPU-accelerated CLIP datasets
│   │   ├── loaders.py            # Dataset loading functions
│   │   └── *_dr.py               # Per-dataset implementations (15 datasets)
│   ├── baselines/                # PCA, t-SNE, UMAP, Parametric UMAP
│   ├── evaluation/               # Model evaluation and metrics
│   │   ├── evaluate_model.py     # Evaluate GNN checkpoints
│   │   ├── evaluate_parametric_umap.py  # Train + evaluate Parametric UMAP
│   │   ├── run_full_evaluation.py  # Run all baselines
│   │   └── show_metric_table.py  # Display results
│   └── utils/                    # Edge weights, logging, visualization
│       ├── umap_weights.py       # UMAP + t-SNE edge weight computation
│       └── logging.py            # W&B logging utilities
└── scripts/
    ├── train.py                  # Training entry point
    ├── evaluate.py               # Evaluate a trained GNN checkpoint
    ├── run_baselines.py          # Run PCA, t-SNE, UMAP baselines
    ├── evaluate_parametric_umap.py  # Train + evaluate Parametric UMAP
    ├── generate_scatter_plots.py # Multi-method scatter plot comparison
    ├── analyze_results.py        # Generate comparison tables (+ LaTeX)
    ├── sweep_iterations.py       # Sweep GNN iteration counts
    ├── plot_iterations.py        # Plot metrics vs. iterations
    ├── generate_figure.py        # Generate TikZ paper figure
    ├── export_data_for_tikz.py   # Export data for TikZ plotting
    └── generate_scatter.py       # Standalone scatter plot from exported data
```
