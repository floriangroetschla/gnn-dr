#!/usr/bin/env python
"""
Run baseline DR methods (PCA, t-SNE, UMAP, Parametric UMAP) on all datasets.

Computes tf-projection-qm metrics for each method/dataset pair and saves
results to a CSV compatible with analyze_results.py.

Usage:
    python scripts/run_baselines.py --output results/baselines.csv
    python scripts/run_baselines.py --datasets mnist_clip cifar10_clip --k 15
    python scripts/run_baselines.py --max-samples 10000 --output results/baselines.csv
"""

import sys
from pathlib import Path

# Allow running from gnn-dr/ root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_dr.evaluation.run_full_evaluation import main

if __name__ == '__main__':
    main()
