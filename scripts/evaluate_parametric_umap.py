#!/usr/bin/env python
"""
Train and evaluate Parametric UMAP on the same datasets as CoRe-DR.

Usage:
    python scripts/evaluate_parametric_umap.py
    python scripts/evaluate_parametric_umap.py --output results/pumap.csv
    python scripts/evaluate_parametric_umap.py --eval-only --checkpoint models/pumap.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_dr.evaluation.evaluate_parametric_umap import main

if __name__ == '__main__':
    main()
