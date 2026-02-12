#!/usr/bin/env python
"""
Evaluate a trained GNN-DR model on multiple datasets.

Loads a Lightning checkpoint and computes tf-projection-qm metrics on all
configured datasets. Output is a CSV compatible with analyze_results.py.

Usage:
    python scripts/evaluate.py --checkpoint models/last.ckpt
    python scripts/evaluate.py --checkpoint models/last.ckpt --datasets mnist_clip cifar10_clip
    python scripts/evaluate.py --checkpoint models/last.ckpt --config configs/default.yaml --rounds 10
"""

import sys
from pathlib import Path

# Allow running from gnn-dr/ root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn_dr.evaluation.evaluate_model import main

if __name__ == '__main__':
    main()
