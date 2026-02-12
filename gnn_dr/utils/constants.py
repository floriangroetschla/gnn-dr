"""Constants used throughout the CoRe-GD codebase."""

# Random seeds for reproducibility
ROME_DATASET_SEED = 12345
DELAUNAY_DATASET_SEED = 0
SUITESPARSE_DATASET_SEED = 42

# Dataset split ratios
TRAIN_SPLIT_RATIO = 0.8
VAL_SPLIT_RATIO = 0.1
TEST_SPLIT_RATIO = 0.1

# Training defaults
DEFAULT_LAYER_NUM = 10
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_GRADIENT_CLIP_VALUE = 1.0

# Data directory paths
DATA_ROOT = "data"
MODEL_DIRECTORY = "models/"
