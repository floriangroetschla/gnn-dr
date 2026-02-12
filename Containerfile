FROM docker.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies — Python 3.10 ships with Ubuntu 22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Ensure modern pip/setuptools for pyproject.toml support
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch 2.4.1 with CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric + extensions (matching CUDA wheels)
RUN pip install --no-cache-dir \
    torch-geometric torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

# Copy package source
WORKDIR /app
COPY pyproject.toml README.md ./
COPY gnn_dr/ gnn_dr/
COPY scripts/ scripts/
COPY configs/ configs/

# Install gnn-dr with evaluation metrics and Parametric UMAP
RUN pip install --no-cache-dir ".[metrics,pumap]"

# CLIP embedding extraction
RUN pip install --no-cache-dir openai-clip

# Volume mount points
VOLUME ["/app/data", "/app/models", "/app/results"]

# NVIDIA container runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["python"]
CMD ["scripts/train.py", "--config", "configs/default.yaml"]
