"""
GNN-DR Training Script

Main entry point for training GNN-DR dimensionality reduction models with:
- PyTorch Lightning for streamlined training workflows
- Weights & Biases (W&B) logging and hyperparameter tracking
- Flexible configuration via YAML files
- Support for GPU/CPU with automatic device detection
- Multi-size evaluation across datasets
- Replay buffer with optimized tensor handling
- Graph coarsening for large-scale layouts

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --epochs 100 --device 0
    python train.py --config configs/default.yaml --fast_dev_run

    # Resume from last checkpoint
    python train.py --config configs/default.yaml --resume_from last

    # Resume from specific checkpoint
    python train.py --config configs/default.yaml --resume_from models/my_model_best_val.ckpt
"""

import argparse
import hashlib
import json
from pathlib import Path
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from gnn_dr.config import load_config
from gnn_dr.network.lightning_module import CoReGDLightningModule
from gnn_dr.network.lightning_datamodule import CoReGDDataModule
from gnn_dr.network.dr_evaluation_callback import DRMetricsCallback
from gnn_dr.network.multi_size_evaluation_callback import MultiSizeEvaluationCallback
from gnn_dr.utils.constants import MODEL_DIRECTORY


def main():
    """
    Main training entry point.

    Orchestrates the entire training pipeline:
    1. Parse command-line arguments
    2. Load and apply configuration overrides
    3. Initialize Lightning module and data module
    4. Setup callbacks (checkpointing, learning rate monitoring, DR metrics)
    5. Configure Lightning trainer with appropriate device
    6. Run training and testing
    """
    parser = argparse.ArgumentParser(
        description='Train GNN-DR model with PyTorch Lightning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file (required)
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (.yaml)'
    )

    # Override options
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--device', type=int, help='Override GPU device')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--model_name', type=str, help='Override model name')
    parser.add_argument('--wandb_project_name', type=str, help='Override W&B project name')
    parser.add_argument('--verbose', action='store_true', default=None,
                        help='Enable verbose output')

    # Lightning-specific options
    parser.add_argument('--fast_dev_run', action='store_true',
                        help='Run 1 batch for debugging')
    parser.add_argument('--limit_train_batches', type=float, default=1.0,
                        help='Limit training batches (for debugging)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from. '
                             'Use "last" to resume from last.ckpt in models/ directory, '
                             'or provide full path to checkpoint file')

    # Multi-GPU options
    parser.add_argument('--devices', type=str, default=None,
                        help='GPU devices to use. Examples: "0,1,2,3" for specific GPUs, "-1" for all GPUs')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Training strategy: auto, ddp, ddp_spawn, fsdp')
    parser.add_argument('--num_nodes', type=int, default=None,
                        help='Number of nodes for multi-node training')

    args = parser.parse_args()

    # Load config from file
    config = load_config(args.config)

    # Apply command-line overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.device is not None:
        config.device = args.device
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.lr = args.lr
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.wandb_project_name is not None:
        config.wandb_project_name = args.wandb_project_name
    if args.verbose is not None:
        config.verbose = args.verbose

    # Generate model name if not provided
    if config.model_name == 'GNN-DR':
        config_str = json.dumps(config.to_flat_dict(), sort_keys=True)
        hash_object = hashlib.sha256(config_str.encode())
        config.model_name = hash_object.hexdigest()[:16]

    # Create Lightning module and datamodule
    model = CoReGDLightningModule(config)
    datamodule = CoReGDDataModule(config)

    # Setup callbacks
    callbacks = []

    # Checkpoint callbacks
    if config.store_models:
        # Best validation checkpoint - only updated when val_loss improves
        checkpoint_callback = ModelCheckpoint(
            dirpath=MODEL_DIRECTORY,
            filename=f'{config.model_name}_best_val',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            verbose=config.verbose
        )
        callbacks.append(checkpoint_callback)

        # Latest checkpoint - saved every epoch (regardless of validation)
        checkpoint_last = ModelCheckpoint(
            dirpath=MODEL_DIRECTORY,
            filename='last',
            every_n_epochs=1,
            save_top_k=1,  # Only keep the most recent
            verbose=config.verbose
        )
        callbacks.append(checkpoint_last)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # DR metrics callback
    if not args.fast_dev_run:
        dr_logging_config = config.logging if hasattr(config, 'logging') else None

        dr_metrics_callback = DRMetricsCallback(
            config=config,
            k_neighbors=config.dimensionality_reduction.knn_k if hasattr(config, 'dimensionality_reduction') else 15,
            compute_every_n_epochs=getattr(dr_logging_config, 'dr_metrics_interval', 1) if dr_logging_config else 1,
            compute_trustworthiness=getattr(dr_logging_config, 'dr_compute_trustworthiness', True) if dr_logging_config else True,
            compute_continuity=getattr(dr_logging_config, 'dr_compute_continuity', True) if dr_logging_config else True,
            compute_knn_recall=getattr(dr_logging_config, 'dr_compute_knn_recall', True) if dr_logging_config else True,
            compute_distance_correlation=getattr(dr_logging_config, 'dr_compute_distance_correlation', True) if dr_logging_config else True,
            compute_silhouette=getattr(dr_logging_config, 'dr_compute_silhouette', True) if dr_logging_config else True,
            compute_umap_loss=getattr(dr_logging_config, 'dr_compute_umap_loss', False) if dr_logging_config else False,
            generate_scatter_plots=getattr(dr_logging_config, 'dr_generate_scatter_plots', True) if dr_logging_config else True,
            scatter_plot_interval=getattr(dr_logging_config, 'dr_scatter_plot_interval', 5) if dr_logging_config else 5,
            track_embedding_stats=getattr(dr_logging_config, 'dr_track_embedding_stats', True) if dr_logging_config else True,
        )
        callbacks.append(dr_metrics_callback)

        # Multi-size evaluation callback (if enabled in config)
        if hasattr(config, 'multi_size_evaluation') and config.multi_size_evaluation.enabled:
            mse_config = config.multi_size_evaluation

            multi_size_callback = MultiSizeEvaluationCallback(
                config=config,
                eval_sizes=getattr(mse_config, 'eval_sizes', [100, 500, 1000, 5000, 10000]),
                eval_intervals=getattr(mse_config, 'eval_intervals', None),
                round_counts=getattr(mse_config, 'round_counts', [1, 2, 5, 10, 20, 50]),
                multi_round_interval=getattr(mse_config, 'multi_round_interval', 10),
                multi_round_sizes=getattr(mse_config, 'multi_round_sizes', None),
                generate_visualizations=getattr(mse_config, 'generate_scatter_plots', True),
                viz_sizes=getattr(mse_config, 'viz_sizes', None),
                viz_interval=getattr(mse_config, 'viz_interval', 10),
                k_neighbors=config.dimensionality_reduction.knn_k if hasattr(config, 'dimensionality_reduction') else 15,
                eval_datasets=getattr(mse_config, 'eval_datasets', None),
                eval_use_test_split=getattr(mse_config, 'eval_use_test_split', True),
            )
            callbacks.append(multi_size_callback)

            if config.verbose:
                print(f"Multi-size evaluation enabled: sizes={mse_config.eval_sizes}")
                eval_datasets = getattr(mse_config, 'eval_datasets', None)
                if eval_datasets:
                    dataset_names = [d.get('label', d.get('name', '?')) if isinstance(d, dict) else d for d in eval_datasets]
                    print(f"Multi-size evaluation datasets: {dataset_names}")
                print(f"Multi-size evaluation viz_sizes={getattr(mse_config, 'viz_sizes', None)}, viz_interval={getattr(mse_config, 'viz_interval', 10)}")

        if config.verbose:
            print(f"DR metrics callback enabled: compute_every_n_epochs={dr_metrics_callback.compute_every_n_epochs}")

    # Setup logger
    if not args.fast_dev_run:
        logger = WandbLogger(
            project=config.wandb_project_name,
            name=config.model_name,
            config=config.to_flat_dict()
        )
    else:
        logger = None

    # ============================================
    # Multi-GPU / Device Configuration
    # ============================================

    # Parse devices from command-line or config
    if args.devices is not None:
        # Parse command-line devices: "0,1,2,3" -> [0,1,2,3], "-1" -> "auto"
        if args.devices == '-1':
            devices = 'auto'  # All available GPUs
        else:
            devices = [int(d.strip()) for d in args.devices.split(',')]
    elif config.devices is not None:
        devices = config.devices
    else:
        # Legacy: single device
        devices = [config.device] if config.device >= 0 else 1

    # Determine strategy
    if args.strategy is not None:
        strategy = args.strategy
    else:
        strategy = config.strategy

    # Number of nodes
    num_nodes = args.num_nodes if args.num_nodes is not None else config.num_nodes

    # Determine accelerator and final device config
    if torch.cuda.is_available() and devices != 1:
        accelerator = 'gpu'

        # Check if multi-GPU
        if devices == 'auto':
            num_gpus = torch.cuda.device_count()
            print(f"[Multi-GPU] Using ALL {num_gpus} available GPUs")
        elif isinstance(devices, list) and len(devices) > 1:
            num_gpus = len(devices)
            print(f"[Multi-GPU] Using {num_gpus} GPUs: {devices}")
        elif isinstance(devices, list) and len(devices) == 1:
            num_gpus = 1
            print(f"[Single-GPU] Using GPU: cuda:{devices[0]}")
        else:
            num_gpus = 1
            print(f"[Single-GPU] Using GPU")

        # Set strategy for multi-GPU
        # Always use DDPStrategy with find_unused_parameters=True for multi-GPU
        # This is required because replay buffer training may not use
        # all model parameters in every step
        if num_gpus > 1:
            strategy = DDPStrategy(find_unused_parameters=True)
            print(f"[Multi-GPU] Strategy: DDP (find_unused_parameters=True)")
    else:
        accelerator = 'cpu'
        devices = 1
        num_gpus = 0
        strategy = 'auto'
        if config.device >= 0:
            print("Warning: GPU requested but not available, using CPU")
        else:
            print("Using CPU")

    # For node budget batching with DDP, we disable Lightning's distributed sampler
    # because our generative dataset produces unique random graphs each call
    use_distributed_sampler = True  # Default
    if config.training.use_node_budget_batching and num_gpus > 1:
        use_distributed_sampler = False
        print(f"[Multi-GPU] Disabled distributed sampler (generative dataset with node budget batching)")

    # Disable determinism - UMAP loss uses torch.multinomial
    # which requires non-deterministic cumsum_cuda_kernel
    use_deterministic = False

    # Create Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        num_nodes=num_nodes,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=config.verbose,
        enable_model_summary=config.verbose,
        deterministic=use_deterministic,
        use_distributed_sampler=use_distributed_sampler,
        # Debugging options
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        # Gradient clipping handled in LightningModule
        gradient_clip_val=None,
    )

    # Resolve checkpoint path for resumption
    ckpt_path = None
    if args.resume_from:
        if args.resume_from == 'last':
            ckpt_path = Path(MODEL_DIRECTORY) / 'last.ckpt'
            if not ckpt_path.exists():
                print(f"Warning: last.ckpt not found at {ckpt_path}, starting from scratch")
                ckpt_path = None
        else:
            ckpt_path = Path(args.resume_from)
            if not ckpt_path.exists():
                print(f"Warning: Checkpoint not found at {ckpt_path}, starting from scratch")
                ckpt_path = None

        if ckpt_path:
            ckpt_path = str(ckpt_path)
            print(f"Resuming from checkpoint: {ckpt_path}")

    # Train the model
    print(f"Starting training: {config.model_name}")
    print(f"Dataset: {config.dataset.dataset}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.lr}")
    if config.coarsening.coarsen:
        print(f"Coarsening: enabled")
    if config.replay_buffer.use_replay_buffer:
        print(f"Replay buffer: {config.replay_buffer.replay_buffer_size}")
    print()

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Test the model (automatically loads best checkpoint if available)
    print("\nRunning final test...")
    if config.store_models and not args.fast_dev_run:
        # Test with best checkpoint
        trainer.test(model, datamodule=datamodule, ckpt_path='best')
        print(f"\nBest validation loss: {checkpoint_callback.best_model_score:.6f}")
        print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    else:
        # Test with current model
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
