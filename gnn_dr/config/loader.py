"""
Configuration loader for CoRe-GD.

Supports loading from:
- YAML files (preferred, structured)
- JSON files (backward compatibility)
- Dictionary objects (programmatic usage)
"""

import json
import yaml
from pathlib import Path
from typing import Union
from dataclasses import asdict

from .config import ExperimentConfig


def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load configuration from file (YAML or JSON).
    
    Args:
        config_path: Path to config file (.yaml or .json)
        
    Returns:
        ExperimentConfig: Loaded configuration object
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load based on file extension
    if config_path.suffix in ['.yaml', '.yml']:
        return load_yaml_config(config_path)
    elif config_path.suffix == '.json':
        return load_json_config(config_path)
    else:
        raise ValueError(
            f"Unsupported config format: {config_path.suffix}. "
            "Use .yaml, .yml, or .json"
        )


def load_yaml_config(yaml_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML config file
        
    Returns:
        ExperimentConfig: Loaded configuration object
    """
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # YAML files are structured, so we can load directly
    return dict_to_config(config_dict)


def load_json_config(json_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load configuration from JSON file (backward compatibility).
    
    Args:
        json_path: Path to JSON config file
        
    Returns:
        ExperimentConfig: Loaded configuration object
    """
    with open(json_path, 'r') as f:
        flat_dict = json.load(f)
    
    # JSON files are flat, use from_flat_dict
    return ExperimentConfig.from_flat_dict(flat_dict)


def dict_to_config(config_dict: dict) -> ExperimentConfig:
    """
    Convert dictionary to ExperimentConfig.
    
    Handles both structured (nested) and flat dictionaries.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        ExperimentConfig: Configuration object
    """
    # Check if dictionary is structured (has nested keys like 'model', 'training')
    if 'model' in config_dict or 'training' in config_dict:
        # Structured config from YAML
        from .config import (
            ModelConfig, TrainingConfig, DatasetConfig,
            CoarseningConfig, ReplayBufferConfig, ValidationConfig,
            LoggingConfig, DimensionalityReductionConfig, MultiSizeEvaluationConfig
        )
        
        return ExperimentConfig(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            dataset=DatasetConfig(**config_dict.get('dataset', {})),
            coarsening=CoarseningConfig(**config_dict.get('coarsening', {})),
            replay_buffer=ReplayBufferConfig(**config_dict.get('replay_buffer', {})),
            validation=ValidationConfig(**config_dict.get('validation', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            dimensionality_reduction=DimensionalityReductionConfig(**config_dict.get('dimensionality_reduction', {})),
            multi_size_evaluation=MultiSizeEvaluationConfig(**config_dict.get('multi_size_evaluation', {})),
            device=config_dict.get('device', 0),
            verbose=config_dict.get('verbose', True),
            use_cupy=config_dict.get('use_cupy', False),
            # Multi-GPU configuration
            devices=config_dict.get('devices', None),
            strategy=config_dict.get('strategy', 'auto'),
            num_nodes=config_dict.get('num_nodes', 1),
            model_name=config_dict.get('model_name', 'CoRe-GD'),
            wandb_project_name=config_dict.get('wandb_project_name', 'CoRe-GD'),
            store_models=config_dict.get('store_models', True),
        )
    else:
        # Flat config from JSON or old format
        return ExperimentConfig.from_flat_dict(config_dict)


def save_config(config: ExperimentConfig, output_path: Union[str, Path], 
                format: str = 'yaml'):
    """
    Save configuration to file.
    
    Args:
        config: Configuration object to save
        output_path: Path to save config file
        format: Output format ('yaml' or 'json')
    """
    output_path = Path(output_path)
    
    if format == 'yaml':
        # Save as structured YAML
        config_dict = {
            'model': asdict(config.model),
            'training': asdict(config.training),
            'dataset': asdict(config.dataset),
            'coarsening': asdict(config.coarsening),
            'replay_buffer': asdict(config.replay_buffer),
            'validation': asdict(config.validation),
            'logging': asdict(config.logging),
            'dimensionality_reduction': asdict(config.dimensionality_reduction),
            'device': config.device,
            'verbose': config.verbose,
            'use_cupy': config.use_cupy,
            'model_name': config.model_name,
            'wandb_project_name': config.wandb_project_name,
            'store_models': config.store_models,
        }
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    elif format == 'json':
        # Save as flat JSON (backward compatibility)
        flat_dict = config.to_flat_dict()
        with open(output_path, 'w') as f:
            json.dump(flat_dict, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")


def config_to_namespace(config: ExperimentConfig):
    """
    Convert ExperimentConfig to a namespace-like object for backward compatibility.
    
    This allows the config to be used with code expecting argparse.Namespace
    or test_tube.HyperOptArgumentParser results.
    
    Args:
        config: ExperimentConfig object
        
    Returns:
        SimpleNamespace: Namespace with flat config attributes
    """
    from types import SimpleNamespace
    return SimpleNamespace(**config.to_flat_dict())
