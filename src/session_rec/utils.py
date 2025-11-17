"""Utility functions for session-based recommendations."""

import random
import numpy as np
import torch
from typing import Any, Dict, Optional
import yaml
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save the configuration file.
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device() -> torch.device:
    """Get the appropriate device for computation.
    
    Returns:
        PyTorch device (cuda if available, else cpu).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
