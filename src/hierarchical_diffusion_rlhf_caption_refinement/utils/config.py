"""Configuration management utilities."""

import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")

    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary.
        save_path: Path to save configuration file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to {save_path}")


def set_random_seeds(seed: int = 42) -> None:
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

    logger.info(f"Set random seeds to {seed}")


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
) -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge two configuration dictionaries.

    Args:
        base_config: Base configuration.
        override_config: Configuration with override values.

    Returns:
        Merged configuration dictionary.
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters.

    Args:
        config: Configuration dictionary.

    Returns:
        True if configuration is valid.

    Raises:
        ValueError: If configuration is invalid.
    """
    required_keys = [
        "model",
        "training",
        "data",
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")

    # Validate model config
    model_config = config["model"]
    if model_config.get("hidden_dim", 0) <= 0:
        raise ValueError("hidden_dim must be positive")

    if model_config.get("num_layers", 0) <= 0:
        raise ValueError("num_layers must be positive")

    # Validate training config
    train_config = config["training"]
    if train_config.get("learning_rate", 0) <= 0:
        raise ValueError("learning_rate must be positive")

    if train_config.get("num_epochs", 0) <= 0:
        raise ValueError("num_epochs must be positive")

    if train_config.get("batch_size", 0) <= 0:
        raise ValueError("batch_size must be positive")

    logger.info("Configuration validation passed")

    return True


def get_device(device_name: str = "auto") -> torch.device:
    """Get PyTorch device.

    Args:
        device_name: Device name ('auto', 'cuda', 'cpu', 'mps').

    Returns:
        PyTorch device object.
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        device = torch.device(device_name)
        logger.info(f"Using specified device: {device_name}")

    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dictionary as string.

    Args:
        metrics: Dictionary of metric names to values.
        precision: Number of decimal places.

    Returns:
        Formatted metrics string.
    """
    formatted = []
    for name, value in sorted(metrics.items()):
        formatted.append(f"{name}: {value:.{precision}f}")

    return " | ".join(formatted)
