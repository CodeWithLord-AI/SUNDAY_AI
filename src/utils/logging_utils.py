"""
SUNDAY LLM - Logging Utilities
================================

Why do we need custom logging?
  During training, we need to track:
  - Loss values (training + validation)
  - Learning rate changes
  - GPU memory usage
  - Training speed (tokens/second, steps/second)
  - Generated text samples

  We use Python's logging module for console/file output,
  and Weights & Biases (wandb) for cloud experiment tracking.

What this module does:
  - Sets up a consistent logger for the project
  - Provides helpers for logging training metrics
  - Integrates with wandb for experiment tracking
"""

import logging
import os
import sys
from typing import Optional


def get_logger(
    name: str = "SUNDAY",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create a configured logger.

    Args:
        name: Logger name (appears in log messages)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs to

    Returns:
        Configured logging.Logger object

    Example:
        >>> logger = get_logger("SUNDAY.training")
        >>> logger.info("Training started")
        [2024-01-15 10:30:00] [INFO] [SUNDAY.training] Training started
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Format: [timestamp] [level] [module] message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
