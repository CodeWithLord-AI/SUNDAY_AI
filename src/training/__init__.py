"""
SUNDAY Training Package
========================

Exports for the training pipeline.

Usage:
    from src.training import Trainer
    from src.training.optimizer import configure_optimizer
    from src.training.checkpoint import save_checkpoint, load_checkpoint
    from src.training.metrics import MetricsTracker
"""

from src.training.trainer import Trainer
from src.training.optimizer import configure_optimizer, get_cosine_lr, update_lr
from src.training.checkpoint import (
    save_checkpoint, save_best_checkpoint,
    load_checkpoint, find_latest_checkpoint
)
from src.training.metrics import MetricsTracker, compute_grad_norm

__all__ = [
    "Trainer",
    "configure_optimizer",
    "get_cosine_lr",
    "update_lr",
    "save_checkpoint",
    "save_best_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    "MetricsTracker",
    "compute_grad_norm",
]
