"""
SUNDAY LLM - Checkpoint Management
=====================================

Phase 6 Implementation.

Save and load model checkpoints during training.

Checkpoint format (a dict saved with torch.save):
  {
    "step": int,                    # Current training step
    "epoch": int,                   # Current epoch
    "model_state_dict": OrderedDict, # Model weights
    "optimizer_state_dict": dict,    # Optimizer state (momenta)
    "scaler_state_dict": dict,       # GradScaler state (for FP16)
    "config": dict,                  # Model config
    "train_loss": float,             # Last training loss
    "val_loss": float,               # Last validation loss
    "best_val_loss": float,          # Best validation loss so far
  }

Why save optimizer state?
  Adam optimizer maintains per-parameter moving averages (m, v).
  Without these, resuming training would start the optimizer from scratch,
  causing a few hundred "warm-up" steps where gradients look weird.
"""

import os
import glob
from typing import Optional, Tuple

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    config: dict,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    save_dir: str = "checkpoints",
    scaler: Optional[object] = None,
    keep_last_n: int = 3,
) -> str:
    """
    Save a training checkpoint to disk.

    Args:
        model: The SUNDAY model.
        optimizer: The AdamW optimizer.
        step: Current training step.
        epoch: Current epoch.
        config: Model configuration dict.
        train_loss: Most recent training loss.
        val_loss: Most recent validation loss.
        best_val_loss: Best validation loss seen so far.
        save_dir: Directory to save checkpoints.
        scaler: Optional torch.cuda.amp.GradScaler for FP16.
        keep_last_n: Keep only the last N checkpoints (save disk space).

    Returns:
        Path to the saved checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    filename = f"checkpoint_step{step:07d}_loss{val_loss:.4f}.pt"
    filepath = os.path.join(save_dir, filename)

    torch.save(checkpoint, filepath)

    # --- Force OS buffer flush so Google Drive FUSE actually writes to cloud ---
    # Without this, Colab may crash before Drive syncs the buffered bytes.
    try:
        import sys
        if sys.platform != "win32":          # os.sync() is Linux/macOS only
            os.sync()
    except Exception:
        pass

    print(f"[Checkpoint] Saved: {filepath}")

    # --- Keep only the last N checkpoints ---
    if keep_last_n > 0:
        _prune_old_checkpoints(save_dir, keep_last_n, exclude=filepath)

    return filepath


def save_best_checkpoint(
    model: nn.Module,
    config: dict,
    val_loss: float,
    save_dir: str = "checkpoints",
) -> str:
    """
    Save the best model checkpoint (model weights only, no optimizer state).

    This is a compact checkpoint for inference/deployment.

    Args:
        model: The SUNDAY model.
        config: Model configuration dict.
        val_loss: Validation loss at this checkpoint.
        save_dir: Directory to save.

    Returns:
        Path to the best checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "best_model.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "val_loss": val_loss,
    }, filepath)

    # --- Force OS buffer flush so Google Drive FUSE actually writes to cloud ---
    try:
        import sys
        if sys.platform != "win32":
            os.sync()
    except Exception:
        pass

    print(f"[Checkpoint] New best model saved: {filepath} (val_loss={val_loss:.4f})")
    return filepath


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[object] = None,
    device: str = "cpu",
) -> dict:
    """
    Load a checkpoint from disk.

    Args:
        filepath: Path to the .pt checkpoint file.
        model: The model to load weights into.
        optimizer: Optional optimizer to restore state into.
        scaler: Optional GradScaler to restore state into.
        device: Device to map tensors to ('cpu', 'cuda', 'cuda:0').

    Returns:
        The checkpoint dict (contains step, epoch, losses, etc.)

    Example:
        >>> ckpt = load_checkpoint("checkpoints/best_model.pt", model)
        >>> print(f"Resuming from step {ckpt['step']}")
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    print(f"[Checkpoint] Loading: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Checkpoint] Model weights loaded")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[Checkpoint] Optimizer state loaded")

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"[Checkpoint] GradScaler state loaded")

    step = checkpoint.get("step", 0)
    val_loss = checkpoint.get("val_loss", float("inf"))
    print(f"[Checkpoint] Resumed from step {step:,}, val_loss={val_loss:.4f}")

    return checkpoint


def find_latest_checkpoint(save_dir: str = "checkpoints") -> Optional[str]:
    """
    Find the most recent checkpoint file in a directory.

    Args:
        save_dir: Directory to search.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.
    """
    pattern = os.path.join(save_dir, "checkpoint_step*.pt")
    checkpoints = sorted(glob.glob(pattern))

    if not checkpoints:
        return None

    latest = checkpoints[-1]
    print(f"[Checkpoint] Found latest: {latest}")
    return latest


def _prune_old_checkpoints(
    save_dir: str,
    keep_last_n: int,
    exclude: Optional[str] = None,
) -> None:
    """Remove old checkpoints, keeping only the last N."""
    pattern = os.path.join(save_dir, "checkpoint_step*.pt")
    checkpoints = sorted(glob.glob(pattern))

    # Filter out the one we just saved and best_model.pt
    to_consider = [c for c in checkpoints if c != exclude and "best_model" not in c]

    # Remove oldest if we have too many
    while len(to_consider) > keep_last_n - 1:
        old = to_consider.pop(0)
        os.remove(old)
        print(f"[Checkpoint] Removed old checkpoint: {old}")
