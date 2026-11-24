"""
SUNDAY LLM - Training Metrics
================================

Phase 6 Implementation.

Track and log training metrics:
  - Loss (train and validation)
  - Perplexity = exp(loss)
  - Learning rate over time
  - Tokens per second (throughput)
  - Gradient norm (for monitoring training health)

Integrations:
  - Console logging (always)
  - Weights & Biases (when WANDB_ENABLED=True)
"""

import time
import math
from typing import Dict, List, Optional
from collections import deque


class MetricsTracker:
    """
    Track training metrics and compute running statistics.

    Maintains a sliding window of recent loss values for
    smooth logging, and accumulates metrics for W&B.

    Args:
        window_size: Number of steps for the rolling average.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.loss_window = deque(maxlen=window_size)
        self.step = 0
        self.epoch = 0
        self.start_time = time.time()
        self.step_start_time = time.time()

        # History for plotting
        self.history: Dict[str, List] = {
            "step": [],
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "grad_norm": [],
            "tokens_per_sec": [],
        }

    def update(
        self,
        loss: float,
        lr: float,
        grad_norm: float,
        tokens_per_step: int,
    ) -> None:
        """
        Record a training step's metrics.

        Args:
            loss: Training loss for this step.
            lr: Current learning rate.
            grad_norm: Gradient norm after clipping.
            tokens_per_step: Number of tokens processed this step.
        """
        self.step += 1
        self.loss_window.append(loss)

        # Compute tokens per second
        now = time.time()
        dt = now - self.step_start_time
        self.step_start_time = now
        tokens_per_sec = tokens_per_step / max(dt, 1e-8)

        # Store in history
        self.history["step"].append(self.step)
        self.history["train_loss"].append(loss)
        self.history["lr"].append(lr)
        self.history["grad_norm"].append(grad_norm)
        self.history["tokens_per_sec"].append(tokens_per_sec)

    def get_smoothed_loss(self) -> float:
        """Return the rolling average loss over the last window_size steps."""
        if not self.loss_window:
            return float("inf")
        return sum(self.loss_window) / len(self.loss_window)

    def get_perplexity(self, loss: Optional[float] = None) -> float:
        """
        Compute perplexity from loss.

        Perplexity = exp(loss)
        Lower = better. A perplexity of K means the model is
        as uncertain as if it had to choose among K options at every step.

        Args:
            loss: If None, uses the current smoothed loss.

        Returns:
            Perplexity value.
        """
        loss = loss if loss is not None else self.get_smoothed_loss()
        try:
            return math.exp(loss)
        except OverflowError:
            return float("inf")

    def log_step(
        self,
        step: int,
        total_steps: int,
        loss: float,
        lr: float,
        grad_norm: float,
        tokens_per_step: int,
        log_interval: int = 100,
    ) -> None:
        """
        Print a training step summary to console.

        Args:
            step: Current step.
            total_steps: Total training steps.
            loss: Current step loss.
            lr: Current learning rate.
            grad_norm: Gradient norm.
            tokens_per_step: Tokens processed this step.
            log_interval: Only print every N steps.
        """
        if step % log_interval != 0:
            return

        elapsed = time.time() - self.start_time
        smoothed_loss = self.get_smoothed_loss()
        ppl = self.get_perplexity(smoothed_loss)

        # Time estimate
        progress = step / max(total_steps, 1)
        eta_sec = (elapsed / max(progress, 1e-8)) * (1 - progress)
        eta_min = eta_sec / 60

        print(
            f"Step {step:>7,}/{total_steps:,} | "
            f"loss={smoothed_loss:.4f} | "
            f"ppl={ppl:.1f} | "
            f"lr={lr:.2e} | "
            f"|g|={grad_norm:.3f} | "
            f"ETA: {eta_min:.0f}m"
        )

    def log_validation(self, step: int, val_loss: float) -> None:
        """
        Print a validation summary.

        Args:
            step: Current training step.
            val_loss: Validation loss.
        """
        val_ppl = self.get_perplexity(val_loss)
        self.history["val_loss"].append(val_loss)

        print(
            f"\n{'='*55}\n"
            f"[Validation] Step {step:,} | "
            f"val_loss={val_loss:.4f} | "
            f"val_ppl={val_ppl:.1f}\n"
            f"{'='*55}\n"
        )

    def log_to_wandb(
        self,
        step: int,
        train_loss: float,
        lr: float,
        grad_norm: float,
        val_loss: Optional[float] = None,
    ) -> None:
        """
        Log metrics to Weights & Biases.

        Args:
            step: Current step.
            train_loss: Training loss.
            lr: Learning rate.
            grad_norm: Gradient norm.
            val_loss: Optional validation loss.
        """
        try:
            import wandb
            if wandb.run is None:
                return

            metrics = {
                "train/loss": train_loss,
                "train/perplexity": self.get_perplexity(train_loss),
                "train/lr": lr,
                "train/grad_norm": grad_norm,
            }
            if val_loss is not None:
                metrics["val/loss"] = val_loss
                metrics["val/perplexity"] = self.get_perplexity(val_loss)

            wandb.log(metrics, step=step)

        except ImportError:
            pass  # wandb not installed, skip silently


def compute_grad_norm(model: "torch.nn.Module") -> float:
    """
    Compute the total gradient norm across all parameters.

    Used to monitor training health and after gradient clipping.
    If grad_norm is very large (>10), gradients may be exploding.
    If grad_norm is near zero, gradients may be vanishing.

    Args:
        model: The model with computed gradients.

    Returns:
        L2 norm of all gradients (float).
    """
    import torch
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
