"""
SUNDAY LLM - Training Loop
=============================

Phase 6 Implementation.

The main training loop handles:
  1. Forward pass: model(input_ids) → logits
  2. Loss computation: cross_entropy(logits, targets)
  3. Backward pass: loss.backward() → compute gradients
  4. Gradient accumulation: sum gradients over N micro-batches
  5. Gradient clipping: prevent exploding gradients
  6. Optimizer step: update model weights
  7. LR scheduler step: adjust learning rate
  8. Logging: metrics to console + wandb
  9. Checkpointing: save model state periodically
  10. Evaluation: validate on held-out data

Mixed Precision Training (FP16):
  Uses torch.cuda.amp.autocast() for forward pass (FP16)
  and GradScaler for gradient scaling (prevents underflow).
  Saves ~50% VRAM, ~30% faster on T4.

Gradient Accumulation:
  When batch_size is limited by VRAM, we can simulate larger
  effective batches by accumulating gradients over multiple
  forward passes before each optimizer step.
  effective_batch = batch_size × grad_accum_steps
"""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.optimizer import configure_optimizer, get_cosine_lr, update_lr
from src.training.checkpoint import (
    save_checkpoint, save_best_checkpoint, find_latest_checkpoint, load_checkpoint
)
from src.training.metrics import MetricsTracker, compute_grad_norm


class Trainer:
    """
    SUNDAY Model Trainer.

    Implements the full training loop with:
      - Mixed precision (FP16) via torch.cuda.amp
      - Gradient accumulation for simulating larger batches
      - Cosine LR schedule with warmup
      - Periodic validation and checkpointing
      - Weights & Biases logging

    Args:
        model: The SundayModel to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Training configuration dict. See configs/training_config.py.
        device: Device to train on ('cpu', 'cuda', 'cuda:0').

    Example:
        >>> from configs.training_config import TRAINING_CONFIG_TINY
        >>> trainer = Trainer(model, train_loader, val_loader,
        ...                   TRAINING_CONFIG_TINY, device='cuda')
        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cpu",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device)

        # Move model to device
        self.model.to(self.device)

        # --- Optimizer ---
        self.optimizer = configure_optimizer(
            model=self.model,
            learning_rate=config["max_lr"],
            weight_decay=config.get("weight_decay", 0.1),
            beta1=config.get("beta1", 0.9),
            beta2=config.get("beta2", 0.95),
        )

        # --- Mixed Precision Scaler (only for CUDA) ---
        self.use_amp = (self.device.type == "cuda") and config.get("use_amp", True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # --- Training State ---
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # --- Metrics ---
        self.metrics = MetricsTracker(window_size=100)

        # --- Config shortcuts ---
        self.max_steps = config["max_steps"]
        self.warmup_steps = config.get("warmup_steps", 2000)
        self.max_lr = config["max_lr"]
        self.min_lr = config.get("min_lr", self.max_lr / 10)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.grad_accum_steps = config.get("grad_accum_steps", 1)
        self.val_interval = config.get("val_interval", 500)
        self.save_interval = config.get("save_interval", 1000)
        self.log_interval = config.get("log_interval", 100)
        self.save_dir = config.get("save_dir", "checkpoints")

        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Mixed precision: {self.use_amp}")
        print(f"[Trainer] Grad accumulation: {self.grad_accum_steps} steps")
        print(f"[Trainer] Effective batch: {config.get('batch_size', '?')} × {self.grad_accum_steps}")

    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Run the full training loop.

        Args:
            resume_from: Path to checkpoint to resume from.
                         If None, tries to auto-detect latest checkpoint.
        """
        # --- Resume from checkpoint if requested ---
        if resume_from is None:
            resume_from = find_latest_checkpoint(self.save_dir)

        if resume_from and os.path.exists(resume_from):
            ckpt = load_checkpoint(
                resume_from, self.model, self.optimizer, self.scaler,
                device=str(self.device)
            )
            self.step = ckpt.get("step", 0)
            self.epoch = ckpt.get("epoch", 0)
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

        # --- Initialize W&B ---
        self._init_wandb()

        print(f"\n[Trainer] Starting training from step {self.step:,}/{self.max_steps:,}")
        print(f"[Trainer] {'='*50}")

        # --- Training Loop ---
        self.model.train()
        self.metrics.start_time = time.time()

        while self.step < self.max_steps:
            self.epoch += 1

            for batch_idx, (input_ids, target_ids) in enumerate(self.train_loader):
                if self.step >= self.max_steps:
                    break

                # Train one step (possibly with grad accumulation)
                loss, grad_norm = self._train_step(
                    input_ids, target_ids, batch_idx
                )

                # --- Update LR ---
                lr = get_cosine_lr(
                    self.step, self.max_lr, self.min_lr,
                    self.warmup_steps, self.max_steps
                )
                update_lr(self.optimizer, lr)

                # --- Log ---
                tokens_per_step = (
                    input_ids.shape[0] * input_ids.shape[1] * self.grad_accum_steps
                )
                self.metrics.update(loss, lr, grad_norm, tokens_per_step)
                self.metrics.log_step(
                    self.step, self.max_steps, loss, lr,
                    grad_norm, tokens_per_step, self.log_interval
                )
                self.metrics.log_to_wandb(self.step, loss, lr, grad_norm)

                # --- Validation ---
                if self.step % self.val_interval == 0:
                    val_loss = self.validate()
                    self.metrics.log_validation(self.step, val_loss)
                    self.metrics.log_to_wandb(self.step, loss, lr, grad_norm, val_loss)

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_best_checkpoint(
                            self.model, self.model.config,
                            val_loss, self.save_dir
                        )

                    self.model.train()

                # --- Checkpoint ---
                if self.step % self.save_interval == 0:
                    save_checkpoint(
                        self.model, self.optimizer, self.step, self.epoch,
                        self.model.config,
                        self.metrics.get_smoothed_loss(),
                        self.best_val_loss,
                        self.best_val_loss,
                        self.save_dir,
                        self.scaler if self.use_amp else None,
                    )

        print("\n[Trainer] Training complete!")
        print(f"[Trainer] Best val_loss: {self.best_val_loss:.4f}")

        try:
            import wandb
            if wandb.run:
                wandb.finish()
        except ImportError:
            pass

    def _train_step(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        batch_idx: int,
    ):
        """
        Run one training step (or accumulate gradients).

        Returns:
            (loss, grad_norm) after the optimizer step.
        """
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Gradient accumulation: only step optimizer every N mini-batches
        is_last_accum_step = (batch_idx + 1) % self.grad_accum_steps == 0

        # --- Forward + Loss (with optional FP16) ---
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss, _ = self.model(input_ids, targets=target_ids)
            # Scale loss for gradient accumulation
            loss = loss / self.grad_accum_steps

        # --- Backward ---
        self.scaler.scale(loss).backward()

        grad_norm = 0.0

        if is_last_accum_step:
            # --- Gradient Clipping ---
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip
            ).item()

            # --- Optimizer Step ---
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.step += 1

        return loss.item() * self.grad_accum_steps, grad_norm

    @torch.no_grad()
    def validate(self) -> float:
        """
        Evaluate the model on the validation set.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for input_ids, target_ids in self.val_loader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss, _ = self.model(input_ids, targets=target_ids)

            total_loss += loss.item()
            n_batches += 1

            # Limit validation for speed (evaluate on first 100 batches max)
            if n_batches >= 100:
                break

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging if configured."""
        if not self.config.get("wandb_enabled", False):
            return

        try:
            import wandb
            wandb.init(
                project=self.config.get("wandb_project", "SUNDAY_AI"),
                name=self.config.get("wandb_run_name", None),
                config={**self.config, **self.model.config},
                resume="allow",
            )
            print(f"[W&B] Logging to: {wandb.run.url}")
        except ImportError:
            print("[W&B] wandb not installed, skipping")
        except Exception as e:
            print(f"[W&B] Failed to initialize: {e}")
