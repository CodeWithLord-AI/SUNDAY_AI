"""
SUNDAY LLM - Supervised Fine-Tuning (SFT) Trainer
=====================================================

Phase 9 Implementation.

Fine-tunes a pretrained SUNDAY model on instruction-following data
(Alpaca format) using either:

  1. Full fine-tuning: Update all model parameters
  2. LoRA fine-tuning: Freeze base model, only train LoRA adapters

The key difference from pretraining:
  - Pretraining: loss on ALL tokens (learn language)
  - Fine-tuning:  loss only on RESPONSE tokens (learn instructions)
  - Labels have -1 for prompt tokens → ignored in cross_entropy
"""

import os
import time
import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader

from src.training.optimizer import configure_optimizer, get_cosine_lr, update_lr
from src.training.checkpoint import save_checkpoint, save_best_checkpoint
from src.training.metrics import MetricsTracker, compute_grad_norm


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for SUNDAY LLM.

    Trains a pretrained model to follow instructions using
    the Alpaca dataset format.

    Key difference from Trainer:
      - Labels may be -1 (masked prompt tokens)
      - Cross-entropy ignores -1 labels
      - Typically uses lower LR than pretraining
      - Shorter training (3-5 epochs, not 100K steps)

    Args:
        model: Pretrained SundayModel.
        train_loader: Training DataLoader (yields input_ids, labels).
        val_loader: Validation DataLoader.
        config: Fine-tuning config dict. See configs/training_config.py.
        device: Training device.
        use_lora: Whether model has LoRA adapters (affects optimizer setup).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cpu",
        use_lora: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device)
        self.use_lora = use_lora

        # Move model to device
        self.model.to(self.device)

        # --- Optimizer (only trainable params) ---
        self.optimizer = configure_optimizer(
            model=self.model,
            learning_rate=config.get("learning_rate", 2e-5),
            weight_decay=config.get("weight_decay", 0.01),
            beta1=config.get("beta1", 0.9),
            beta2=config.get("beta2", 0.999),
        )

        # --- Mixed Precision ---
        self.use_amp = (self.device.type == "cuda") and config.get("mixed_precision", True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # --- Training State ---
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.metrics = MetricsTracker(window_size=50)

        # --- Config shortcuts ---
        self.max_epochs = config.get("max_epochs", 3)
        self.max_lr = config.get("learning_rate", 2e-5)
        self.min_lr = config.get("min_lr", 1e-6)
        self.warmup_steps = config.get("warmup_steps", 100)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.save_dir = config.get("checkpoint_dir", "checkpoints/finetuned")
        self.log_interval = config.get("log_every", 10)
        self.val_interval = config.get("eval_every", 200)

        total_steps = len(train_loader) * self.max_epochs
        self.max_steps = config.get("max_steps") or total_steps

        print(f"[SFT] Fine-tuning for {self.max_epochs} epochs / {self.max_steps:,} steps")
        print(f"[SFT] LoRA mode: {use_lora}")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[SFT] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def train(self) -> None:
        """Run the full fine-tuning loop."""
        print(f"\n[SFT] Starting fine-tuning...")
        self.model.train()
        self.metrics.start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            self.epoch = epoch
            print(f"\n[SFT] Epoch {epoch}/{self.max_epochs}")

            epoch_loss = 0.0
            n_batches = 0

            for batch_idx, (input_ids, labels) in enumerate(self.train_loader):
                if self.step >= self.max_steps:
                    break

                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                # --- Forward + Loss ---
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Pass labels directly — model uses cross_entropy with ignore_index=-1
                    loss, _ = self.model(input_ids, targets=labels)

                # --- Backward ---
                self.scaler.scale(loss).backward()

                # --- Gradient Clipping + Optimizer Step ---
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                ).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                self.step += 1
                epoch_loss += loss.item()
                n_batches += 1

                # --- LR Update ---
                lr = get_cosine_lr(
                    self.step, self.max_lr, self.min_lr,
                    self.warmup_steps, self.max_steps
                )
                update_lr(self.optimizer, lr)

                # --- Log ---
                if self.step % self.log_interval == 0:
                    self.metrics.update(loss.item(), lr, grad_norm, input_ids.numel())
                    print(
                        f"  Step {self.step:>5} | "
                        f"loss={loss.item():.4f} | "
                        f"lr={lr:.2e} | "
                        f"|g|={grad_norm:.3f}"
                    )

                # --- Validation ---
                if self.step % self.val_interval == 0:
                    val_loss = self._validate()
                    print(f"\n  [Val] step={self.step} | val_loss={val_loss:.4f}\n")

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_best_checkpoint(
                            self.model, getattr(self.model, 'config', {}),
                            val_loss, self.save_dir
                        )
                    self.model.train()

            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            print(f"[SFT] Epoch {epoch} complete | avg_loss={avg_epoch_loss:.4f}")

        print(f"\n[SFT] Fine-tuning complete! Best val_loss: {self.best_val_loss:.4f}")

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for input_ids, labels in self.val_loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss, _ = self.model(input_ids, targets=labels)

            total_loss += loss.item()
            n_batches += 1

            if n_batches >= 50:
                break

        return total_loss / max(n_batches, 1)
