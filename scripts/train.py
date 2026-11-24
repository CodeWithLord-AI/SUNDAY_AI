"""
SUNDAY LLM - CLI Training Script
===================================

Trains SUNDAY from scratch (or resumes from checkpoint).

Usage:
  # Quick smoke-test with tiny model
  python scripts/train.py --model_size tiny --steps 100

  # Full 50M pretraining
  python scripts/train.py --model_size 50m --data_dir data/processed --steps 50000

  # Resume from latest checkpoint
  python scripts/train.py --model_size 50m --resume

  # Use custom save directory
  python scripts/train.py --model_size 50m --save_dir runs/experiment_1
"""

import argparse
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src` and `configs` are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.model_config import get_model_config
from configs.training_config import TRAINING_CONFIG
from src.model.sunday_model import SundayModel
from src.tokenizer.tokenizer import SundayTokenizer
from src.data.dataset import SundayDataset
from src.data.dataloader import create_dataloaders, create_dataloader_from_datasets
from src.training.trainer import Trainer
from src.training.checkpoint import find_latest_checkpoint


# ---------------------------------------------------------------------------
# Config key normalisation
# TRAINING_CONFIG uses different key names than what Trainer expects.
# This function builds a Trainer-compatible config from TRAINING_CONFIG.
# ---------------------------------------------------------------------------
def build_trainer_config(base: dict, args) -> dict:
    """
    Map TRAINING_CONFIG keys → Trainer keys and apply CLI overrides.

    TRAINING_CONFIG key      Trainer key
    ----------------------   ------------------
    learning_rate            max_lr
    micro_batch_size         batch_size
    grad_accumulation_steps  grad_accum_steps
    eval_every               val_interval
    checkpoint_every         save_interval
    log_every                log_interval
    use_wandb                wandb_enabled
    checkpoint_dir           save_dir
    """
    cfg = dict(base)                    # copy so we don't mutate the original

    # Rename keys
    renames = {
        "learning_rate":          "max_lr",
        "micro_batch_size":       "batch_size",
        "grad_accumulation_steps":"grad_accum_steps",
        "eval_every":             "val_interval",
        "checkpoint_every":       "save_interval",
        "log_every":              "log_interval",
        "use_wandb":              "wandb_enabled",
        "checkpoint_dir":         "save_dir",
        "mixed_precision":        "use_amp",
    }
    for old, new in renames.items():
        if old in cfg and new not in cfg:
            cfg[new] = cfg.pop(old)

    # Ensure required keys always present
    cfg.setdefault("max_lr",          3e-4)
    cfg.setdefault("min_lr",          1e-5)
    cfg.setdefault("warmup_steps",    2000)
    cfg.setdefault("max_steps",       50000)
    cfg.setdefault("batch_size",      4)
    cfg.setdefault("grad_accum_steps",8)
    cfg.setdefault("weight_decay",    0.1)
    cfg.setdefault("beta1",           0.9)
    cfg.setdefault("beta2",           0.95)
    cfg.setdefault("grad_clip",       1.0)
    cfg.setdefault("val_interval",    500)
    cfg.setdefault("save_interval",   1000)
    cfg.setdefault("log_interval",    10)
    cfg.setdefault("save_dir",        "checkpoints")
    cfg.setdefault("wandb_enabled",   False)
    cfg.setdefault("wandb_project",   "SUNDAY_AI")
    cfg.setdefault("use_amp",         True)

    # Apply CLI overrides
    if args.steps is not None:
        cfg["max_steps"] = args.steps
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    if args.no_wandb:
        cfg["wandb_enabled"] = False

    return cfg


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("[Train] ⚠ No GPU detected — training on CPU (slow!)")
    return device


# ---------------------------------------------------------------------------
# Build datasets / dataloaders
# ---------------------------------------------------------------------------
def build_dataloaders(args, tokenizer, model_config, train_cfg):
    """
    Returns (train_loader, val_loader).

    If --data_dir is given, reads pre-tokenised binary files from it.
    Otherwise falls back to a tiny synthetic dataset for smoke-testing.
    """
    batch_size = train_cfg["batch_size"]
    seq_len    = model_config["max_seq_len"]

    if args.data_dir and os.path.isdir(args.data_dir):
        # Look for train.bin / val.bin
        train_bin = os.path.join(args.data_dir, "train.bin")
        val_bin   = os.path.join(args.data_dir, "val.bin")

        if os.path.exists(train_bin) and os.path.exists(val_bin):
            from src.data.dataset import SundayDatasetFromFile
            train_ds = SundayDatasetFromFile(train_bin, max_seq_len=seq_len)
            val_ds   = SundayDatasetFromFile(val_bin,   max_seq_len=seq_len)
            print(f"[Train] Loaded binary datasets — train: {len(train_ds)}, val: {len(val_ds)}")
        else:
            raise FileNotFoundError(
                f"Expected train.bin and val.bin inside {args.data_dir}"
            )
    else:
        # Synthetic fallback: 10K random tokens for smoke-testing
        print("[Train] No --data_dir provided. Using synthetic data for smoke-test.")
        import torch
        fake_tokens = torch.randint(0, tokenizer.vocab_size, (10_000,)).tolist()
        train_ds = SundayDataset(fake_tokens, max_seq_len=seq_len)
        val_ds   = SundayDataset(fake_tokens[:1000], max_seq_len=seq_len)

    # Use create_dataloader_from_datasets for pre-split datasets
    train_loader, val_loader = create_dataloader_from_datasets(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=batch_size,
        num_workers=0,          # Windows: keep 0 to avoid multiprocessing issues
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train SUNDAY LLM from scratch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_size", default="tiny",
        choices=["tiny", "50m", "125m"],
        help="Model configuration to use.",
    )
    parser.add_argument(
        "--data_dir", default=None,
        help="Path to directory containing train.bin and val.bin.",
    )
    parser.add_argument(
        "--save_dir", default=None,
        help="Directory to save checkpoints (overrides config).",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Total training steps (overrides config).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint in --save_dir.",
    )
    parser.add_argument(
        "--tokenizer_path", default="tokenizer_files",
        help="Path to trained tokenizer files.",
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable Weights & Biases logging.",
    )
    args = parser.parse_args()

    # ----- Model config -----
    model_config = get_model_config(args.model_size)
    print(f"\n[Train] Model: {model_config['model_name']}")

    # ----- Training config -----
    train_cfg = build_trainer_config(TRAINING_CONFIG, args)
    print(f"[Train] Steps: {train_cfg['max_steps']}, LR: {train_cfg['max_lr']}")
    print(f"[Train] Batch: {train_cfg['batch_size']} × {train_cfg['grad_accum_steps']} = "
          f"{train_cfg['batch_size'] * train_cfg['grad_accum_steps']} effective")

    # ----- Tokenizer -----
    tokenizer = SundayTokenizer()
    if os.path.exists(args.tokenizer_path):
        tokenizer.load(args.tokenizer_path)
        print(f"[Train] Tokenizer loaded — vocab size: {tokenizer.vocab_size}")
    else:
        print(f"[Train] ⚠ Tokenizer not found at '{args.tokenizer_path}' — using untrained tokenizer")

    # ----- Device -----
    device = get_device()

    # ----- Model -----
    model = SundayModel(model_config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Parameters: {n_params / 1e6:.2f}M")

    # ----- Data -----
    train_loader, val_loader = build_dataloaders(args, tokenizer, model_config, train_cfg)

    # ----- Trainer -----
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=device,
    )

    # ----- Resume -----
    resume_path = None
    if args.resume:
        resume_path = find_latest_checkpoint(train_cfg["save_dir"])
        if resume_path:
            print(f"[Train] Resuming from: {resume_path}")
        else:
            print("[Train] No checkpoint found — starting fresh.")

    # ----- Train -----
    start = time.time()
    try:
        trainer.train(resume_from=resume_path)
    except KeyboardInterrupt:
        print("\n[Train] Interrupted by user. Saving emergency checkpoint...")
        from src.training.checkpoint import save_checkpoint
        save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            step=trainer.step,
            loss=float("nan"),
            config=train_cfg,
            save_dir=train_cfg["save_dir"],
            filename="interrupted.pt",
        )
        print("[Train] Emergency checkpoint saved.")

    elapsed = time.time() - start
    print(f"\n[Train] Finished in {elapsed/3600:.2f}h")


if __name__ == "__main__":
    main()
