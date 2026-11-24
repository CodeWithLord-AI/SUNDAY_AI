"""
SUNDAY LLM - Training Configuration
=====================================

Hyperparameters for pretraining and fine-tuning.

Key training concepts:
- AdpamW otimizer: Adam with decoupled weight decay (better for transformers)
- Cosine LR schedule: Learning rate decays following cosine curve after warmup
- Mixed precision (FP16): Halves memory usage on T4 GPU
- Gradient accumulation: Simulates larger batch sizes with limited VRAM
- Gradient clipping: Prevents exploding gradients (max norm = 1.0)

VRAM Budget (125M model, FP16, batch=4, seq=512):
  Model params (FP16):    ~0.22 GB
  Optimizer states (FP32): ~1.33 GB
  Gradients (FP16):       ~0.22 GB
  Activations:            ~3-6 GB
  CUDA overhead:          ~2-3 GB
  Total:                  ~7-12 GB  (T4 has 15.1 GB)
"""


# ============================================
# Pretraining Configuration
# ============================================
TRAINING_CONFIG = {
    # --- Optimizer ---
    "optimizer": "AdamW",
    "learning_rate": 3e-4,          # Peak learning rate
    "weight_decay": 0.1,            # L2 regularization (decoupled)
    "beta1": 0.9,                   # Adam first moment decay
    "beta2": 0.95,                  # Adam second moment decay (0.95 for LLMs)
    "eps": 1e-8,                    # Adam epsilon for numerical stability

    # --- Learning Rate Schedule ---
    "lr_scheduler": "cosine",       # Cosine annealing with warmup
    "warmup_steps": 2000,           # Linear warmup steps
    "min_lr": 1e-5,                 # Minimum LR at end of cosine decay

    # --- Batch Size ---
    "micro_batch_size": 4,          # Batch size per forward pass
    "grad_accumulation_steps": 8,   # Accumulate gradients over N steps
    # Effective batch size = micro_batch_size × grad_accumulation_steps = 32

    # --- Training Duration ---
    "max_steps": 50000,             # Total training steps (adjust per dataset)
    "max_epochs": None,             # If set, overrides max_steps

    # --- Regularization ---
    "grad_clip": 1.0,               # Max gradient norm (prevents exploding gradients)
    "dropout": 0.1,                 # Dropout rate (also in model config)

    # --- Mixed Precision ---
    "mixed_precision": True,        # Use FP16 on CUDA, disabled on CPU
    "bf16": False,                  # BFloat16 (better than FP16 but T4 doesn't support)

    # --- Checkpointing ---
    "checkpoint_dir": "checkpoints",
    "checkpoint_every": 500,        # Save checkpoint every N steps
    "keep_last_n": 3,               # Keep only last N checkpoints (save disk)
    "save_best": True,              # Also save best model (lowest val loss)

    # --- Evaluation ---
    "eval_every": 500,              # Evaluate on validation set every N steps
    "eval_steps": 50,               # Number of validation batches per eval

    # --- Logging ---
    "log_every": 10,                # Log metrics every N steps
    "use_wandb": True,              # Log to Weights & Biases
    "wandb_project": "SUNDAY_AI",   # W&B project name
    "wandb_run_name": None,         # Auto-generated if None

    # --- Reproducibility ---
    "seed": 42,                     # Random seed for reproducibility

    # --- Device ---
    "device": "auto",               # 'auto', 'cuda', 'cpu'
}


# ============================================
# Fine-tuning Configuration
# ============================================
FINETUNING_CONFIG = {
    # --- Optimizer (lower LR for fine-tuning) ---
    "optimizer": "AdamW",
    "learning_rate": 2e-5,          # Much lower LR than pretraining
    "weight_decay": 0.01,           # Less aggressive weight decay
    "beta1": 0.9,
    "beta2": 0.999,                 # Standard Adam beta2 for fine-tuning
    "eps": 1e-8,

    # --- Schedule ---
    "lr_scheduler": "cosine",
    "warmup_steps": 100,            # Short warmup for fine-tuning
    "min_lr": 1e-6,

    # --- Batch Size ---
    "micro_batch_size": 2,          # Smaller batches for fine-tuning
    "grad_accumulation_steps": 16,  # Effective batch = 32

    # --- Duration ---
    "max_epochs": 3,                # Fine-tuning is few epochs
    "max_steps": None,              # Determined by epochs

    # --- Regularization ---
    "grad_clip": 1.0,
    "dropout": 0.1,

    # --- Mixed Precision ---
    "mixed_precision": True,

    # --- Checkpointing ---
    "checkpoint_dir": "checkpoints/finetuned",
    "checkpoint_every": 200,
    "keep_last_n": 2,
    "save_best": True,

    # --- Evaluation ---
    "eval_every": 200,
    "eval_steps": 25,

    # --- Logging ---
    "log_every": 10,
    "use_wandb": True,
    "wandb_project": "SUNDAY_AI",
    "wandb_run_name": None,

    # --- Reproducibility ---
    "seed": 42,
    "device": "auto",
}
