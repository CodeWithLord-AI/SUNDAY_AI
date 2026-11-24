"""
SUNDAY LLM - Model Configuration
==================================

This file defines the hyperparameters for different model sizes.

Architecture: GPT-2 style decoder-only Transformer
- Token + Positional Embeddings
- Pre-Norm (LayerNorm before attention/FFN)
- Multi-Head Causal Self-Attention
- GELU activation in Feed-Forward Network
- Weight tying (LM head shares weights with token embeddings)

Three configurations:
1. SUNDAY_TINY  (~5M params)  - For debugging & pipeline validation
2. SUNDAY_50M   (~50M params) - For initial training & validation
3. SUNDAY_125M  (~125M params) - Target production model

Parameter Count Formula (with weight tying):
  params = vocab_size * d_model              (token embeddings)
         + max_seq_len * d_model             (position embeddings)
         + n_layers * (
             4 * d_model * d_model           (Q, K, V, O projections)
           + 2 * d_model * d_ff              (FFN up + down projections)
           + 4 * d_model                     (2 LayerNorms × 2 params each)
           )
         + 2 * d_model                       (final LayerNorm)
"""


# ============================================
# TINY Model - For Debugging Only
# ============================================
SUNDAY_TINY = {
    "model_name": "SUNDAY-Tiny",
    "vocab_size": 32768,        # 32K BPE vocabulary
    "max_seq_len": 128,         # Short context for fast debugging
    "d_model": 128,             # Embedding dimension
    "n_heads": 4,               # Number of attention heads
    "n_layers": 4,              # Number of transformer blocks
    "d_ff": 512,                # Feed-forward hidden dimension (4x d_model)
    "dropout": 0.1,             # Dropout rate
    "activation": "gelu",       # Activation function in FFN
    "norm_type": "pre_norm",    # LayerNorm placement (pre_norm = modern standard)
    "bias": False,              # No bias in linear layers (modern practice)
    "weight_tying": True,       # Share token embedding with LM head
    "norm_eps": 1e-5,           # LayerNorm epsilon
}
# Approximate params: ~5.1M


# ============================================
# 50M Model - Initial Training & Validation
# ============================================
SUNDAY_50M = {
    "model_name": "SUNDAY-50M",
    "vocab_size": 32768,        # 32K BPE vocabulary
    "max_seq_len": 512,         # Context window (512 tokens ≈ 400 words)
    "d_model": 512,             # Embedding dimension
    "n_heads": 8,               # Number of attention heads (d_k = 512/8 = 64)
    "n_layers": 8,              # Number of transformer blocks
    "d_ff": 2048,               # Feed-forward hidden dim (4x d_model)
    "dropout": 0.1,             # Dropout rate
    "activation": "gelu",       # Activation function in FFN
    "norm_type": "pre_norm",    # LayerNorm placement
    "bias": False,              # No bias in linear layers
    "weight_tying": True,       # Share token embedding with LM head
    "norm_eps": 1e-5,           # LayerNorm epsilon
}
# Approximate params: ~49.5M
# VRAM estimate (FP16 + AdamW): ~6-8 GB
# T4 compatible: YES (single session, 4-8 hours)


# ============================================
# 125M Model - Target Production Model
# ============================================
SUNDAY_125M = {
    "model_name": "SUNDAY-125M",
    "vocab_size": 32768,        # 32K BPE vocabulary
    "max_seq_len": 512,         # Context window (512 tokens ≈ 400 words)
    "d_model": 768,             # Embedding dimension
    "n_heads": 12,              # Number of attention heads (d_k = 768/12 = 64)
    "n_layers": 12,             # Number of transformer blocks
    "d_ff": 3072,               # Feed-forward hidden dim (4x d_model)
    "dropout": 0.1,             # Dropout rate
    "activation": "gelu",       # Activation function in FFN
    "norm_type": "pre_norm",    # LayerNorm placement
    "bias": False,              # No bias in linear layers
    "weight_tying": True,       # Share token embedding with LM head
    "norm_eps": 1e-5,           # LayerNorm epsilon
}
# Approximate params: ~110.6M (with weight tying)
# VRAM estimate (FP16 + AdamW + grad accum): ~12-15 GB
# T4 compatible: YES (requires mixed precision + gradient accumulation)


# ============================================
# Helper: Get config by name
# ============================================
MODEL_CONFIGS = {
    "tiny": SUNDAY_TINY,
    "50m": SUNDAY_50M,
    "125m": SUNDAY_125M,
}


def get_model_config(size: str = "50m") -> dict:
    """
    Get model configuration by size name.

    Args:
        size: One of 'tiny', '50m', '125m'

    Returns:
        Dictionary of model hyperparameters.

    Example:
        >>> config = get_model_config("50m")
        >>> print(config["d_model"])  # 512
    """
    size = size.lower()
    if size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size '{size}'. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[size].copy()
