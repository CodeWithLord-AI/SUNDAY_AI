"""
SUNDAY LLM - Device Detection Utilities
=========================================

Why do we need device management?
  Our code must work on:
  - Local laptop (CPU only — AMD Radeon R4 has no CUDA)
  - Google Colab (NVIDIA T4 GPU with CUDA)

  Instead of hardcoding 'cuda' or 'cpu', we detect the best available
  device automatically and move tensors/models to it.

What this module does:
  - Detects if CUDA GPU is available
  - Returns the appropriate torch.device
  - Prints GPU info for verification in Colab
"""

import torch


def get_device(preferred: str = "auto") -> torch.device:
    """
    Get the best available compute device.

    Args:
        preferred: Device preference.
            - 'auto': Use CUDA if available, else CPU
            - 'cuda': Force CUDA (raises error if unavailable)
            - 'cpu': Force CPU

    Returns:
        torch.device object

    Example:
        >>> device = get_device()
        >>> print(device)  # device(type='cuda') or device(type='cpu')
        >>> model = model.to(device)
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. "
                "Are you running on Colab with GPU enabled?"
            )
        device = torch.device("cuda")
    elif preferred == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device: '{preferred}'. Use 'auto', 'cuda', or 'cpu'.")

    return device


def print_device_info() -> None:
    """
    Print detailed device information.
    Useful for verifying GPU setup in Colab.

    Example output (on Colab T4):
        ===== SUNDAY Device Info =====
        Device: cuda
        GPU: Tesla T4
        VRAM Total: 15.1 GB
        VRAM Used: 0.5 GB
        VRAM Free: 14.6 GB
        CUDA Version: 12.1
        PyTorch Version: 2.1.0+cu121
        ==============================
    """
    print("=" * 35)
    print("  SUNDAY Device Info")
    print("=" * 35)

    device = get_device()
    print(f"  Device: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        reserved_mem = torch.cuda.memory_reserved(0) / 1e9
        allocated_mem = torch.cuda.memory_allocated(0) / 1e9
        free_mem = total_mem - reserved_mem

        print(f"  GPU: {gpu_name}")
        print(f"  VRAM Total: {total_mem:.1f} GB")
        print(f"  VRAM Allocated: {allocated_mem:.1f} GB")
        print(f"  VRAM Reserved: {reserved_mem:.1f} GB")
        print(f"  VRAM Free (approx): {free_mem:.1f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("  GPU: None (CPU only)")
        print("  Note: Training requires a CUDA GPU (use Colab)")

    print(f"  PyTorch Version: {torch.__version__}")
    print("=" * 35)
