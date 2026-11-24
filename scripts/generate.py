"""
SUNDAY LLM - CLI Text Generation Script
==========================================

Generate text from a trained SUNDAY checkpoint.

Usage:
  python scripts/generate.py --checkpoint checkpoints/best.pt \\
      --prompt "Once upon a time" --max_tokens 200

  # Beam-less sampling with temperature/top-p
  python scripts/generate.py --checkpoint checkpoints/step_5000.pt \\
      --prompt "The capital of France is" --temperature 0.8 --top_p 0.9

  # Greedy (temperature = 0)
  python scripts/generate.py --checkpoint checkpoints/best.pt \\
      --prompt "Hello world" --temperature 0.0
"""

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.model_config import get_model_config
from src.model.sunday_model import SundayModel
from src.tokenizer.tokenizer import SundayTokenizer
from src.inference.generator import Generator
from src.training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with a trained SUNDAY model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pt checkpoint file.",
    )
    parser.add_argument(
        "--tokenizer_path", default="tokenizer_files",
        help="Path to tokenizer files.",
    )
    parser.add_argument(
        "--model_size", default="50m",
        choices=["tiny", "50m", "125m"],
        help="Model size (must match checkpoint).",
    )
    parser.add_argument(
        "--prompt", default="Once upon a time",
        help="Starting prompt for generation.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=200,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature. 0 = greedy.",
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Top-K sampling. 0 = disabled.",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Nucleus (top-p) sampling. 1.0 = disabled.",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.1,
        help="Repetition penalty > 1.0 discourages repeating tokens.",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream tokens to stdout as they are generated.",
    )
    args = parser.parse_args()

    # ----- Device -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Generate] Device: {device}")

    # ----- Tokenizer -----
    tokenizer = SundayTokenizer()
    if os.path.exists(args.tokenizer_path):
        tokenizer.load(args.tokenizer_path)
        print(f"[Generate] Tokenizer loaded — vocab: {tokenizer.vocab_size}")
    else:
        print(f"[Generate] ⚠ Tokenizer not found at '{args.tokenizer_path}'")

    # ----- Model -----
    model_config = get_model_config(args.model_size)
    model = SundayModel(model_config)

    if not os.path.exists(args.checkpoint):
        print(f"[Generate] ✗ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    state = load_checkpoint(args.checkpoint, model=model, optimizer=None, device=device)
    step = state.get("step", "?")
    print(f"[Generate] Checkpoint loaded — step {step}")

    model.to(device)
    model.eval()

    # ----- Generator -----
    generator = Generator(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    sampling_kwargs = dict(
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if args.top_p < 1.0 else None,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_tokens,
    )

    print(f"\n[Generate] Prompt: {args.prompt!r}")
    print("[Generate] Output:")
    print("-" * 60)

    if args.stream:
        # Stream tokens one by one
        generated = args.prompt
        for token in generator.stream(args.prompt, **sampling_kwargs):
            print(token, end="", flush=True)
            generated += token
        print()  # newline at end
    else:
        output = generator.generate(args.prompt, **sampling_kwargs)
        print(output)

    print("-" * 60)


if __name__ == "__main__":
    main()
