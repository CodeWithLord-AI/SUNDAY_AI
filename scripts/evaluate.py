"""
SUNDAY LLM - CLI Evaluation Script
=====================================

Evaluate a SUNDAY checkpoint on validation data.

Usage:
  python scripts/evaluate.py --checkpoint checkpoints/best.pt \\
      --val_data data/processed/val.bin

  # Also compute generation metrics
  python scripts/evaluate.py --checkpoint checkpoints/best.pt \\
      --val_data data/processed/val.bin --gen_metrics
"""

import argparse
import os
import sys
import json

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.model_config import get_model_config
from src.model.sunday_model import SundayModel
from src.tokenizer.tokenizer import SundayTokenizer
from src.evaluation.evaluator import Evaluator
from src.training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a SUNDAY model checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pt checkpoint file.",
    )
    parser.add_argument(
        "--val_data", default=None,
        help="Path to val.bin (pre-tokenised binary).",
    )
    parser.add_argument(
        "--tokenizer_path", default="tokenizer_files",
        help="Path to tokenizer files.",
    )
    parser.add_argument(
        "--model_size", default="50m",
        choices=["tiny", "50m", "125m"],
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
    )
    parser.add_argument(
        "--max_batches", type=int, default=50,
        help="Maximum validation batches to evaluate (use -1 for all).",
    )
    parser.add_argument(
        "--gen_metrics", action="store_true",
        help="Also compute generation quality metrics (distinct-N, repetition).",
    )
    parser.add_argument(
        "--output_json", default=None,
        help="If set, save results as JSON to this path.",
    )
    args = parser.parse_args()

    # ----- Device -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Eval] Device: {device}")

    # ----- Tokenizer -----
    tokenizer = SundayTokenizer()
    if os.path.exists(args.tokenizer_path):
        tokenizer.load(args.tokenizer_path)
        print(f"[Eval] Tokenizer loaded — vocab: {tokenizer.vocab_size}")
    else:
        print(f"[Eval] ⚠ Tokenizer not found at '{args.tokenizer_path}'")

    # ----- Model -----
    model_config = get_model_config(args.model_size)
    model = SundayModel(model_config)

    if not os.path.exists(args.checkpoint):
        print(f"[Eval] ✗ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    state = load_checkpoint(args.checkpoint, model=model, optimizer=None, device=device)
    step  = state.get("step", "?")
    print(f"[Eval] Checkpoint loaded — step {step}")

    model.to(device)
    model.eval()

    # ----- Val DataLoader -----
    val_loader = None
    if args.val_data and os.path.exists(args.val_data):
        from src.data.dataset import SundayDatasetFromFile
        from torch.utils.data import DataLoader
        seq_len = model_config["max_seq_len"]
        val_ds  = SundayDatasetFromFile(args.val_data, seq_len=seq_len)
        max_b   = None if args.max_batches < 0 else args.max_batches
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        print(f"[Eval] Val dataset: {len(val_ds)} chunks → {len(val_loader)} batches")
    else:
        print("[Eval] ⚠ No --val_data provided — skipping perplexity evaluation.")

    # ----- Evaluator -----
    evaluator = Evaluator(model=model, tokenizer=tokenizer, device=device)

    results = {}

    # Perplexity
    if val_loader is not None:
        max_b = None if args.max_batches < 0 else args.max_batches
        ppl_results = evaluator.evaluate(val_loader, max_batches=max_b)
        results.update(ppl_results)

    # Generation metrics
    if args.gen_metrics:
        test_prompts = [
            "The quick brown fox",
            "In the beginning",
            "Science has shown that",
            "Once upon a time",
        ]
        gen_results = evaluator.evaluate_generation(test_prompts, max_new_tokens=100)
        results.update(gen_results)

    # ----- Report -----
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key:<30} {val:.4f}")
        else:
            print(f"  {key:<30} {val}")
    print("=" * 50)

    # ----- Save JSON -----
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Eval] Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
