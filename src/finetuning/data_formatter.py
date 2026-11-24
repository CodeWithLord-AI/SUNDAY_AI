"""
SUNDAY LLM - Alpaca Data Formatter
=====================================

Phase 9 Implementation.

Formats instruction-following data for supervised fine-tuning (SFT).

The Alpaca format:
  {
    "instruction": "What is 2+2?",
    "input": "",           # Optional extra context
    "output": "2+2 = 4."
  }

We convert this to a prompt template that the model learns to follow:

  Template (with input):
    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:
    {output}<eos>

  Template (without input):
    ### Instruction:
    {instruction}

    ### Response:
    {output}<eos>

Training objective:
  During SFT, we only compute loss on the RESPONSE tokens (not the prompt).
  This is done by setting prompt token labels to -1 (ignored in cross-entropy).
  The model learns to follow instructions, not memorize them.
"""

from typing import List, Dict, Optional, Tuple


# Alpaca prompt templates
PROMPT_WITH_INPUT = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

PROMPT_WITHOUT_INPUT = """### Instruction:
{instruction}

### Response:
{output}"""

# Used for inference (no output yet)
PROMPT_INFERENCE_WITH_INPUT = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

PROMPT_INFERENCE_WITHOUT_INPUT = """### Instruction:
{instruction}

### Response:
"""


def format_alpaca_example(
    instruction: str,
    output: str,
    input_text: str = "",
) -> Tuple[str, str]:
    """
    Format a single Alpaca example into prompt + response strings.

    Args:
        instruction: The instruction text.
        output: The expected response.
        input_text: Optional extra input context.

    Returns:
        (prompt, full_text) where:
          - prompt = everything before the response (for masking)
          - full_text = the complete formatted text for tokenization
    """
    if input_text.strip():
        full_text = PROMPT_WITH_INPUT.format(
            instruction=instruction.strip(),
            input=input_text.strip(),
            output=output.strip(),
        )
        prompt = PROMPT_INFERENCE_WITH_INPUT.format(
            instruction=instruction.strip(),
            input=input_text.strip(),
        )
    else:
        full_text = PROMPT_WITHOUT_INPUT.format(
            instruction=instruction.strip(),
            output=output.strip(),
        )
        prompt = PROMPT_INFERENCE_WITHOUT_INPUT.format(
            instruction=instruction.strip(),
        )

    return prompt, full_text


def format_inference_prompt(
    instruction: str,
    input_text: str = "",
) -> str:
    """
    Format an instruction for inference (no output needed).

    Args:
        instruction: The instruction text.
        input_text: Optional extra input context.

    Returns:
        Formatted prompt string.
    """
    if input_text.strip():
        return PROMPT_INFERENCE_WITH_INPUT.format(
            instruction=instruction.strip(),
            input=input_text.strip(),
        )
    else:
        return PROMPT_INFERENCE_WITHOUT_INPUT.format(
            instruction=instruction.strip(),
        )


def prepare_sft_dataset(
    examples: List[Dict],
    tokenizer,
    max_seq_len: int = 512,
    verbose: bool = True,
) -> List[Dict]:
    """
    Tokenize Alpaca examples for supervised fine-tuning.

    For each example:
      1. Format as prompt + response
      2. Tokenize the full text
      3. Create labels where prompt tokens = -1 (ignored)
         and response tokens = actual token IDs (trained on)

    Args:
        examples: List of Alpaca dicts with 'instruction', 'input', 'output'.
        tokenizer: SundayTokenizer instance.
        max_seq_len: Maximum sequence length (longer examples are truncated).
        verbose: Print progress.

    Returns:
        List of dicts with 'input_ids' and 'labels'.
        Labels have -1 for prompt tokens (ignored in loss).

    Example:
        >>> examples = [{"instruction": "Add 1+1", "input": "", "output": "2"}]
        >>> data = prepare_sft_dataset(examples, tokenizer)
        >>> data[0].keys()  # dict_keys(['input_ids', 'labels'])
    """
    if verbose:
        print(f"[SFT] Preparing {len(examples):,} examples...")

    prepared = []
    skipped = 0

    for example in examples:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        # Format
        prompt, full_text = format_alpaca_example(instruction, output, input_text)

        # Tokenize full text
        full_ids = tokenizer.encode(full_text, add_bos=True, add_eos=True)

        # Tokenize prompt only (to find where response starts)
        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)

        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
            skipped += 0  # Note: truncated, not skipped

        if len(full_ids) < 4:  # Too short
            skipped += 1
            continue

        # Create labels: -1 for prompt, actual IDs for response
        labels = [-1] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:len(full_ids)]  # Match length

        prepared.append({
            "input_ids": full_ids,
            "labels": labels,
        })

    if verbose:
        print(f"[SFT] Prepared: {len(prepared):,} examples, {skipped:,} skipped")

    return prepared


class AlpacaDataset:
    """
    PyTorch Dataset for Alpaca-style instruction fine-tuning.

    Returns (input_ids, labels) where labels are -1 for prompt tokens
    so they're ignored in the cross-entropy loss.
    """

    def __init__(self, prepared_data: List[Dict], max_seq_len: int = 512):
        import torch
        self.data = prepared_data
        self.max_seq_len = max_seq_len
        self.torch = torch

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        import torch

        item = self.data[idx]
        input_ids = item["input_ids"][:self.max_seq_len]
        labels = item["labels"][:self.max_seq_len]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
