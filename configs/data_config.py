"""
SUNDAY LLM - Data Configuration
=================================

Dataset paths, processing parameters, and split ratios.

Datasets used:
1. OpenWebText (pretraining) - Open-source GPT-2 training data replica
   - Source: https://huggingface.co/datasets/openwebtext
   - License: MIT-like (derived from Reddit submissions)
   - Size: ~38 GB full, we use ~3 GB subset

2. The Pile (pretraining supplement) - Diverse text corpus
   - Source: https://huggingface.co/datasets/EleutherAI/pile
   - License: MIT
   - Size: ~800 GB full, we use ~2 GB subset

3. Stanford Alpaca (fine-tuning) - Instruction-following dataset
   - Source: https://huggingface.co/datasets/tatsu-lab/alpaca
   - License: Apache 2.0
   - Size: ~52K instruction-response pairs
"""

import os


# ============================================
# Pretraining Data Configuration
# ============================================
PRETRAINING_DATA = {
    # --- Primary Dataset: OpenWebText ---
    "openwebtext": {
        "hf_name": "openwebtext",           # HuggingFace dataset name
        "hf_config": None,                   # Dataset configuration
        "subset_size": 500_000,              # Number of documents to use
        "text_column": "text",               # Column containing text data
        "license": "MIT-like",
    },

    # --- Supplementary Dataset: The Pile ---
    "the_pile": {
        "hf_name": "monology/pile-uncopyrighted",  # Uncopyrighted subset
        "hf_config": None,
        "subset_size": 200_000,
        "text_column": "text",
        "license": "MIT",
    },
}


# ============================================
# Fine-tuning Data Configuration
# ============================================
FINETUNING_DATA = {
    "alpaca": {
        "hf_name": "tatsu-lab/alpaca",
        "hf_config": None,
        "text_column": None,                 # Uses instruction/input/output format
        "license": "Apache 2.0",

        # Instruction format template
        "prompt_template": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        ),
        "prompt_template_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Response:\n{output}"
        ),
    },
}


# ============================================
# Data Processing Configuration
# ============================================
DATA_CONFIG = {
    # --- Tokenizer ---
    "tokenizer_type": "bpe",                 # Byte-Pair Encoding
    "vocab_size": 32768,                     # 32K vocabulary
    "min_frequency": 2,                      # Minimum token frequency
    "special_tokens": [
        "<pad>",                             # Padding token
        "<unk>",                             # Unknown token
        "<bos>",                             # Beginning of sequence
        "<eos>",                             # End of sequence
        "<sep>",                             # Separator (for instruction tuning)
    ],

    # --- Tokenizer Files ---
    "tokenizer_dir": "tokenizer_files",
    "tokenizer_file": "sunday_tokenizer.json",

    # --- Data Processing ---
    "max_seq_len": 512,                      # Must match model config
    "train_split": 0.95,                     # 95% train, 5% validation
    "val_split": 0.05,
    "shuffle_seed": 42,
    "num_workers": 2,                        # DataLoader workers (local)
    "num_workers_colab": 2,                  # DataLoader workers (Colab)

    # --- Data Directories ---
    "raw_data_dir": "data/raw",              # Downloaded raw data
    "processed_data_dir": "data/processed",  # Tokenized/processed data
    "cache_dir": "data/cache",               # HuggingFace cache

    # --- Tokenizer Training ---
    "tokenizer_training_size": 500_000,      # Number of texts for tokenizer training
    "tokenizer_training_file": "data/tokenizer_train.txt",
}
