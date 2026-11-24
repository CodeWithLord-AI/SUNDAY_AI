# 🌅 SUNDAY — A GPT-Style Language Model from Scratch

> Building a decoder-only Transformer language model from randomly initialized weights.

## 🎯 Project Overview

**SUNDAY** is a GPT-style large language model built entirely from scratch as a deep learning research and education project. The model is trained from randomly initialized weights (no pretrained models) using PyTorch on Google Colab T4 GPUs.

### Key Features

- 🧠 **Custom Transformer Architecture** — GPT-2 style decoder-only transformer
- 🔤 **Custom BPE Tokenizer** — 32K vocabulary trained on our data
- 🔥 **Pretrained from Scratch** — No transfer learning, all custom weights
- 💡 **Instruction Fine-tuned** — Supervised fine-tuning on Alpaca dataset
- 💬 **Interactive Chatbot** — Streamlit-based chat interface
- 📊 **Full ML Pipeline** — Training, evaluation, optimization, deployment

### Model Configurations

| Model | Parameters | Layers | Heads | d_model | d_ff | Context |
|-------|-----------|--------|-------|---------|------|---------|
| SUNDAY-Tiny | ~5M | 4 | 4 | 128 | 512 | 128 |
| SUNDAY-50M | ~50M | 8 | 8 | 512 | 2048 | 512 |
| SUNDAY-125M | ~110M | 12 | 12 | 768 | 3072 | 512 |

## 🏗️ Project Structure

```
SUNDAY_AI/
├── configs/          # Model, training, and data configurations
├── src/
│   ├── tokenizer/    # Custom BPE tokenizer
│   ├── model/        # Transformer architecture
│   ├── data/         # Dataset processing & loading
│   ├── training/     # Training loop & optimization
│   ├── evaluation/   # Metrics & evaluation
│   ├── inference/    # Text generation & sampling
│   ├── finetuning/   # Instruction tuning & LoRA
│   └── utils/        # Logging, device, seed, helpers
├── notebooks/        # Colab training notebooks
├── scripts/          # CLI scripts
├── app/              # Streamlit chatbot
├── tests/            # Unit tests
└── checkpoints/      # Saved models (git-ignored)
```

## 🚀 Quick Start

### Local Setup (VS Code)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SUNDAY_AI.git
cd SUNDAY_AI

# Install dependencies (CPU only)
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### Google Colab Training

```python
# 1. Mount Drive & clone repo
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/YOUR_USERNAME/SUNDAY_AI.git
%cd SUNDAY_AI

# 2. Install dependencies
!pip install -r requirements-colab.txt

# 3. Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 4. Start training (see notebooks/)
```

## 🔧 Technology Stack

- **Framework**: PyTorch
- **Tokenizer**: HuggingFace Tokenizers (BPE)
- **Datasets**: HuggingFace Datasets
- **Training**: Google Colab (T4 GPU)
- **Tracking**: Weights & Biases
- **UI**: Streamlit

## 📈 Training Data

| Dataset | Purpose | Size |
|---------|---------|------|
| OpenWebText | Pretraining | ~3 GB subset |
| The Pile | Pretraining | ~2 GB subset |
| Stanford Alpaca | Fine-tuning | 52K examples |

## 📝 Development Phases

- [x] Phase 1: Research & Architecture Design
- [x] Phase 2: Project Setup
- [ ] Phase 3: Tokenizer Development
- [ ] Phase 4: Transformer Architecture
- [ ] Phase 5: Dataset Engineering
- [ ] Phase 6: Pretraining
- [ ] Phase 7: Hyperparameter Tuning
- [ ] Phase 8: Evaluation
- [ ] Phase 9: Fine-tuning
- [ ] Phase 10: Inference & Optimization
- [ ] Phase 11: Chatbot UI
- [ ] Phase 12: Documentation & Deployment

## ⚠️ Limitations

This is a **research prototype**, not a production-ready model. Key limitations:

- Small model size (~50M-125M params) compared to modern LLMs (7B-70B+)
- Limited training data (~5 GB vs hundreds of GB for production models)
- Single GPU training (T4 16GB)
- English-only pretraining
- No RLHF or safety alignment

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- OpenAI GPT-2 paper ("Language Models are Unsupervised Multitask Learners")
- Andrej Karpathy's nanoGPT
- Hugging Face ecosystem
- Google Colab

---

*Built as a learning project to deeply understand LLM architecture, training, and deployment.*
