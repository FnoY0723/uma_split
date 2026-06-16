# UMA-SPLIT: Unimodal Aggregation Modules

This repository contains the standalone PyTorch implementations of the **UMA** and **Split** modules from the official paper: **"UMA-SPLIT: UNIMODAL AGGREGATION FOR BOTH ENGLISH AND MANDARIN NON-AUTOREGRESSIVE SPEECH RECOGNITION"**.

## Installation

No dedicated installation is needed. They rely purely on standard PyTorch. Just clone the repository and import the modules from `uma.py` and `split.py`.

```bash
git clone https://github.com/your-username/uma-split-modules.git
cd uma-split-modules
```

## Requirements
- Python >= 3.8
- PyTorch >= 1.9.0

## Usage

```python
import torch
from uma import UMA
from split import Split

batch_size, length, dim = 4, 100, 256
dummy_lengths = torch.tensor([100, 80, 50, 40])
dummy_input = torch.randn(batch_size, length, dim)

# 1. UMA Module
# Aggregates frame-level acoustic features into phoneme-level representations
uma_layer = UMA(input_size=dim, output_size=dim)
uma_out, uma_lens = uma_layer(dummy_input, dummy_lengths)
print(f"UMA output shape: {uma_out.shape}, New lengths: {uma_lens}")

# 2. Split Module
# Extracts split features and doubles the sequence length for decoding 
split_layer = Split(output_size=dim, dropout_rate=0.1)
split_out, split_lens = split_layer(uma_out, uma_lens)
print(f"Split output shape: {split_out.shape}, New lengths: {split_lens}")
```

## Module Descriptions

### UMA Module (`uma.py`)
Predicts frame similarities and continuously pools representations around detected unimodal distribution **valleys**. Using continuous sequence segment boundaries, it aggregates inputs over variable-length unigram zones using weighted sums before layer normalization.

### Split Module (`split.py`)
Acts as length-grower and interweaver logic. Projects encoded embeddings via expanded intermediary limits, then strictly concatenates outputs back interleaving with source vectors to logically double output emission capability length bounds efficiently.

## License

MIT License.
