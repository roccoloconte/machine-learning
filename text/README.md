# Text Generation Model

A flexible text generation model built with PyTorch that can be trained on any text dataset.

## Installation

Install requirements:
```bash
pip install torch numpy requests tqdm wandb
```

## Dataset Preparation

The model can work with any text dataset. We provide utilities to download some common datasets:

```bash
python3 -m text.data.download tinyshakespeare
```

## Training Pipeline

The training pipeline consists of three main steps:

### 1. Train the Tokenizer

First, train a BPE tokenizer on your dataset:

```bash
python3 -m text.main tokenizer \
    --data-path text/dataset/tinyshakespeare.txt \
    --output-path text/models/tokenizer.pkl \
    --vocab-size 1000
```

Parameters:
- `--data-path`: Path to your training text file
- `--vocab-size`: Size of the vocabulary (default: 50000)
- `--output-path`: Where to save the trained tokenizer

### 2. Train the Model

Train the model using your dataset and the trained tokenizer:

```bash
python3 -m text.main train \
    --data-path text/dataset/tinyshakespeare.txt \
    --tokenizer-path text/models/tokenizer.pkl \
    --model-path text/models/model.pth \
    --epochs 10 \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --grad-clip 1.0 \
    --seed 42
```

Parameters:
- `--data-path`: Path to your training text file
- `--tokenizer-path`: Path to the trained tokenizer
- `--model-path`: Where to save the trained model
- `--epochs`: Number of training epochs (default: 1)
- `--batch-size`: Batch size for training (default: 8)
- `--learning-rate`: Learning rate (default: 5e-4)
- `--seed`: Random seed for reproducibility (default: 42)

### 3. Generate Text

Generate text using your trained model:

```bash
python3 -m text.main inference \
    --model-path text/models/model.pth \
    --tokenizer-path text/models/tokenizer.pkl \
    --prompt "Once upon a time" \
    --max-length 100 \
    --seed 42 \
    --top-k 50
```

Parameters:
- `--model-path`: Path to the trained model
- `--tokenizer-path`: Path to the trained tokenizer
- `--prompt`: Text prompt to start generation
- `--max-length`: Maximum length of generated text (default: 50)
- `--top-k`: Top-k sampling parameter (default: 50)
