# Vision Transformer for MNIST

This folder contains code for training and evaluating a Vision Transformer (ViT) model on the MNIST dataset.

## Architecture

The implementation uses a Vision Transformer architecture:
- Patch embedding to convert images into sequences of patches
- Multi-head self-attention mechanism
- MLP blocks for feature processing
- Transformer encoders for learning representations
- Classification head using the CLS token

## Directory Structure

```
image/
├── core/              # Core model implementation
├── dataset/           # Dataset loading and preprocessing
├── training/          # Training utilities
├── utils/             # General utilities
├── models/            # Saved model implementations
├── checkpoints/       # Model checkpoints
├── logs/              # Training logs
├── visualizations/    # Visualization outputs
└── main.py            # Main training script
```

## Getting Started

### Prerequisites

The code requires the following Python packages:
- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm
- tensorboard

### Training

To train a Vision Transformer model on MNIST:

```bash
python -m image.main --mode train --epochs 10 --batch_size 128
```

### Inference

To run inference with a trained model:

```bash
python -m image.main --mode inference --model_path ./image/checkpoints/best_model.pt
```

## Configuration Options

You can customize the training and model by passing various arguments:

### Model Configuration
- `--patch_size`: Size of image patches (default: 4)
- `--hidden_size`: Transformer hidden dimension (default: 256)
- `--num_layers`: Number of transformer layers (default: 6)
- `--num_heads`: Number of attention heads (default: 8)
- `--dropout`: Dropout rate (default: 0.1)

### Training Configuration
- `--epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--weight_decay`: Weight decay for optimizer (default: 0.01)
- `--batch_size`: Batch size (default: 128)
- `--seed`: Random seed (default: 42)
- `--grad_clip`: Gradient clipping value (default: 1.0)

## Results

After training, you'll find:
- Model checkpoints in `./image/checkpoints/`
- Training logs in `./image/logs/`
- Visualizations of model predictions in `./image/logs/visualizations/`
- Attention map visualizations in `./image/logs/visualizations/attention/`

## Attention Visualization

The code includes functionality to visualize attention maps from the Vision Transformer, showing how different parts of the model attend to different regions of the image.

## Performance

On MNIST, the Vision Transformer can achieve >98% accuracy with the default settings after 10 epochs of training. 