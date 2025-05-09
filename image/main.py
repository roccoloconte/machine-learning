import os
import torch
import random
import numpy as np
import argparse
import logging
import sys
from dataclasses import dataclass, asdict
from image.core.model import VisionTransformer, ViTConfig
from image.training.trainer import Trainer
from image.dataset.dataset import MNISTDataset
from image.utils.logging import setup_logging
from image.utils.visualization import visualize_predictions, visualize_attention

logger = logging.getLogger(__name__)

def train(args):
    """Handle training workflow"""
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup logging
    log_file = setup_logging(log_dir=args.log_dir)
    logger.info("Starting training execution")

    # Initialize trainer and device
    config = {
        'learning_rate': args.learning_rate,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'warmup_steps': 100,
        'save_steps': 100,
        'gradient_accumulation_steps': args.gradient_accumulation,
        'grad_clip': args.grad_clip,
        'log_dir': args.log_dir,
        'checkpoint_dir': args.checkpoint_dir
    }
    trainer = Trainer(config)
    
    # Set device (MPS, CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    trainer.device = device

    # Create dataloaders
    logger.info("Creating dataloaders")
    train_loader, val_loader, test_loader = MNISTDataset.create_dataloaders(
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=args.num_workers
    )
    logger.info("Dataloaders created")

    # Initialize Vision Transformer model
    logger.info("Initializing Vision Transformer model")
    model_config = ViTConfig(
        image_size=28,  # MNIST images are 28x28
        patch_size=args.patch_size,
        num_channels=1,  # MNIST is grayscale
        num_classes=10,  # 10 digit classes (0-9)
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.hidden_size * 4,  # Common practice to use 4x hidden size
        dropout=args.dropout,
        attention_dropout=args.dropout,
        norm_eps=1e-6
    )
    model = VisionTransformer(model_config)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Add model_config to the trainer config dictionary so it gets saved
    config['model_config'] = asdict(model_config)

    # Train the model
    logger.info("Starting model training")
    # Get model and final optimizer state from trainer
    model, optimizer = trainer.train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader
    )

    # Evaluate on test set
    logger.info("Evaluating model on test set")
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = trainer.evaluate(model, test_loader, criterion)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # Visualize predictions
    logger.info("Visualizing predictions")
    visualize_predictions(model, test_loader, device, num_images=10, save_dir=os.path.join(args.log_dir, 'visualizations'))

    # Visualize attention maps
    logger.info("Visualizing attention maps")
    visualize_attention(
        model, 
        test_loader, 
        device, 
        layer_idx=model_config.num_layers-1,  # Visualize last layer
        head_idx=0,
        num_images=5,
        save_dir=os.path.join(args.log_dir, 'visualizations/attention')
    )

    logger.info("Training and evaluation complete")

def inference(args):
    """Handle inference workflow"""
    # Setup logging
    log_file = setup_logging(log_dir=args.log_dir)
    logger.info("Starting inference execution")

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Load checkpoint to get config and model state
    if not os.path.exists(args.model_path):
        logger.error(f"Model checkpoint file not found at: {args.model_path}")
        raise FileNotFoundError(f"Model checkpoint file not found at: {args.model_path}")

    logger.info(f"Loading checkpoint from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract config dictionary from checkpoint
    if 'config' not in checkpoint:
        logger.error("Configuration not found in the checkpoint.")
        raise ValueError("Configuration not found in the checkpoint.")
    config_dict = checkpoint['config']

    # Extract the nested model_config dictionary
    if 'model_config' not in config_dict:
        logger.error("Model configuration (model_config) not found within the saved config.")
        raise ValueError("Model configuration (model_config) not found within the saved config.")
    model_config_dict = config_dict['model_config']

    # Convert model_config dictionary to ViTConfig object
    try:
        model_config = ViTConfig(**model_config_dict)
    except TypeError as e:
        logger.error(f"Error creating ViTConfig from loaded dictionary: {e}")
        raise ValueError(f"Error creating ViTConfig from loaded dictionary: {e}")

    # Initialize model with the loaded config
    model = VisionTransformer(model_config)
    logger.info("Model initialized with loaded configuration")

    # Load model weights from the checkpoint state_dict
    if 'model_state_dict' not in checkpoint:
        logger.error("Model state_dict not found in the checkpoint.")
        raise ValueError("Model state_dict not found in the checkpoint.")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info("Model weights loaded successfully")

    # Create test dataloader
    _, _, test_loader = MNISTDataset.create_dataloaders(
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=1
    )
    
    # Evaluate on test set
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = Trainer({}).evaluate(model, test_loader, criterion)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Visualize predictions
    logger.info("Visualizing predictions")
    visualize_predictions(model, test_loader, device, num_images=10, save_dir=os.path.join(args.log_dir, 'visualizations'))
    
    # Visualize attention maps
    logger.info("Visualizing attention maps")
    visualize_attention(
        model, 
        test_loader, 
        device, 
        layer_idx=model_config.num_layers-1,  # Visualize last layer
        head_idx=0,
        num_images=5,
        save_dir=os.path.join(args.log_dir, 'visualizations/attention')
    )
    
    logger.info("Inference completed")

def main():
    """Main function to parse arguments and dispatch to appropriate function"""
    parser = argparse.ArgumentParser(description="Train or run inference with Vision Transformer on MNIST")
    
    # Common arguments
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], default='train',
                      help='Mode to run: train or inference')
    parser.add_argument('--log_dir', type=str, default='./image/logs',
                      help='Directory to save logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./image/checkpoints',
                      help='Directory to save model checkpoints')
    parser.add_argument('--model_path', type=str, default='./image/checkpoints/best_model.pt',
                      help='Path to model checkpoint for inference')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training and inference')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    # Training specific arguments
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                      help='Gradient accumulation steps')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                      help='Gradient clipping')
    
    # Model specific arguments
    parser.add_argument('--patch_size', type=int, default=4,
                      help='Patch size for Vision Transformer')
    parser.add_argument('--hidden_size', type=int, default=256,
                      help='Hidden size for Vision Transformer')
    parser.add_argument('--num_layers', type=int, default=6,
                      help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                      help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Dispatch to appropriate function based on mode
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)

if __name__ == "__main__":
    main() 