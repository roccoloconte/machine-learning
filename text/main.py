import os
import pickle
import torch
import random
import numpy as np
import argparse
import logging
import sys
from dataclasses import dataclass, asdict
from text.core.tokenizer import BPETokenizer
from text.core.model import Model
from text.training.trainer import Trainer
from text.utils.logging import setup_logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_head: int = 64
    block_size: int = 64
    max_seq_length: int = 512
    dropout: float = 0.1
    ff_expansion_factor: int = 2
    norm_eps: float = 1e-6

def train(args):
    """Handle training workflow"""
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup logging
    setup_logging()
    logger.info("Starting training execution")

    # Initialize trainer and device
    config = {
        'learning_rate': args.learning_rate,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'warmup_steps': 1000,
        'save_steps': 1000,
        'gradient_accumulation_steps': args.gradient_accumulation,
        'grad_clip': args.grad_clip
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

    # Load training data
    logger.info(f"Loading training data from: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split text into smaller chunks for training
    chunk_size = 512  # Match the model's max sequence length
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    logger.info(f"Split text into {len(text_chunks)} chunks")
    
    # Split into train/val sets (90/10 split)
    split_idx = int(len(text_chunks) * 0.9)
    train_chunks = text_chunks[:split_idx]
    val_chunks = text_chunks[split_idx:]
    logger.info(f"Created {len(train_chunks)} training chunks and {len(val_chunks)} validation chunks")
    
    # Load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = BPETokenizer()
    with open(args.tokenizer_path, 'rb') as f:
        tokenizer_data = pickle.load(f)
        tokenizer.vocab = tokenizer_data["vocab"]
        tokenizer.merge_rules = tokenizer_data["merge_rules"]
        tokenizer.id_to_token = tokenizer_data["id_to_token"]
    logger.info(f"Loaded tokenizer with vocabulary size: {len(tokenizer.vocab)}")

    # Initialize model with custom norm_eps
    model_config = ModelConfig(
        vocab_size=len(tokenizer.vocab),
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_head=64,
        block_size=64,
        max_seq_length=512,
        dropout=0.1,
        ff_expansion_factor=2,
        norm_eps=args.norm_eps  # Use the command-line provided value
    )
    model = Model(model_config)
    logger.info("Model initialized")

    # Add model_config to the trainer config dictionary so it gets saved
    config['model_config'] = asdict(model_config) 

    # Create dataloaders
    from text.data.dataset import TextDataset
    train_loader, val_loader = TextDataset.create_dataloaders(
        train_texts=train_chunks,
        val_texts=val_chunks,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=512,
        num_workers=4  # Use multiple workers for data loading
    )
    logger.info("Dataloaders created")

    # Train the model
    logger.info("Starting model training")
    # Get model and final optimizer state from trainer
    model, optimizer = trainer.train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        tokenizer=tokenizer
    )

    # Save the final trained model as a full checkpoint
    final_checkpoint = {
        'epoch': trainer.config.get('num_epochs', args.epochs), # Get final epoch
        'global_step': -1, # Indicate final save, or get actual last global step if needed
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': trainer.val_losses[-1] if trainer.val_losses else float('inf'), # Last validation loss
        'best_val_loss': min(trainer.val_losses) if trainer.val_losses else float('inf'),
        'config': trainer.config # Save the trainer config (which now includes model_config)
    }
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(final_checkpoint, args.model_path)
    logger.info(f"Final model checkpoint saved to: {args.model_path}")

def inference(args):
    """Handle inference workflow"""
    # Setup logging
    log_file = setup_logging()
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

    # Convert model_config dictionary to ModelConfig object
    try:
        model_config = ModelConfig(**model_config_dict)
    except TypeError as e:
        logger.error(f"Error creating ModelConfig from loaded dictionary: {e}. Loaded keys: {model_config_dict.keys()}")
        raise ValueError(f"Error creating ModelConfig from loaded dictionary: {e}")

    # Initialize model with the loaded config
    model = Model(model_config)
    logger.info("Model initialized with loaded configuration")

    # Load model weights from the checkpoint state_dict
    if 'model_state_dict' not in checkpoint:
        logger.error("Model state_dict not found in the checkpoint.")
        raise ValueError("Model state_dict not found in the checkpoint.")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info("Model weights loaded successfully")

    # Load tokenizer
    logger.info("Initializing tokenizer")
    tokenizer = BPETokenizer()
    
    if not os.path.exists(args.tokenizer_path):
        logger.error(f"Tokenizer file not found at: {args.tokenizer_path}")
        raise FileNotFoundError(f"Tokenizer file not found at: {args.tokenizer_path}")

    with open(args.tokenizer_path, 'rb') as f:
        tokenizer_data = pickle.load(f)
        tokenizer.vocab = tokenizer_data["vocab"]
        tokenizer.merge_rules = tokenizer_data["merge_rules"]
        tokenizer.id_to_token = tokenizer_data["id_to_token"]
    logger.info(f"Tokenizer initialized with vocabulary size: {len(tokenizer.vocab)}")

    # Generate text
    logger.info(f"Generating text with prompt: '{args.prompt}'")
    generated_text = model.generate(tokenizer, args.prompt, 
                                  max_length=args.max_length, 
                                  top_k=args.top_k, 
                                  device=device)
    logger.info(f"Generated text:\n{generated_text}")
    print(f"\nGenerated text:\n{generated_text}")

def train_tokenizer(args):
    """Handle tokenizer training workflow"""
    # Setup logging
    log_file = setup_logging()
    logger.info("Starting tokenizer training")

    # Load training data
    logger.info(f"Loading training data from: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize and train tokenizer
    tokenizer = BPETokenizer()
    vocab, merge_rules = tokenizer.train(text, args.vocab_size)
    logger.info(f"Trained tokenizer with vocabulary size: {len(vocab)}")
    
    # Save tokenizer
    tokenizer_data = {
        "vocab": vocab,
        "merge_rules": merge_rules,
        "id_to_token": {v: k for k, v in vocab.items()}
    }
    
    with open(args.output_path, 'wb') as f:
        pickle.dump(tokenizer_data, f)
    logger.info(f"Saved tokenizer to: {args.output_path}")

def main():
    parser = argparse.ArgumentParser(description='Text Generation Model Training and Inference')
    parser.add_argument('mode', choices=['train', 'inference', 'tokenizer'], 
                      help='Mode to run the script in (train, inference, or tokenizer)')
    
    # Training arguments
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                      help='Learning rate')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                      help='Number of steps to accumulate gradients')
    parser.add_argument('--norm-eps', type=float, default=1e-6,
                      help='Epsilon value for normalization layers')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                      help='Weight decay for optimizer')
    parser.add_argument('--grad-clip', type=float, default=0.5,
                      help='Gradient clipping threshold')
    
    # Inference arguments
    parser.add_argument('--model-path', type=str, default='text/models/model.pth',
                      help='Path to the trained model')
    parser.add_argument('--tokenizer-path', type=str, default='text/models/tokenizer.pkl',
                      help='Path to the trained tokenizer')
    parser.add_argument('--prompt', type=str, default='To be or not to be',
                      help='Text prompt for generation')
    parser.add_argument('--max-length', type=int, default=50,
                      help='Maximum length of generated text')
    parser.add_argument('--top-k', type=int, default=50,
                      help='Top-k sampling parameter')

    # Tokenizer arguments
    parser.add_argument('--data-path', type=str, required='tokenizer' in sys.argv,
                      help='Path to the text file for training the tokenizer')
    parser.add_argument('--vocab-size', type=int, default=50000,
                      help='Desired vocabulary size for the tokenizer')
    parser.add_argument('--output-path', type=str, default='tokenizer.pkl',
                      help='Path to save the trained tokenizer')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    else:
        train_tokenizer(args)

if __name__ == "__main__":
    main()
