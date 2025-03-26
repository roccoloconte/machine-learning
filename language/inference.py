import os
import torch
import logging
from language.training import Model, BPETokenizer

# Get the logger without setting up logging twice
logger = logging.getLogger(__name__)

# Model configuration - must match what was used during training
config = {
    'vocab_size': 1001,  # 1000 + unknown token
    'hidden_size': 256,
    'num_layers': 4,
    'num_heads': 4,
    'latent_dim': 64,
    'block_size': 64,
    'max_position_embeddings': 512,
    'dropout': 0.1,
    'ff_expansion_factor': 2,
    'rms_norm_eps': 1e-6,
    'max_gen_length': 50,
    'top_k': 50
}

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else 
                     "cpu")
logger.info(f"Using device: {device}")

# Initialize model
model = Model(config)
logger.info("Model initialized")

# Update path to look for model in the current directory
model_path = "model.pth"  # Since we're now in the language folder
if not os.path.exists(model_path):
    logger.error(f"Model file not found at: {model_path}")
    logger.info("Please check the following locations for your model file:")
    logger.info("  - Current directory: " + os.getcwd())
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load model weights with weights_only=True for security
logger.info(f"Loading model from: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()
logger.info("Model loaded successfully")

# Load and initialize tokenizer
logger.info("Initializing tokenizer")
tokenizer = BPETokenizer()

# Update path to look for Shakespeare text in the current directory
shakespeare_path = "tinyshakespeare.txt"
if not os.path.exists(shakespeare_path):
    logger.error(f"Shakespeare text file not found at: {shakespeare_path}")
    raise FileNotFoundError(f"Shakespeare text file not found at: {shakespeare_path}")

with open(shakespeare_path, "r") as f:
    text = f.read()
train_text = text[:int(0.8 * len(text))]  # Use the same training split
vocab, _ = tokenizer.train(train_text, 1000)
logger.info(f"Tokenizer initialized with vocabulary size: {len(vocab)}")

# Generate text
prompt = "To be or not to be"
logger.info(f"Generating text with prompt: '{prompt}'")
generated_text = model.generate(tokenizer, prompt, 
                               max_length=config['max_gen_length'], 
                               top_k=config['top_k'], 
                               device=device)
logger.info(f"Generated text:\n{generated_text}")
print(f"\nGenerated text:\n{generated_text}")