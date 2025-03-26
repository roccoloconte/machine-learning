import os
import time
import math
import requests
import logging
import logging.handlers
from tqdm import tqdm
from typing import Optional, Tuple
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Set up logging configuration to track training progress and debugging information
def setup_logging():
    """
    Configure logging to write to both a rotating file and the console.
    Returns the path to the log file.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"language_model_{time.strftime('%Y%m%d')}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB per file, keep 5 backups
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.addHandler(console_handler)
    
    return log_file

log_file = setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"Logging to {log_file}")

class BPETokenizer:
    """
    Implementation of Byte Pair Encoding tokenizer for text compression and tokenization.
    BPE iteratively merges the most frequent pairs of adjacent tokens.
    """
    def __init__(self):
        self.vocab = {}  # Maps tokens to token IDs
        self.merge_rules = []  # List of token pairs to merge in order
        self.id_to_token = {}  # Maps token IDs back to tokens (for detokenization)
        logger.info("Initialized BPETokenizer")

    def train(self, text, desired_vocab_size):
        """
        Train the BPE tokenizer on the input text to reach the desired vocabulary size.
        
        Args:
            text: The input text to train on
            desired_vocab_size: Target size for the vocabulary
            
        Returns:
            vocab: The vocabulary mapping (token -> id)
            merge_rules: List of merge rules learned during training
        """
        logger.info(f"Starting tokenizer training with desired vocab size: {desired_vocab_size}")
        # Start with character-level vocabulary
        chars = sorted(list(set(text)))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        tokens = list(text)
        num_merges = desired_vocab_size - len(self.vocab)
        logger.info(f"Initial vocabulary size: {len(self.vocab)}, Number of merges required: {num_merges}")
        
        # Iteratively find and merge the most frequent adjacent token pairs
        for i in range(num_merges):
            pair_freq = self.get_pair_freq(tokens)
            if not pair_freq:
                logger.warning("No more pairs to merge")
                break
            most_freq_pair = max(pair_freq, key=pair_freq.get)
            self.merge_rules.append(most_freq_pair)
            tokens = self.merge_pair(tokens, most_freq_pair)
            new_token = ''.join(most_freq_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
            if i % 100 == 0:
                logger.info(f"Merge {i}: Added token '{new_token}', Current vocab size: {len(self.vocab)}")
                
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        logger.info(f"Tokenizer training completed. Final vocab size: {len(self.vocab)}")
        return self.vocab, self.merge_rules

    def tokenize(self, text):
        """
        Convert text into tokens using the learned BPE merge rules.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        logger.debug(f"Tokenizing text of length: {len(text)}")
        tokens = list(text)  # Start with character-level tokens
        # Apply merge rules sequentially
        for pair in self.merge_rules:
            tokens = self.merge_pair(tokens, pair)
        logger.debug(f"Tokenization complete, produced {len(tokens)} tokens")
        return tokens

    def detokenize(self, token_ids):
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Reconstructed text
        """
        logger.debug(f"Detokenizing {len(token_ids)} token IDs")
        tokens = [self.id_to_token[id] for id in token_ids]
        result = ''.join(tokens)
        logger.debug(f"Detokenization complete, output length: {len(result)}")
        return result

    def get_pair_freq(self, tokens):
        """
        Count frequencies of adjacent token pairs in the list of tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary mapping token pairs to their frequencies
        """
        pair_freq = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freq[pair] += 1
        logger.debug(f"Found {len(pair_freq)} unique pairs")
        return pair_freq

    def merge_pair(self, tokens, pair):
        """
        Apply a single merge operation on the token list.
        
        Args:
            tokens: List of tokens
            pair: The pair of tokens to merge
            
        Returns:
            New list of tokens with specified pair merged
        """
        logger.debug(f"Merging pair: {pair}")
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        logger.debug(f"Merge complete, reduced from {len(tokens)} to {len(new_tokens)} tokens")
        return new_tokens

class RotaryPositionalEmbedding(nn.Module):
    """
    Implements rotary positional embeddings (RoPE) for transformer models.
    RoPE performs position-dependent rotation of embedding vectors to encode
    positional information directly in the attention mechanism.
    """
    def __init__(self, dim: int, max_position: int):
        """
        Initialize the rotary positional embedding.
        
        Args:
            dim: Dimension of the embedding
            max_position: Maximum sequence length supported
        """
        super().__init__()
        # Create inverse frequency for wavelengths calculation
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position = max_position

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos and sin components for rotary embeddings.
        
        Args:
            positions: Position indices tensor
            
        Returns:
            Tuple of (cos, sin) tensors for applying rotary embeddings
        """
        freqs = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head attention mechanism with latent space projection for more efficient computation.
    Uses rotary positional embeddings for position-aware attention.
    """
    def __init__(self, config):
        """
        Initialize the multi-head latent attention module.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super().__init__()
        self.num_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = self.hidden_size // self.num_heads
        self.latent_dim = config['latent_dim']
        # Project queries and keys to a lower-dimensional latent space
        self.q_latent_proj = nn.Linear(self.head_dim, self.latent_dim)
        self.k_latent_proj = nn.Linear(self.head_dim, self.latent_dim)
        self.v_proj = nn.Linear(self.head_dim, self.head_dim)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config['dropout'])
        self.rotary_emb = RotaryPositionalEmbedding(self.latent_dim, config['max_position_embeddings'])
        self.block_size = config.get('block_size', 64)  # Block size for chunked attention computation

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-head latent attention.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional mask tensor [batch, seq_len]
            
        Returns:
            Output tensor after attention [batch, seq_len, hidden_size]
        """
        batch_size, seq_length = hidden_states.shape[:2]
        # Reshape for multi-head processing
        hidden_states = hidden_states.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Project to query, key, value representations
        q = self.q_latent_proj(hidden_states)  # [batch, seq, heads, latent_dim]
        k = self.k_latent_proj(hidden_states)  # [batch, seq, heads, latent_dim]
        v = self.v_proj(hidden_states)         # [batch, seq, heads, head_dim]

        # Apply rotary positional embeddings
        positions = torch.arange(seq_length, device=hidden_states.device).float()
        cos, sin = self.rotary_emb(positions)
        cos = cos.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.num_heads, -1)
        sin = sin.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.num_heads, -1)
        
        # Apply rotation to q and k
        q_rot = q.clone()
        k_rot = k.clone()
        q_rot[..., ::2] = q[..., ::2] * cos[..., ::2] - q[..., 1::2] * sin[..., ::2]
        q_rot[..., 1::2] = q[..., ::2] * sin[..., ::2] + q[..., 1::2] * cos[..., ::2]
        k_rot[..., ::2] = k[..., ::2] * cos[..., ::2] - k[..., 1::2] * sin[..., ::2]
        k_rot[..., 1::2] = k[..., ::2] * sin[..., ::2] + k[..., 1::2] * cos[..., ::2]

        # Create causal (lower triangular) attention mask
        causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=hidden_states.device)).view(1, 1, seq_length, seq_length)
        if attention_mask is not None:
            combined_mask = causal_mask * attention_mask[:, None, None, :].to(causal_mask.dtype)
        else:
            combined_mask = causal_mask

        # Memory-efficient attention implementation that processes the sequence in blocks
        def flash_attention(q, k, v, mask, block_size):
            """
            Efficient blocked implementation of attention to reduce memory usage.
            Processes the sequence in chunks to improve cache locality.
            
            Args:
                q, k, v: Query, key, and value tensors
                mask: Attention mask
                block_size: Size of chunks to process at once
                
            Returns:
                Output tensor after attention
            """
            batch_size, seq_len, num_heads, _ = q.shape
            q = q.permute(0, 2, 1, 3)  # [batch, head, seq, dim]
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            
            # Optimize memory access with contiguous tensors
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            
            # Preallocate output to avoid repeated memory allocations
            output = torch.zeros_like(v)
            
            # Process blocks of the sequence for better memory efficiency
            for i in range(0, seq_len, block_size):
                j_end = min(i + block_size, seq_len)
                q_block = q[:, :, i:j_end, :]
                
                # Process blocks of sequences to improve cache locality
                for j in range(0, j_end, block_size):
                    j_block_end = min(j + block_size, j_end)
                    k_block = k[:, :, j:j_block_end, :]
                    v_block = v[:, :, j:j_block_end, :]
                    
                    # Compute attention scores and apply mask
                    scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(self.latent_dim)
                    if mask is not None:
                        mask_block = mask[:, :, i:j_end, j:j_block_end]
                        scores = scores.masked_fill(mask_block == 0, -1e9)
                    
                    # Apply softmax and dropout
                    attn = F.softmax(scores, dim=-1)
                    attn = self.dropout(attn)
                    context = torch.matmul(attn, v_block)
                    output[:, :, i:j_end, :] += context
            
            return output.permute(0, 2, 1, 3)  # [batch, seq, head, dim]
        
        # Apply flash attention for memory efficiency
        context = flash_attention(q_rot, k_rot, v, combined_mask, self.block_size)
        context = context.reshape(batch_size, seq_length, self.hidden_size)
        return self.out_proj(context)

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization - a simpler alternative to LayerNorm
    that normalizes by RMS of activations rather than mean and variance.
    """
    def __init__(self, hidden_size, eps):
        """
        Initialize RMSNorm.
        
        Args:
            hidden_size: Dimension of the input
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Calculate root mean square along last dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x  # Scale by learned parameters

class Block(nn.Module):
    """
    Transformer block with pre-norm architecture (norm before attention/ffn).
    Uses RMSNorm and multi-head latent attention.
    """
    def __init__(self, config):
        """
        Initialize a transformer block.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super().__init__()
        self.attention = MultiHeadLatentAttention(config)
        self.norm1 = RMSNorm(config['hidden_size'], config['rms_norm_eps'])
        self.norm2 = RMSNorm(config['hidden_size'], config['rms_norm_eps'])
        # Feed-forward network with expansion and GELU activation
        self.ff = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size'] * config['ff_expansion_factor']),
            nn.GELU(),
            nn.Linear(config['hidden_size'] * config['ff_expansion_factor'], config['hidden_size']),
            nn.Dropout(config['dropout'])
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input through the transformer block.
        
        Args:
            x: Input tensor
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor after block processing
        """
        # Pre-norm architecture: normalize before attention and feedforward
        normalized = self.norm1(x)
        x = x + self.attention(normalized, attention_mask)  # Residual connection
        normalized = self.norm2(x)
        x = x + self.ff(normalized)  # Residual connection
        return x

class Dataset(Dataset):
    """
    PyTorch Dataset for language modeling with sliding window approach.
    Creates training examples from a sequence of token IDs.
    """
    def __init__(self, token_ids, seq_length):
        """
        Initialize dataset.
        
        Args:
            token_ids: List of token IDs
            seq_length: Length of sequence for each example
        """
        self.token_ids = token_ids
        # Adjust sequence length to be no longer than the data we have
        self.seq_length = min(seq_length, max(1, len(token_ids) // 2))
        
        # Ensure we have at least one example
        if len(token_ids) <= self.seq_length:
            # Pad the tokens to ensure we have at least one example
            self.token_ids = token_ids + [0] * (self.seq_length + 1 - len(token_ids))
            logger.warning(f"Data too small, padded from {len(token_ids)} to {len(self.token_ids)} tokens")

    def __len__(self):
        """Return the number of examples in the dataset."""
        return max(1, len(self.token_ids) - self.seq_length)

    def __getitem__(self, idx):
        """
        Get a training example at the given index.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary containing input_ids, labels, and attention_mask
        """
        # Ensure we don't go out of bounds
        idx = min(idx, len(self.token_ids) - self.seq_length - 1)
        
        # Input is sequence starting at idx
        input_ids = self.token_ids[idx:idx + self.seq_length]
        # Labels are the next tokens (shifted by 1)
        labels = self.token_ids[idx + 1:idx + self.seq_length + 1]
        
        # Ensure both are the right length
        if len(input_ids) < self.seq_length:
            input_ids = input_ids + [0] * (self.seq_length - len(input_ids))
        if len(labels) < self.seq_length:
            labels = labels + [0] * (self.seq_length - len(labels))
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.ones(self.seq_length, dtype=torch.long)
        }

class Model(nn.Module):
    """
    Language model based on the transformer architecture.
    Consists of token embeddings, multiple transformer blocks, and a prediction head.
    """
    def __init__(self, config):
        """
        Initialize the language model.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['num_layers'])])
        self.norm = RMSNorm(config['hidden_size'], config['rms_norm_eps'])
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights of the model.
        
        Args:
            module: The module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Logits for next token prediction [batch, seq_len, vocab_size]
        """
        hidden_states = self.token_embeddings(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def generate(self, tokenizer, prompt, max_length, top_k, device):
        """
        Generate text given a prompt.
        
        Args:
            tokenizer: BPE tokenizer instance
            prompt: Text prompt to continue
            max_length: Maximum number of tokens to generate
            top_k: Number of top tokens to sample from
            device: Device to run generation on
            
        Returns:
            Generated text including the prompt
        """
        self.eval()
        input_tokens = tokenizer.tokenize(prompt)
        input_ids = torch.tensor([tokenizer.vocab[token] for token in input_tokens], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            for _ in range(max_length):
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
                outputs = self(input_ids, attention_mask)
                next_token_logits = outputs[:, -1, :]  # Get logits for next token
                # Sample from top-k options for variety
                top_k_logits, top_k_indices = next_token_logits.topk(top_k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token = top_k_indices.gather(-1, torch.multinomial(probs, 1))
                input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_ids = input_ids.squeeze(0).tolist()
            return tokenizer.detokenize(generated_ids)

class WarmupLearningRate:
    """
    Learning rate scheduler with linear warmup and cosine decay.
    Gradually increases the learning rate during warmup phase, then
    decreases it following a cosine schedule.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, peak_lr):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            peak_lr: Maximum learning rate after warmup
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.step_num = 0

    def step(self):
        """
        Update the learning rate based on the current step.
        """
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            # Linear warmup
            lr = self.peak_lr * self.step_num / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.peak_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class PrefetchDataLoader:
    """
    DataLoader wrapper that prefetches the next batch while the current batch
    is being processed to reduce data loading bottlenecks.
    Particularly useful for GPU acceleration.
    """
    def __init__(self, dataloader, device):
        """
        Initialize the prefetch dataloader.
        
        Args:
            dataloader: Base dataloader to wrap
            device: Device to preload data to
        """
        self.dataloader = dataloader
        self.device = device
        self.iter = None
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self.next_batch = None

    def __iter__(self):
        """Set up iterator and preload first batch."""
        self.iter = iter(self.dataloader)
        self.preload()
        return self

    def preload(self):
        """Preload the next batch on the appropriate device."""
        try:
            self.next_batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
            
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_batch = {k: v.to(self.device, non_blocking=True) 
                                  for k, v in self.next_batch.items()}

    def __next__(self):
        """Return the preloaded batch and load the next batch in background."""
        if self.next_batch is None:
            raise StopIteration

        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

    def __len__(self):
        """Return the length of the wrapped dataloader."""
        return len(self.dataloader)

class Trainer:
    """
    Trainer class for managing the training process of the language model.
    Handles configuration, data, optimization, and evaluation.
    """
    def __init__(self):
        """Initialize the trainer with configuration and set up device."""
        self.config = self.create_config()
        self.device = torch.device("mps")
        logger.info(f"Trainer initialized with device: {self.device}")

    def create_config(self):
        """
        Create and return the model and training configuration.
        
        Returns:
            Dictionary with configuration parameters
        """
        config = {
            # Model architecture
            'vocab_size': 500,                # Initial value, will be updated based on tokenizer
            'hidden_size': 256,               # Dimension of hidden layers
            'num_layers': 4,                  # Number of transformer blocks
            'num_heads': 4,                   # Number of attention heads
            'latent_dim': 64,                 # Dimension of latent projection for attention
            'block_size': 64,                 # Size of chunks for blocked attention
            'max_position_embeddings': 512,   # Maximum sequence length
            'dropout': 0.1,                   # Dropout rate
            'ff_expansion_factor': 2,         # Multiplier for FFN hidden layer size
            'rms_norm_eps': 1e-6,             # Epsilon for RMSNorm

            # Training parameters
            'learning_rate': 5e-4,            # Peak learning rate
            'weight_decay': 0.01,             # Weight decay for AdamW optimizer
            'batch_size': 8,                  # Batch size per update
            'gradient_accumulation_steps': 4, # Number of steps to accumulate gradients
            'num_epochs': 2,                  # Number of training epochs
            'warmup_steps': 500,              # Steps for learning rate warmup
            'seq_length': 64,                 # Sequence length for training
            
            # Generation parameters
            'max_gen_length': 50,             # Maximum tokens to generate
            'top_k': 50,                      # Top-k sampling parameter
            
            # Z-loss scaling factor
            'z_loss_base': 1e-5,              # Base value for z-loss (prevents output distribution collapse)
            
            # Precision settings
            'precision': 'fp32',              # Data precision ('fp32', 'fp16', 'bf16')
        }
        logger.info("Configuration created with parameters: %s", config)
        return config

    def train_model(self, model, train_dataloader, val_dataloader):
        # Move model to the appropriate device (CPU, GPU, or MPS)
        model = model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")
        
        # Enable PyTorch 2.0+ compilation if available
        if hasattr(torch, 'compile') and self.device.type != 'mps':
            logger.info("Using torch.compile for performance optimization")
            model = torch.compile(model)
        else:
            logger.info(f"Skipping torch.compile as it's not supported on {self.device.type}")
        
        # Set numerical precision based on configuration
        precision = self.config['precision']
        
        if precision == 'fp16':
            dtype = torch.float16  # 16-bit floating point (half precision)
        elif precision == 'bf16':
            dtype = torch.bfloat16  # Brain floating point format (better numerical stability than fp16)
        else:
            dtype = torch.float32  # 32-bit floating point (full precision)
        
        logger.info(f"Using precision: {precision}")

        try:
            # Attempt to convert model parameters to the specified precision
            for param in model.parameters():
                param.data = param.data.to(dtype)
            logger.info(f"Model converted to {precision}")
        except Exception as e:
            # Fallback to fp32 if conversion fails
            logger.warning(f"Failed to convert model to {precision}: {e}")
            logger.info("Falling back to fp32")
            precision = 'fp32'
            dtype = torch.float32
        
        # Initialize optimizer with weight decay to prevent overfitting
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Calculate total training steps and prepare learning rate scheduler with warmup
        total_steps = self.config['num_epochs'] * len(train_dataloader)
        scheduler = WarmupLearningRate(optimizer, self.config['warmup_steps'], total_steps, self.config['learning_rate'])
        
        # Calculate Z-loss weight (scales with model size and vocabulary size)
        # Z-loss helps stabilize training by penalizing large logit values
        z_loss_weight = self.config['z_loss_base'] * (self.config['hidden_size'] / 512) * (self.config['vocab_size'] / 1000)
        
        logger.info(f"Starting training for {self.config['num_epochs']} epochs with {precision} precision")

        # Begin training loop
        model.train()
        for epoch in range(self.config['num_epochs']):
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(progress_bar):
                # Move batch data to the appropriate device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                logits = outputs.view(-1, self.config['vocab_size'])
                labels_flat = labels.view(-1)
                
                # Calculate cross-entropy loss (primary loss function)
                ce_loss = F.cross_entropy(logits, labels_flat)
                
                # Add Z-loss to stabilize training (prevents logits from growing too large)
                z_loss = z_loss_weight * (logits.logsumexp(dim=-1) ** 2).mean()
                loss = ce_loss + z_loss
                
                # Scale loss for gradient accumulation (allows effective larger batch sizes)
                loss = loss / self.config['gradient_accumulation_steps']
                
                # Backward pass to calculate gradients
                loss.backward()
                
                # Only update weights after accumulating gradients for specified number of steps
                if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    logger.debug(f"Step {step+1}: Optimizer step executed, gradients cleared")
                
                # Update learning rate according to schedule
                scheduler.step()
                current_lr = scheduler.optimizer.param_groups[0]['lr']
                total_loss += loss.item() * self.config['gradient_accumulation_steps']
                
                # Log progress less frequently to reduce overhead
                if step % 500 == 0:
                    logger.info(f"Epoch {epoch+1}, Step {step}: Loss = {loss.item():.4f}, LR = {current_lr:.6f}")
                
                progress_bar.set_postfix({'loss': total_loss / (step + 1), 'lr': current_lr})
            
            # End of epoch processing
            avg_train_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} completed - Average training loss: {avg_train_loss:.4f}")
            
            # Evaluate model on validation data
            val_loss, val_perplexity = self.evaluate(model, val_dataloader)
            logger.info(f"Epoch {epoch+1} - Validation loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}")
        
        logger.info("Training completed")
        return model

    def load_model(self, model, path):
        """Load a pre-trained model from disk"""
        logger.info(f"Loading model from path: {path}")
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        logger.info("Model successfully loaded")
        return model

    def evaluate(self, model, dataloader):
        """Evaluate model performance on a dataset"""
        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        # Calculate Z-loss weight (same formula as in training)
        z_loss_weight = self.config['z_loss_base'] * (self.config['hidden_size'] / 512) * (self.config['vocab_size'] / 1000)
        logger.info(f"Starting evaluation on {len(dataloader)} batches")
        
        # Disable gradient calculation for evaluation (saves memory and speeds up inference)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # Move batch data to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Forward pass
                outputs = model(input_ids, attention_mask)
                logits = outputs.view(-1, self.config['vocab_size'])
                labels_flat = labels.view(-1)
                
                # Calculate loss with sum reduction (to properly account for total tokens)
                ce_loss = F.cross_entropy(logits, labels_flat, reduction='sum')
                z_loss = z_loss_weight * (logits.logsumexp(dim=-1) ** 2).sum()
                loss = ce_loss + z_loss
                
                total_loss += loss.item()
                total_tokens += labels.numel()
                if i % 50 == 0:
                    logger.debug(f"Evaluation batch {i}: Loss contribution = {loss.item():.4f}")
        
        # Calculate per-token loss and perplexity (a standard language model metric)
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        logger.info(f"Evaluation completed - Average loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        # Return model to training mode
        model.train()
        return avg_loss, perplexity

    def prepare_tokenized_data(self, text, tokenizer, vocab, cache_file=None):
        """Cache tokenized data to disk to speed up subsequent runs"""
        # Check if cached data exists and load it if available
        if cache_file and os.path.exists(cache_file):
            logger.info(f"Loading tokenized data from cache: {cache_file}")
            return torch.load(cache_file)
        
        # Tokenize data if cache doesn't exist
        logger.info("Tokenizing dataset")
        
        # Add handling for unknown tokens
        token_ids = []
        unknown_token_id = len(vocab)  # Use the next available ID for unknown tokens
        unknown_tokens = set()
        
        for token in tokenizer.tokenize(text):
            if token in vocab:
                token_ids.append(vocab[token])
            else:
                token_ids.append(unknown_token_id)
                unknown_tokens.add(token)
        
        if unknown_tokens:
            logger.warning(f"Found {len(unknown_tokens)} unknown tokens: {unknown_tokens}")
        
        # Save tokenized data to cache for future use
        if cache_file:
            logger.info(f"Saving tokenized data to cache: {cache_file}")
            torch.save(token_ids, cache_file)
        
        return token_ids

if __name__ == "__main__":
    # Entry point when running script directly
    logger.info("Starting main execution")
    
    # Download sample dataset (Shakespeare text)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    logger.info(f"Downloading dataset from {url}")
    response = requests.get(url)
    with open("tinyshakespeare.txt", "w") as f:
        f.write(response.text)
    logger.info("Dataset downloaded and saved")

    # Load dataset into memory
    with open("tinyshakespeare.txt", "r") as f:
        text = f.read()
    logger.info(f"Loaded text data with length: {len(text)}")

    # Split data into training, validation, and test sets
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    total_length = len(text)
    train_end = int(train_ratio * total_length)
    val_end = int((train_ratio + val_ratio) * total_length)
    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]
    logger.info(f"Dataset split - Train: {len(train_text)}, Val: {len(val_text)}, Test: {len(test_text)}")

    # Create and train tokenizer with target vocabulary size
    desired_vocab_size = 1000
    tokenizer = BPETokenizer()
    vocab, merge_rules = tokenizer.train(train_text, desired_vocab_size)

    # Create cache directory for tokenized data
    os.makedirs("cache", exist_ok=True)
    
    # Create the trainer instance
    trainer = Trainer()
    
    # Set device to Apple Silicon GPU if available, otherwise CPU
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
    
    # Tokenize and prepare datasets (with caching for efficiency)
    train_token_ids = trainer.prepare_tokenized_data(train_text, tokenizer, vocab, "cache/train_tokens.pt")
    val_token_ids = trainer.prepare_tokenized_data(val_text, tokenizer, vocab, "cache/val_tokens.pt")
    test_token_ids = trainer.prepare_tokenized_data(test_text, tokenizer, vocab, "cache/test_tokens.pt")
    logger.info(f"Tokenization complete - Train tokens: {len(train_token_ids)}, Val tokens: {len(val_token_ids)}, Test tokens: {len(test_token_ids)}")

    # Create PyTorch datasets from token IDs
    train_dataset = Dataset(train_token_ids, trainer.config['seq_length'])
    val_dataset = Dataset(val_token_ids, trainer.config['seq_length'])
    test_dataset = Dataset(test_token_ids, trainer.config['seq_length'])
    logger.info(f"Datasets created - Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # Update config with actual vocabulary size plus one for unknown tokens
    trainer.config['vocab_size'] = len(vocab) + 1
    
    # Adjust sequence length based on the data size
    max_data_size = max(len(train_token_ids), len(val_token_ids), len(test_token_ids))
    trainer.config['seq_length'] = min(trainer.config['seq_length'], max(1, max_data_size // 2))
    logger.info(f"Adjusted sequence length to {trainer.config['seq_length']} based on data size")

    # Create data loaders for efficient batched processing
    train_dataloader = DataLoader(train_dataset, batch_size=trainer.config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=trainer.config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=trainer.config['batch_size'], shuffle=False)
    logger.info(f"Dataloaders created - Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}, Test batches: {len(test_dataloader)}")

    # Initialize the model
    model = Model(trainer.config)
    logger.info("Model initialized")
    
    # Train the model
    trained_model = trainer.train_model(model, train_dataloader, val_dataloader)

    # Evaluate on test set
    test_loss, test_perplexity = trainer.evaluate(trained_model, test_dataloader)
    logger.info(f"Final test results - Loss: {test_loss:.4f}, Perplexity: {test_perplexity:.4f}")

    # Save trained model to disk
    torch.save(trained_model.state_dict(), "model.pth")
    logger.info("Model saved to model.pth")

    # Generate sample text with the trained model
    prompt = "To be or not to be"
    logger.info(f"Generating text with prompt: '{prompt}'")
    generated_text = trained_model.generate(tokenizer, prompt, max_length=50, top_k=50, device=device)
    logger.info(f"Generated text: {generated_text}")
    logger.info("Execution completed")