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

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"language_model_{time.strftime('%Y%m%d')}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
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
    def __init__(self):
        self.vocab = {}
        self.merge_rules = []
        self.id_to_token = {}
        logger.info("Initialized BPETokenizer")

    def train(self, text, desired_vocab_size):
        logger.info(f"Starting tokenizer training with desired vocab size: {desired_vocab_size}")
        chars = sorted(list(set(text)))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        tokens = list(text)
        num_merges = desired_vocab_size - len(self.vocab)
        logger.info(f"Initial vocabulary size: {len(self.vocab)}, Number of merges required: {num_merges}")
        
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
        logger.debug(f"Tokenizing text of length: {len(text)}")
        tokens = list(text)
        for pair in self.merge_rules:
            tokens = self.merge_pair(tokens, pair)
        logger.debug(f"Tokenization complete, produced {len(tokens)} tokens")
        return tokens

    def detokenize(self, token_ids):
        logger.debug(f"Detokenizing {len(token_ids)} token IDs")
        tokens = [self.id_to_token[id] for id in token_ids]
        result = ''.join(tokens)
        logger.debug(f"Detokenization complete, output length: {len(result)}")
        return result

    def get_pair_freq(self, tokens):
        pair_freq = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freq[pair] += 1
        logger.debug(f"Found {len(pair_freq)} unique pairs")
        return pair_freq

    def merge_pair(self, tokens, pair):
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
    def __init__(self, dim: int, max_position: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position = max_position

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = self.hidden_size // self.num_heads
        self.latent_dim = config['latent_dim']
        self.q_latent_proj = nn.Linear(self.head_dim, self.latent_dim)
        self.k_latent_proj = nn.Linear(self.head_dim, self.latent_dim)
        self.v_proj = nn.Linear(self.head_dim, self.head_dim)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config['dropout'])
        self.rotary_emb = RotaryPositionalEmbedding(self.latent_dim, config['max_position_embeddings'])
        self.block_size = config.get('block_size', 64)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = hidden_states.shape[:2]
        hidden_states = hidden_states.view(batch_size, seq_length, self.num_heads, self.head_dim)

        q = self.q_latent_proj(hidden_states)
        k = self.k_latent_proj(hidden_states)
        v = self.v_proj(hidden_states)

        positions = torch.arange(seq_length, device=hidden_states.device).float()
        cos, sin = self.rotary_emb(positions)
        cos = cos.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.num_heads, -1)
        sin = sin.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, self.num_heads, -1)
        q_rot = q.clone()
        k_rot = k.clone()
        q_rot[..., ::2] = q[..., ::2] * cos[..., ::2] - q[..., 1::2] * sin[..., ::2]
        q_rot[..., 1::2] = q[..., ::2] * sin[..., ::2] + q[..., 1::2] * cos[..., ::2]
        k_rot[..., ::2] = k[..., ::2] * cos[..., ::2] - k[..., 1::2] * sin[..., ::2]
        k_rot[..., 1::2] = k[..., ::2] * sin[..., ::2] + k[..., 1::2] * cos[..., ::2]

        causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=hidden_states.device)).view(1, 1, seq_length, seq_length)
        if attention_mask is not None:
            combined_mask = causal_mask * attention_mask[:, None, None, :].to(causal_mask.dtype)
        else:
            combined_mask = causal_mask

        def flash_attention(q, k, v, mask, block_size):
            batch_size, seq_len, num_heads, _ = q.shape
            q = q.permute(0, 2, 1, 3)  # [batch, head, seq, dim]
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            
            # Unified reshaping to optimize memory access
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            
            # Preallocate output to avoid repeated memory allocations
            output = torch.zeros_like(v)
            
            # Use torch.baddbmm for optimized batched matrix multiplication
            for i in range(0, seq_len, block_size):
                j_end = min(i + block_size, seq_len)
                q_block = q[:, :, i:j_end, :]
                
                # Process blocks of sequences to improve cache locality
                for j in range(0, j_end, block_size):
                    j_block_end = min(j + block_size, j_end)
                    k_block = k[:, :, j:j_block_end, :]
                    v_block = v[:, :, j:j_block_end, :]
                    
                    scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(self.latent_dim)
                    if mask is not None:
                        mask_block = mask[:, :, i:j_end, j:j_block_end]
                        scores = scores.masked_fill(mask_block == 0, -1e9)
                    
                    attn = F.softmax(scores, dim=-1)
                    attn = self.dropout(attn)
                    context = torch.matmul(attn, v_block)
                    output[:, :, i:j_end, :] += context
            
            return output.permute(0, 2, 1, 3)  # [batch, seq, head, dim]
        
        context = flash_attention(q_rot, k_rot, v, combined_mask, self.block_size)
        context = context.view(batch_size, seq_length, self.hidden_size)
        return self.out_proj(context)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadLatentAttention(config)
        self.norm1 = RMSNorm(config['hidden_size'], config['rms_norm_eps'])
        self.norm2 = RMSNorm(config['hidden_size'], config['rms_norm_eps'])
        self.ff = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size'] * config['ff_expansion_factor']),
            nn.GELU(),
            nn.Linear(config['hidden_size'] * config['ff_expansion_factor'], config['hidden_size']),
            nn.Dropout(config['dropout'])
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normalized = self.norm1(x)
        x = x + self.attention(normalized, attention_mask)
        normalized = self.norm2(x)
        x = x + self.ff(normalized)
        return x

class Dataset(Dataset):
    def __init__(self, token_ids, seq_length):
        self.token_ids = token_ids
        self.seq_length = seq_length

    def __len__(self):
        return len(self.token_ids) - self.seq_length

    def __getitem__(self, idx):
        input_ids = self.token_ids[idx:idx + self.seq_length]
        labels = self.token_ids[idx + 1:idx + self.seq_length + 1]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.ones(self.seq_length, dtype=torch.long)
        }

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['num_layers'])])
        self.norm = RMSNorm(config['hidden_size'], config['rms_norm_eps'])
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.token_embeddings(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def generate(self, tokenizer, prompt, max_length, top_k, device):
        self.eval()
        input_tokens = tokenizer.tokenize(prompt)
        input_ids = torch.tensor([tokenizer.vocab[token] for token in input_tokens], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            for _ in range(max_length):
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
                outputs = self(input_ids, attention_mask)
                next_token_logits = outputs[:, -1, :]
                top_k_logits, top_k_indices = next_token_logits.topk(top_k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token = top_k_indices.gather(-1, torch.multinomial(probs, 1))
                input_ids = torch.cat([input_ids, next_token], dim=-1)
            generated_ids = input_ids.squeeze(0).tolist()
            return tokenizer.detokenize(generated_ids)

    if hasattr(torch, 'compile'):
        logger.info("Using torch.compile for performance optimization")
        model = torch.compile(model)

class WarmupLearningRate:
    def __init__(self, optimizer, warmup_steps, total_steps, peak_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.peak_lr * self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.peak_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class PrefetchDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.iter = None
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self.next_batch = None

    def __iter__(self):
        self.iter = iter(self.dataloader)
        self.preload()
        return self

    def preload(self):
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
        if self.next_batch is None:
            raise StopIteration

        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

    def __len__(self):
        return len(self.dataloader)

class Trainer:
    def __init__(self):
        self.config = self.create_config()
        self.device = torch.device("mps")
        logger.info(f"Trainer initialized with device: {self.device}")

    def create_config(self):
        config = {
            # Model architecture
            'vocab_size': 500,
            'hidden_size': 256,
            'num_layers': 4,
            'num_heads': 4,
            'latent_dim': 64,
            'block_size': 64,
            'max_position_embeddings': 512,
            'dropout': 0.1,
            'ff_expansion_factor': 2,
            'rms_norm_eps': 1e-6,

            # Training parameters
            'learning_rate': 5e-4,
            'weight_decay': 0.01,
            'batch_size': 8,
            'gradient_accumulation_steps': 4,
            'num_epochs': 2,
            'warmup_steps': 500,
            'seq_length': 64,
            
            # Generation parameters
            'max_gen_length': 50,
            'top_k': 50,
            
            # Z-loss scaling factor
            'z_loss_base': 1e-5,
            
            # Precision settings
            'precision': 'fp32',  # Options: 'fp32', 'fp16', 'bf16'
        }
        logger.info("Configuration created with parameters: %s", config)
        return config

    def train_model(self, model, train_dataloader, val_dataloader):
        model = model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")
        
        # Set precision based on config
        precision = self.config['precision']
        
        if precision == 'fp16':
            dtype = torch.float16
        elif precision == 'bf16':
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        logger.info(f"Using precision: {precision}")

        try:
            # Try to convert model parameters to the desired dtype
            for param in model.parameters():
                param.data = param.data.to(dtype)
            logger.info(f"Model converted to {precision}")
        except Exception as e:
            logger.warning(f"Failed to convert model to {precision}: {e}")
            logger.info("Falling back to fp32")
            precision = 'fp32'
            dtype = torch.float32
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        total_steps = self.config['num_epochs'] * len(train_dataloader)
        scheduler = WarmupLearningRate(optimizer, self.config['warmup_steps'], total_steps, self.config['learning_rate'])
        z_loss_weight = self.config['z_loss_base'] * (self.config['hidden_size'] / 512) * (self.config['vocab_size'] / 1000)
        
        
        logger.info(f"Starting training for {self.config['num_epochs']} epochs with {precision} precision")

        model.train()
        for epoch in range(self.config['num_epochs']):
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
            for step, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                logits = outputs.view(-1, self.config['vocab_size'])
                labels_flat = labels.view(-1)
                ce_loss = F.cross_entropy(logits, labels_flat)
                z_loss = z_loss_weight * (logits.logsumexp(dim=-1) ** 2).mean()
                loss = ce_loss + z_loss
                loss = loss / self.config['gradient_accumulation_steps']
                loss.backward()
                
                if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    logger.debug(f"Step {step+1}: Optimizer step executed, gradients cleared")
                
                scheduler.step()
                current_lr = scheduler.optimizer.param_groups[0]['lr']
                total_loss += loss.item() * self.config['gradient_accumulation_steps']
                
                # Reduce logging frequency
                if step % 500 == 0:
                    logger.info(f"Epoch {epoch+1}, Step {step}: Loss = {loss.item():.4f}, LR = {current_lr:.6f}")
                
                progress_bar.set_postfix({'loss': total_loss / (step + 1), 'lr': current_lr})
            
            avg_train_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} completed - Average training loss: {avg_train_loss:.4f}")
            
            val_loss, val_perplexity = self.evaluate(model, val_dataloader)
            logger.info(f"Epoch {epoch+1} - Validation loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}")
        
        logger.info("Training completed")
        return model

    def load_model(self, model, path):
        logger.info(f"Loading model from path: {path}")
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        logger.info("Model successfully loaded")
        return model

    def evaluate(self, model, dataloader):
        model.eval()
        total_loss = 0
        total_tokens = 0
        z_loss_weight = self.config['z_loss_base'] * (self.config['hidden_size'] / 512) * (self.config['vocab_size'] / 1000)
        logger.info(f"Starting evaluation on {len(dataloader)} batches")
        
        # Get precision settings from config
        precision = self.config['precision']
        
        if precision == 'fp16':
            dtype = torch.float16
        elif precision == 'bf16':
            dtype = torch.bfloat16
        else:  # default to fp32
            dtype = torch.float32
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = model(input_ids, attention_mask)
                logits = outputs.view(-1, self.config['vocab_size'])
                labels_flat = labels.view(-1)
                ce_loss = F.cross_entropy(logits, labels_flat, reduction='sum')
                z_loss = z_loss_weight * (logits.logsumexp(dim=-1) ** 2).sum()
                loss = ce_loss + z_loss
                
                total_loss += loss.item()
                total_tokens += labels.numel()
                if i % 50 == 0:
                    logger.debug(f"Evaluation batch {i}: Loss contribution = {loss.item():.4f}")
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        logger.info(f"Evaluation completed - Average loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        model.train()
        return avg_loss, perplexity

    def prepare_tokenized_data(self, text, tokenizer, vocab, cache_file=None):
        """Cache tokenized data to disk to speed up subsequent runs"""
        if cache_file and os.path.exists(cache_file):
            logger.info(f"Loading tokenized data from cache: {cache_file}")
            return torch.load(cache_file)
        
        logger.info("Tokenizing dataset")
        token_ids = [vocab[token] for token in tokenizer.tokenize(text)]
        
        if cache_file:
            logger.info(f"Saving tokenized data to cache: {cache_file}")
            torch.save(token_ids, cache_file)
        
        return token_ids

if __name__ == "__main__":
    logger.info("Starting main execution")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    logger.info(f"Downloading dataset from {url}")
    response = requests.get(url)
    with open("tinyshakespeare.txt", "w") as f:
        f.write(response.text)
    logger.info("Dataset downloaded and saved")

    with open("tinyshakespeare.txt", "r") as f:
        text = f.read()
    logger.info(f"Loaded text data with length: {len(text)}")

    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    total_length = len(text)
    train_end = int(train_ratio * total_length)
    val_end = int((train_ratio + val_ratio) * total_length)
    train_text = text[:train_end]
    val_text = text[train_end:val_end]
    test_text = text[val_end:]
    logger.info(f"Dataset split - Train: {len(train_text)}, Val: {len(val_text)}, Test: {len(test_text)}")

    desired_vocab_size = 1000
    tokenizer = BPETokenizer()
    vocab, merge_rules = tokenizer.train(train_text, desired_vocab_size)

    os.makedirs("cache", exist_ok=True)
    
    # Create the trainer instance BEFORE using it
    trainer = Trainer()
    
    # Configure for MPS - set precision and disable mixed precision
    if torch.backends.mps.is_available():
        logger.info("MPS is available - configuring for optimal performance")
        # For MPS, FP32 is generally more reliable than FP16
        trainer.config['precision'] = 'fp32'
        logger.info(f"Set precision to {trainer.config['precision']} for MPS")
    
    # Now we can use the trainer methods
    train_token_ids = trainer.prepare_tokenized_data(train_text, tokenizer, vocab, "cache/train_tokens.pt")
    val_token_ids = trainer.prepare_tokenized_data(val_text, tokenizer, vocab, "cache/val_tokens.pt")
    test_token_ids = trainer.prepare_tokenized_data(test_text, tokenizer, vocab, "cache/test_tokens.pt")
    logger.info(f"Tokenization complete - Train tokens: {len(train_token_ids)}, Val tokens: {len(val_token_ids)}, Test tokens: {len(test_token_ids)}")

    train_dataset = Dataset(train_token_ids, trainer.config['seq_length'])
    val_dataset = Dataset(val_token_ids, trainer.config['seq_length'])
    test_dataset = Dataset(test_token_ids, trainer.config['seq_length'])
    logger.info(f"Datasets created - Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    trainer.config['vocab_size'] = len(vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=trainer.config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=trainer.config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=trainer.config['batch_size'], shuffle=False)
    logger.info(f"Dataloaders created - Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}, Test batches: {len(test_dataloader)}")

    device = torch.device("mps")
    model = Model(trainer.config)
    logger.info("Model initialized")
    trained_model = trainer.train_model(model, train_dataloader, val_dataloader)

    test_loss, test_perplexity = trainer.evaluate(trained_model, test_dataloader)
    logger.info(f"Final test results - Loss: {test_loss:.4f}, Perplexity: {test_perplexity:.4f}")

    torch.save(trained_model.state_dict(), "model.pth")
    logger.info("Model saved to model.pth")

    prompt = "To be or not to be"
    logger.info(f"Generating text with prompt: '{prompt}'")
    generated_text = trained_model.generate(tokenizer, prompt, max_length=50, top_k=50, device=device)
    logger.info(f"Generated text: {generated_text}")
    logger.info("Execution completed")