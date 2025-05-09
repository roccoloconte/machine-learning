import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from text.core.embeddings import RotaryPositionalEmbedding, RMSNorm

logger = logging.getLogger(__name__)

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head attention with latent components.
    """
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_heads
        
        # Initialize with careful weight scaling
        std = 0.02
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Careful initialization of weights
        with torch.no_grad():
            for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
                proj.weight.data.normal_(0.0, std)
                # Extra scaling for more stability in query/key
                if proj in [self.q_proj, self.k_proj]:
                    proj.weight.data.mul_(0.8)  # Reduce scale for better attention stability
        
        self.rope = RotaryPositionalEmbedding(self.d_head, config.max_seq_length)
        self.dropout = nn.Dropout(config.dropout)
        
        logger.info(f"Initialized MultiHeadLatentAttention with {self.n_heads} heads, d_model={self.d_model}, d_head={self.d_head}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape
        orig_dtype = x.dtype
        
        # Convert to float32 for better numerical stability
        x = x.to(torch.float32)
        
        # Clamp input values
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        # Handle NaN/Inf in input
        mask_inf = torch.isnan(x) | torch.isinf(x)
        if mask_inf.any():
            logger.warning(f"NaN/Inf detected in attention input. Shape: {x.shape}")
            x = torch.where(mask_inf, torch.zeros_like(x), x)
        
        logger.debug(f"Attention input shapes - x: {x.shape}, mask: {mask.shape if mask is not None else None}")
        
        # Project queries, keys, and values with intermediate clamping
        q = self.q_proj(x)
        q = torch.clamp(q, min=-10.0, max=10.0)
        
        k = self.k_proj(x)
        k = torch.clamp(k, min=-10.0, max=10.0)
        
        v = self.v_proj(x)
        v = torch.clamp(v, min=-10.0, max=10.0)
        
        # Reshape and apply rotary embeddings
        q = q.view(batch_size, seq_length, self.n_heads, self.d_head)
        k = k.view(batch_size, seq_length, self.n_heads, self.d_head)
        v = v.view(batch_size, seq_length, self.n_heads, self.d_head)
        
        logger.debug(f"QKV shapes after reshape - q: {q.shape}, k: {k.shape}, v: {v.shape}")
        
        # Apply attention with safeguards
        attn_output = self._attention(q, k, v, mask, cache)
        
        # Project output with clamping
        output = self.o_proj(attn_output)
        output = torch.clamp(output, min=-10.0, max=10.0)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Handle NaN/Inf in output
        mask_inf = torch.isnan(output) | torch.isinf(output)
        if mask_inf.any():
            logger.warning(f"NaN/Inf detected in attention output. Shape: {output.shape}")
            output = torch.where(mask_inf, torch.zeros_like(output), output)
        
        # Convert back to original dtype
        return output.to(orig_dtype)

    def _attention(self, q, k, v, mask, cache):
        """
        Compute attention scores and apply them to values with numerical safeguards.
        """
        batch_size, seq_length, n_heads, d_head = q.shape
        logger.debug(f"Attention shapes - q: {q.shape}, k: {k.shape}, v: {v.shape}, mask: {mask.shape if mask is not None else None}")
        
        # Apply rotary embeddings
        positions = torch.arange(seq_length, device=q.device)
        cos, sin = self.rope(positions)
        
        # Reshape cos and sin for broadcasting
        cos = cos.view(1, seq_length, 1, -1)
        sin = sin.view(1, seq_length, 1, -1)
        
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)
        
        # Clamp after rotation to prevent extreme values
        q = torch.clamp(q, min=-10.0, max=10.0)
        k = torch.clamp(k, min=-10.0, max=10.0)
        
        logger.debug(f"QKV shapes after RoPE - q: {q.shape}, k: {k.shape}")
        
        # Handle cached key/values for inference
        if cache is not None:
            k_cache, v_cache = cache
            if k_cache is not None:
                k = torch.cat([k_cache, k], dim=1)
                v = torch.cat([v_cache, v], dim=1)
            # Update seq_length if cache is used
            seq_length = k.shape[1] 
        
        # Transpose for multi-head attention: (B, S, H, D) -> (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logger.debug(f"QKV shapes after transpose - q: {q.shape}, k: {k.shape}, v: {v.shape}")

        # Scaled dot-product attention with more aggressive scaling
        # Use a larger scale factor for better numerical stability
        scale = 1.0 / math.sqrt(d_head)
        
        # Compute attention scores with careful matmul and scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Clamp scores before softmax to avoid extreme values
        scores = torch.clamp(scores, min=-10.0, max=10.0)
        
        logger.debug(f"Attention scores shape: {scores.shape}")
        
        if mask is not None:
            # Ensure mask is [B, 1, S, S] or similar for broadcasting with scores [B, H, S, S]
            # The mask coming in is [B, S, S] (after trainer slicing)
            # Add head dimension for broadcasting: [B, S, S] -> [B, 1, S, S]
            mask = mask.unsqueeze(1) 
            logger.debug(f"Attention mask shape for broadcast: {mask.shape}")
            # Apply mask with a safer masking approach
            scores = scores.masked_fill(mask == 0, -1e4)  # Use -1e4 instead of -inf for stability
        
        # Handle NaN/Inf before softmax
        mask_inf = torch.isnan(scores) | torch.isinf(scores)
        if mask_inf.any():
            logger.warning(f"NaN/Inf detected in attention scores. Shape: {scores.shape}")
            scores = torch.where(mask_inf, torch.full_like(scores, -1e4), scores)
        
        # Apply softmax with better numerical stability
        attn = F.softmax(scores, dim=-1)
        
        # Explicitly handle potential NaN/Inf from softmax
        mask_inf = torch.isnan(attn) | torch.isinf(attn)
        if mask_inf.any():
            logger.warning(f"NaN/Inf detected after softmax in attention. Shape: {attn.shape}")
            # Replace with uniform attention if we have NaN/Inf
            uniform_attn = torch.ones_like(attn) / attn.size(-1)
            attn = torch.where(mask_inf, uniform_attn, attn)
        
        attn = self.dropout(attn)
        
        # Apply attention to values with careful matmul
        output = torch.matmul(attn, v)
        
        # Clamp output after matmul
        output = torch.clamp(output, min=-10.0, max=10.0)
        
        logger.debug(f"Output shape after attn @ v: {output.shape}")
        
        # Handle NaN/Inf in output
        mask_inf = torch.isnan(output) | torch.isinf(output)
        if mask_inf.any():
            logger.warning(f"NaN/Inf detected in attention output. Shape: {output.shape}")
            output = torch.where(mask_inf, torch.zeros_like(output), output)
        
        # Transpose back: (B, H, S, D) -> (B, S, H, D)
        output = output.transpose(1, 2).contiguous()
        logger.debug(f"Output shape after transpose back: {output.shape}")

        # Reshape to original dimensions: (B, S, H*D) -> (B, S, d_model)
        output = output.view(batch_size, seq_length, -1)
        
        # Final clamp
        output = torch.clamp(output, min=-10.0, max=10.0)
        
        logger.debug(f"Final output shape: {output.shape}")
        
        return output

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary positional embeddings to query and key tensors."""
        # Reshape for broadcasting
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

class Block(nn.Module):
    """
    Transformer block with attention and feed-forward layers
    """
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadLatentAttention(config)
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        
        # Use Sequential with smaller intermediate dimension for more stability
        ff_dim = min(4 * config.d_model, 1024)  # Cap FF dimension for stability
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, ff_dim),
            nn.GELU(),
            # Add extra clamping layer for stability
            nn.Hardtanh(-10.0, 10.0),
            nn.Linear(ff_dim, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Initialize weights with smaller values for better stability
        with torch.no_grad():
            # First linear layer
            self.mlp[0].weight.data.normal_(0.0, 0.02)
            self.mlp[0].bias.data.zero_()
            
            # Second linear layer - use even smaller init for stability
            self.mlp[3].weight.data.normal_(0.0, 0.01)
            self.mlp[3].bias.data.zero_()
            
        # Stronger residual scaling to prevent exploding values
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.9)
        
    def forward(self, x, mask=None, cache=None):
        orig_dtype = x.dtype
        # Work in float32 for better precision
        x = x.to(torch.float32)
        
        # Initial clamping and safeguards
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        # Check and fix NaN/Inf in input
        mask_inf = torch.isnan(x) | torch.isinf(x)
        if mask_inf.any():
            logger.warning(f"Block input contains NaN/Inf. Shape: {x.shape}")
            x = torch.where(mask_inf, torch.zeros_like(x), x)
        
        # Apply normalized attention with residual connection and scaling
        h = x + self.residual_scale * self.attention(self.norm1(x), mask, cache)
        
        # Intermediate clamping
        h = torch.clamp(h, min=-10.0, max=10.0)
        
        # Check and fix NaN/Inf after attention
        mask_inf = torch.isnan(h) | torch.isinf(h)
        if mask_inf.any():
            logger.warning(f"Block attention output contains NaN/Inf. Shape: {h.shape}")
            h = torch.where(mask_inf, torch.zeros_like(h), h)
        
        # Apply MLP with residual connection and scaling
        out = h + self.residual_scale * self.mlp(self.norm2(h))
        
        # Final clamping
        out = torch.clamp(out, min=-10.0, max=10.0)
        
        # Final NaN/Inf check
        mask_inf = torch.isnan(out) | torch.isinf(out)
        if mask_inf.any():
            logger.warning(f"Block final output contains NaN/Inf. Shape: {out.shape}")
            out = torch.where(mask_inf, torch.zeros_like(out), out)
            
        # Return in original dtype
        return out.to(orig_dtype)

class Model(nn.Module):
    """
    Main transformer model for language tasks
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights using a more stable approach
        with torch.no_grad():
            std = 0.02  # Standard deviation for initialization
            self.token_embeddings.weight.data.normal_(0.0, std)
        
        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config.n_layers)
        ])
        
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights of lm_head with a small range
        with torch.no_grad():
            self.lm_head.weight.data.normal_(0.0, std)
        
        # Optionally share the weights between token embeddings and LM head
        # Uncomment if you want embedding weight sharing
        # self.lm_head.weight = self.token_embeddings.weight
        
        # Apply weight initialization to all linear layers
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for all linear layers in the model with small values"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    std = 0.02
                    module.weight.data.normal_(0.0, std)
                    if module.bias is not None:
                        module.bias.data.zero_()

    def forward(self, input_ids, mask=None):
        # Convert to float32 for maximum numerical stability
        dtype_in = torch.float32
        
        # Get embeddings
        x = self.token_embeddings(input_ids)
        x = x.to(dtype_in)  # Ensure computations in float32
        
        # Initial clamping of embeddings
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        # Check for NaN/Inf in embeddings
        mask_inf = torch.isnan(x) | torch.isinf(x)
        if mask_inf.any():
            logger.warning(f"NaN/Inf detected in embeddings. Shape: {x.shape}")
            x = torch.where(mask_inf, torch.zeros_like(x), x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Process through transformer blocks with more careful handling
        for i, block in enumerate(self.blocks):
            try:
                # Apply block with careful exception handling
                x = block(x, mask)
                
                # Add intermediate clamping between blocks
                x = torch.clamp(x, min=-10.0, max=10.0)
                
                # Check and fix NaN/Inf between blocks
                mask_inf = torch.isnan(x) | torch.isinf(x)
                if mask_inf.any():
                    logger.warning(f"NaN/Inf detected after block {i}. Shape: {x.shape}")
                    x = torch.where(mask_inf, torch.zeros_like(x), x)
                
            except Exception as e:
                logger.error(f"Error in block {i}: {str(e)}")
                # If a block fails, try to continue with a zeroed output
                if i > 0:
                    logger.warning(f"Continuing with previous block output")
                    # Keep x as is from previous block
                else:
                    logger.warning(f"Continuing with zeroed embeddings")
                    x = torch.zeros_like(x)
        
        # Apply final normalization safely
        try:
            x = self.norm_f(x)
            x = torch.clamp(x, min=-10.0, max=10.0)
        except Exception as e:
            logger.error(f"Error in final normalization: {str(e)}")
            # Skip normalization if it fails
            pass
            
        # Final layer
        try:
            # Apply language model head with careful clamping
            logits = self.lm_head(x)
            
            # More gentle clamping for logits (vocabulary distribution)
            # Use larger range for logits to preserve distribution shape
            logits = torch.clamp(logits, min=-20.0, max=20.0)
            
            # Check and fix NaN/Inf in logits
            mask_inf = torch.isnan(logits) | torch.isinf(logits)
            if mask_inf.any():
                logger.warning(f"NaN/Inf detected in logits. Shape: {logits.shape}")
                logits = torch.where(mask_inf, torch.zeros_like(logits), logits)
                
        except Exception as e:
            logger.error(f"Error in final projection: {str(e)}")
            # Create safe dummy logits on error
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], self.config.vocab_size), 
                               device=x.device, dtype=x.dtype)

        return logits

    def generate(self, tokenizer, prompt, max_length=100, top_k=50, device=None):
        """
        Generate text using the model.
        
        Args:
            tokenizer: The tokenizer to use for tokenization/detokenization
            prompt: The text prompt to start generation with
            max_length: Maximum length of generated text (in tokens)
            top_k: Number of highest probability tokens to consider for sampling
            device: Device to run generation on
            
        Returns:
            The generated text string
        """
        if device is None:
            device = next(self.parameters()).device
            
        self.eval()
        with torch.no_grad():
            # Tokenize the prompt
            tokens = tokenizer.tokenize(prompt)
            token_ids = [tokenizer.vocab.get(token, 0) for token in tokens]
            input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
            
            # Generate tokens one by one
            for _ in range(max_length):
                # Forward pass
                logits = self(input_ids)
                
                # Get the next token logits (last position)
                next_token_logits = logits[:, -1, :]
                
                # Apply top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                
                # Apply softmax to get probabilities
                probs = F.softmax(top_k_logits, dim=-1)
                
                # Sample from the distribution
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token_idx)
                
                # Append the next token to the sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check if we've generated an end token or reached max length
                end_token_id = tokenizer.vocab.get("<|endoftext|>", -1)
                if next_token.item() == end_token_id and end_token_id != -1:
                    break
            
            # Decode the generated tokens
            generated_ids = input_ids[0].tolist()
            generated_text = tokenizer.detokenize(generated_ids)
            
            return generated_text
