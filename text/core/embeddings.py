import torch
import torch.nn as nn
import logging
import math

logger = logging.getLogger(__name__)

class RotaryPositionalEmbedding(nn.Module):
    """
    Implements rotary positional embeddings (RoPE) for transformer models.
    """
    def __init__(self, dim: int, max_position: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position = max_position

    # In RotaryPositionalEmbedding class
    def forward(self, positions: torch.Tensor):
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


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization with extra stability measures
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        # Initialize with smaller weights close to 1.0
        with torch.no_grad():
            self.weight.data.fill_(1.0)
            # Add small noise to break symmetry
            self.weight.data += torch.randn_like(self.weight.data) * 0.01

    def forward(self, x):
        """
        Apply RMS normalization with additional safety measures.
        """
        orig_dtype = x.dtype
        # Work in float32 for better numerical stability
        x = x.to(torch.float32)
        
        # Aggressive clamping before normalization
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        # Handle NaN/Inf
        mask = torch.isnan(x) | torch.isinf(x)
        if mask.any():
            logger.warning(f"RMSNorm input contains NaN/Inf. Shape: {x.shape}")
            x = torch.where(mask, torch.zeros_like(x), x)
        
        # Calculate RMS along last dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        
        # Apply a floor to variance to prevent division by zero or near-zero
        variance = torch.clamp(variance, min=self.eps * 10)
        
        # Safe sqrt with careful clamping
        inv_std = torch.rsqrt(variance + self.eps)
        inv_std = torch.clamp(inv_std, max=1.0 / math.sqrt(self.eps))
        
        # Handle NaN/Inf in scaling factor
        mask = torch.isnan(inv_std) | torch.isinf(inv_std)
        if mask.any():
            logger.warning(f"RMSNorm inv_std contains NaN/Inf. Shape: {inv_std.shape}")
            inv_std = torch.where(mask, torch.ones_like(inv_std), inv_std)
        
        # Scale inputs with safe inverse std
        normalized_x = x * inv_std
        
        # Clamp normalized values to reasonable range
        normalized_x = torch.clamp(normalized_x, min=-3.0, max=3.0)
        
        # Apply weight scaling with careful clamping
        # Clamp weights to prevent extreme scaling
        clamped_weight = torch.clamp(self.weight, min=0.1, max=2.0)
        output = normalized_x * clamped_weight
        
        # Final aggressive clamping
        output = torch.clamp(output, min=-10.0, max=10.0)
        
        # Handle any remaining NaN/Inf values
        mask = torch.isnan(output) | torch.isinf(output)
        if mask.any():
            logger.warning(f"RMSNorm final output contains NaN/Inf. Shape: {output.shape}")
            output = torch.where(mask, torch.zeros_like(output), output)
            
        # Convert back to original dtype
        return output.to(orig_dtype)
