import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ViTConfig:
    """Vision Transformer Configuration"""
    image_size: int = 28  # MNIST images are 28x28
    patch_size: int = 4
    num_channels: int = 1  # MNIST is grayscale
    num_classes: int = 10  # 10 digit classes (0-9)
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    norm_eps: float = 1e-6


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size
        
        # Number of patches in one dimension
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # Conv layer to embed patches
        self.projection = nn.Conv2d(
            self.num_channels, 
            self.hidden_size, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Position embeddings for each patch
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.hidden_size)
        )
        
        # [CLS] token embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # (B, C, H, W) -> (B, hidden_size, H//patch_size, W//patch_size)
        x = self.projection(x)
        
        # (B, hidden_size, H', W') -> (B, hidden_size, num_patches)
        x = x.flatten(2)
        
        # (B, hidden_size, num_patches) -> (B, num_patches, hidden_size)
        x = x.transpose(1, 2)
        
        # Add cls token to beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.position_embeddings
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head Self Attention"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = config.attention_dropout
        
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.attention_dropout)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Linear projections and reshape
        q = self.query(x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.hidden_size)
        output = self.proj(attn_output)
        
        return output


class MLP(nn.Module):
    """MLP block for transformer"""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = MultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        # Self attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + residual
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer Model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer Encoder
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_layers)]
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        
        # Classifier head
        self.head = nn.Linear(config.hidden_size, config.num_classes)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)
        x = self.dropout(x)
        
        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Use CLS token for classification
        x = x[:, 0]
        
        # Classification head
        x = self.head(x)
        
        return x 