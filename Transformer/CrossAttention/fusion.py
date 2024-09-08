import torch
import torch.nn as nn
from typing import Optional, List


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: int = 0.0,
    ):
        super(CrossAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        torch._assert(self.head_dim * num_heads == dim,
                      'Embedding dimension must be divisable by num_heads')
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    # cross attention between x1 and x2
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        torch._assert(x1.shape == x2.shape and len(x1.shape) == 3,
                      'Cross attention dimensions should be equal')
        B, N, D = x1.shape
        qkv1 = self.qkv_proj(x1).reshape(
            B, N, 3, self.num_heads, self.head_dim)
        qkv2 = self.qkv_proj(x2).reshape(
            B, N, 3, self.num_heads, self.head_dim)
        q1 = qkv1[:, :, 0].permute(0, 2, 1, 3)  # B x num_heads x N x head_dim
        k2 = qkv2[:, :, 1].permute(0, 2, 1, 3)
        v2 = qkv2[:, :, 2].permute(0, 2, 1, 3)
        attn_weights = torch.matmul(
            q1, k2.transpose(-2, -1)) * self.scale  # B x num_heads x N x N
        if mask:
            attn_weights = attn_weights.masked_fill(mask == 0, float('int'))
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v2)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, D)
        attn_output = self.out_dropout(attn_output)
        return attn_output


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        output_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: int = 0.0
    ):
        super(CrossAttentionFusion, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(dim=dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])

    def forward(
        self,
        features: List,
        mask: Optional[torch.Tensor] = None
    ):
        x = features[0]
        for i in range(1, len(features)):
            for layer in self.layers:
                x = layer(x, features[i])
        return x
