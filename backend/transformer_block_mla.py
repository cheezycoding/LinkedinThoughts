"""
Transformer Block with MLA
"""

import torch
import torch.nn as nn
from mla_v2 import MLAv2
from FFN import FFN


class TransformerBlockMLA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_compressed: int, d_hidden: int):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MLAv2(d_model, n_heads, d_compressed)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
