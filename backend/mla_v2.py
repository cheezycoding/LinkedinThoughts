"""
Multi-Head Latent Attention v2 - Fixed Implementation

Fixes from v1:
1. Proper weight initialization (std=0.02)
2. Batched head computation (no Python loops)
3. Uses nn.Linear for proper initialization
4. Cleaner RoPE implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLAv2(nn.Module):
    """
    Multi-Head Latent Attention with KV compression.
    
    Key insight: Instead of caching separate K,V for each head,
    we cache a compressed latent c_kv and decompress on-the-fly.
    
    Memory savings: n_heads * d_head → d_compressed
    """
    
    def __init__(self, d_model: int, n_heads: int, d_compressed: int):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_compressed = d_compressed
        
        # Query projection (standard)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        
        # KV compression: d_model → d_compressed (shared across heads)
        self.W_down_kv = nn.Linear(d_model, d_compressed, bias=False)
        
        # KV decompression: d_compressed → d_model (for K and V separately)
        self.W_up_K = nn.Linear(d_compressed, d_model, bias=False)
        self.W_up_V = nn.Linear(d_compressed, d_model, bias=False)
        
        # Output projection
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights for stable training"""
        for module in [self.W_Q, self.W_down_kv, self.W_up_K, self.W_up_V, self.W_O]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = X.shape
        device = X.device
        
        # === Query path (standard) ===
        Q = self.W_Q(X)  # (batch, seq, d_model)
        
        # === KV path (compressed) ===
        # Compress to latent space
        c_kv = self.W_down_kv(X)  # (batch, seq, d_compressed)
        
        # Decompress to full K, V
        K = self.W_up_K(c_kv)  # (batch, seq, d_model)
        V = self.W_up_V(c_kv)  # (batch, seq, d_model)
        
        # === Reshape for multi-head attention ===
        # (batch, seq, d_model) → (batch, n_heads, seq, d_head)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # === Apply RoPE to Q and K ===
        Q = self.apply_rope(Q, seq_len, device)
        K = self.apply_rope(K, seq_len, device)
        
        # === Scaled dot-product attention ===
        # (batch, n_heads, seq, d_head) @ (batch, n_heads, d_head, seq)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply to values
        out = attn_weights @ V  # (batch, n_heads, seq, d_head)
        
        # === Reshape back ===
        # (batch, n_heads, seq, d_head) → (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        return self.W_O(out)
    
    def apply_rope(self, x: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Apply Rotary Position Embedding.
        
        Args:
            x: Tensor of shape (batch, n_heads, seq, d_head)
        """
        d_head = x.shape[-1]
        
        # Compute rotation angles
        theta = 10000.0 ** (-torch.arange(0, d_head, 2, device=device).float() / d_head)
        positions = torch.arange(seq_len, device=device).float()
        angles = torch.outer(positions, theta)  # (seq, d_head/2)
        
        # Expand for broadcasting: (1, 1, seq, d_head/2)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        
        # Split into even/odd dimensions
        x_even = x[..., 0::2]  # (batch, n_heads, seq, d_head/2)
        x_odd = x[..., 1::2]
        
        # Apply rotation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        
        # Interleave back
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.flatten(-2)  # (batch, n_heads, seq, d_head)
        
        return x_rotated


# Test
if __name__ == "__main__":
    batch, seq, d_model, n_heads, d_compressed = 2, 10, 64, 4, 16
    
    X = torch.randn(batch, seq, d_model)
    mla = MLAv2(d_model, n_heads, d_compressed)
    
    out = mla(X)
    print(f"Input shape:  {X.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {sum(p.numel() for p in mla.parameters()):,}")
    
    # Compare to standard attention
    from attention import MultiHeadAttention
    std_attn = MultiHeadAttention(d_model, n_heads)
    print(f"\nStandard MHA params: {sum(p.numel() for p in std_attn.parameters()):,}")
    print(f"MLA params:          {sum(p.numel() for p in mla.parameters()):,}")
    print(f"Compression ratio:   {sum(p.numel() for p in std_attn.parameters()) / sum(p.numel() for p in mla.parameters()):.2f}x")
