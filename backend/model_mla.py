"""
LuhGPT with MLA (Multi-Head Latent Attention)
"""

import torch
import torch.nn as nn
from transformer_block_mla import TransformerBlockMLA


class LuhGPT_MLA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_compressed: int,
        d_hidden: int,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlockMLA(d_model, n_heads, d_compressed, d_hidden)
            for _ in range(n_layers)
        ])
        
        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(token_ids)
            last_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids
