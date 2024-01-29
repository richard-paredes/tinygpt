import torch
import torch.nn as nn

import hyperparameters
from models.Head import Head

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # self.proj = nn.Linear(hyperparameters.NUM_EMBEDDING_DIMENSIONS, hyperparameters.NUM_EMBEDDING_DIMENSIONS)
        self.proj = nn.Linear(head_size * num_heads, hyperparameters.NUM_EMBEDDING_DIMENSIONS)
        self.dropout = nn.Dropout(hyperparameters.DROPOUT)

    
    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out