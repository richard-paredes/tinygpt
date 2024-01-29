import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size: int, n_embed: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinites")
        wei = q @k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) => (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # decoder block
        wei = F.softmax(wei, dim=-1)
        # Perform weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out
