import torch
import torch.nn as nn
from torch.nn import functional as F
import hyperparameters

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(hyperparameters.NUM_EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.query = nn.Linear(hyperparameters.NUM_EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.value = nn.Linear(hyperparameters.NUM_EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(hyperparameters.BLOCK_SIZE, hyperparameters.BLOCK_SIZE)))
        self.dropout = nn.Dropout(hyperparameters.DROPOUT)
    
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
