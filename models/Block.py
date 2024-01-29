import torch
import torch.nn as nn

import hyperparameters
from models.FeedForward import FeedForward
from models.MutliHead import MultiHeadAttention

class Block(nn.Module):
    """Transformer block: communication followed by computation """

    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, head_size=n_embed)
        self.ffwd = FeedForward(n_embed)
        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)
    
    def forward(self, x: torch.Tensor):
        # x = self.sa(x)
        # x = self.ffwd(x)

        # x = x + self.sa(x)
        # x = x + self.ffwd(x)

        x = x + self.sa(self.layer_norm_1(x))
        x = x + self.ffwd(self.layer_norm_2(x))

        return x