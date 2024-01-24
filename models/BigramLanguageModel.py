import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        # Tokens read directly off logits for the next token from a lookup table
        # nn.Embedding is almost like a wrapper around a tensor that has the shape of vocab_size^2
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) 
    
    def forward(self, idx: int, targets):

        # Index and targets are both (Batch,Time) tensor of integers
        logits = self.token_embedding_table(idx) # (Batch,Time,Channel) tensor
        # Batch = block_size (4), Time = 8, C = vocab_size (65)
        return logits
