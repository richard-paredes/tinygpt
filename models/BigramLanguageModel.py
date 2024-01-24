import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int):

        super().__init__()
        # Tokens read directly off logits for the next token from a lookup table
        # nn.Embedding is almost like a wrapper around a tensor that has the shape of vocab_size^2
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) 
    
    def forward(self, idx: int, targets=None):

        # Index and targets are both (Batch,Time) tensor of integers
        logits = self.token_embedding_table(idx) # (Batch,Time,Channel) tensor
        # Batch = block_size (4), Time = 8, C = vocab_size (65)
        # logits = scores for next character in the sequence

        # We now want to evaluate a loss function.
        # A good way to measure loss (i.e. the quality of predictions) is to use the negative-log likelihood loss
        # cross_entry expects B, C, T instead of B,T,C
        if targets is None:
            loss = None
        else:
            batch, time, channel = logits.shape
            logits = logits.view(batch*time, channel) # stretch out the array to be 2-dimensions
            targets = targets.view(batch*time)
            loss = F.cross_entropy(logits, targets) # Expect: -ln(1/vocab_size)

        return logits, loss

    # Takes in a (B,T) and generates a (B,T+1), (B,T+2), ... (B,T+max_new_tokens)
    def generate(self, idx: int, max_new_tokens: int):

        # Idx is the (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get predictions
            logits, loss = self(idx)
            # Focus on the last time step
            logits = logits[:, -1, :] # Becomes (B,C)
            # Apply softmax to get the probabilities
            probabilities = F.softmax(logits, dim=-1) # (B,C)
            # Sample from the distribution
            idx_next = torch.multinomial(probabilities, num_samples=1) # (B,1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx