import torch
import torch.nn as nn
from torch.nn import functional as F
from .Head import Head

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, device: str = 'cpu', num_embedding_dimensions: int = 32, block_size: int = 50):

        super().__init__()
        # Tokens read directly off logits for the next token from a lookup table
        # nn.Embedding is almost like a wrapper around a tensor that has the shape of vocab_size^2
        self.device = device
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, num_embedding_dimensions)
        # we want to encode the identity as well as the position of these tokens
        self.position_embedding_table = nn.Embedding(block_size, num_embedding_dimensions)
        self.self_attention_head = Head(head_size=num_embedding_dimensions,n_embed=num_embedding_dimensions,block_size=block_size)
        self.lang_modelling_head = nn.Linear(num_embedding_dimensions, vocab_size)
    
    def forward(self, idx: torch.Tensor, targets=torch.Tensor):
        B, T = idx.shape

        # Index and targets are both (Batch,Time) tensor of integers
        token_embedding = self.token_embedding_table(idx) # (Batch,Time,Channel) tensor
        # Batch = block_size (4), Time = 8, C = vocab_size (65)
        # logits = scores for next character in the sequence
        positional_embedding = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = token_embedding + positional_embedding # (B, T, C)
        x = self.self_attention_head(x)
        logits = self.lang_modelling_head(x) # (B,T,vocab_size)
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
    def generate(self, idx: torch.Tensor, max_new_tokens: int):

        # Idx is the (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            cropped_context = idx[:,-self.block_size:]
            # Get predictions
            logits, loss = self(cropped_context)
            # Focus on the last time step
            logits = logits[:, -1, :] # Becomes (B,C)
            # Apply softmax to get the probabilities
            probabilities = F.softmax(logits, dim=-1) # (B,C)
            # Sample from the distribution
            idx_next = torch.multinomial(probabilities, num_samples=1) # (B,1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx