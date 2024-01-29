import torch
import torch.nn as nn

import hyperparameters

class FeedForward(nn.Module):
    ''' a simple linear layer followed by a non-linearity '''
    def __init__(self, num_embedding_dimensions:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embedding_dimensions, 4 * num_embedding_dimensions),
            nn.ReLU(),
            nn.Linear(4* num_embedding_dimensions, num_embedding_dimensions),
            nn.Dropout(hyperparameters.DROPOUT)
        )
    
    def forward(self, x: torch.Tensor):
        return self.net(x)
