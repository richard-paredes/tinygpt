import torch
import torch.nn as nn
from torch.nn import functional as F
import hyperparameters


def read_file(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def get_distinct_chars(text: str):
    return sorted(list(set(text)))

def create_encoder(chars):
    stoi = { ch:i for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    return encode

def create_decoder(chars):
    itos = { i:ch for i,ch in enumerate(chars) }
    decode = lambda l: ''.join([itos[i] for i in l])
    return decode

def split_data(encoder, dataset: str):
    data = torch.tensor(encoder(dataset), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(training_data: torch.Tensor, 
              validation_data: torch.Tensor, 
              split: str):
    # We generate a small batch of data of inputs X and targets Y
    data = training_data if split == 'train' else validation_data
    # This will get random offsets within the training set
    ix = torch.randint(len(data) - hyperparameters.BLOCK_SIZE, (hyperparameters.BATCH_SIZE,))

    # Torch.stack will stack the 1-dimensional tensors of tokens as rows in a 4x8 tensor
    x = torch.stack([data[i:i+hyperparameters.BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+hyperparameters.BLOCK_SIZE+1] for i in ix])
    x, y = x.to(hyperparameters.DEVICE), y.to(hyperparameters.DEVICE)
    # The result is 32 independent samples packed into a single batch, within X and the corresponding outputs in Y
    return x, y

# Best practice: Indicate to PyTorch when we will not execute back-propagation
@torch.no_grad() # What this means: Everything that happens inside this function, we will not call .backward() on (means more memory efficient as no intermediate vars are stored)
def estimate_loss(model: nn.Module, 
                training_data: torch.Tensor, 
                validation_data: torch.Tensor):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(hyperparameters.EVALUATION_ITERATIONS)
        for k in range(hyperparameters.EVALUATION_ITERATIONS):
            X, Y = get_batch(training_data, validation_data, split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def create_optimizer(model: nn.Module, learning_rate: int):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)

def train(model: nn.Module, 
          optimizer: torch.optim.Optimizer,
          training_data: torch.Tensor, 
          validation_data: torch.Tensor):
    for iter in range(hyperparameters.MAX_ITERATIONS):

        # every once in a while evaluate the loss on train and val sets
        if iter % hyperparameters.EVALUATION_INTERVAL == 0 or iter == hyperparameters.MAX_ITERATIONS - 1:
            losses = estimate_loss(model,training_data,validation_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(training_data, validation_data, 'train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Sample
        xb, yb = get_batch(training_data, validation_data,'train')

        # Eval loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

def evaluate_model(device: str, model: nn.Module, max_new_tokens: int, decoder):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decoder(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()))

# Inefficient because we are doing nest loops. We can instead perform the same logic using matrix multiplication
def self_attention_inefficient(B:int, T:int, C:int):
    x = torch.randn(B,T,C)
    x_bag_of_words = torch.zeros((B,T,C)) # just means we are averaging
    for b in range(B):
        for t in range(T):
            prev_tokens = x[b,:t+1] # shape becomes (t, C)
            x_bag_of_words[b,t] = torch.mean(prev_tokens, 0) # average out the time

def self_attention_matrix_mult(B:int, T:int, C:int):
    x = torch.randn(B,T,C)
    weights = torch.tril(torch.ones(T,T))
    weights = weights / weights.sum(1, keepdim=True)
    xbow = weights @ x # (T, T) @ (B,T,C) --> PyTorch modifies to match dimensions (B, T, T) @ (B, T, C) => (B, T, C)
    return xbow

def self_attention_softmax(B:int, T:int, C:int):
    x = torch.randn(B,T,C)
    tril = torch.tril(torch.ones(T,T)) # lower triangular 1's
    # e.g. 
    # [
    #   [1, 0, 0]
    #   [1, 1, 0]
    #   [1, 1, 1]
    # ]

    wei = torch.zeros((T,T)) # Means: Interaction affinity, how much of each token from the past do we want to aggregate
    wei = wei.masked_fill(tril == 0, float('-inf')) # All elements where tril is zero will become negative inf. Means: tokens from the future cannot communicate with the past -- will not aggregate anything from those tokens
    # e.g. 
    # [
    #   [0, -inf, -inf]
    #   [0, 0, -inf]
    #   [0, 0, 0]
    # ]

    wei = F.softmax(wei, dim=-1) # exponentiates every element and divide by the sum.
     # e.g. 
    # [
    #   [1.0, 0, 0]
    #   [0.5, 0.5, 0]
    #   [0.33, 0.33, 0.33]
    # ]


    xbow = wei @ x

def trivial_matrix_multiplication_example():
    torch.manual_seed(42)
    a = torch.tril(torch.ones(3,3)) # Will zero out elements in a multi-dimensional array so it is triangular
    # e.g. 
    # [
    #   [1, 0, 0]
    #   [1, 1, 0]
    #   [1, 1, 1]
    # ]

    # This will let us perform an average
    a = a / torch.sum(a, 1, keepdim=True) # distributes the 1 value equally among elements that are non-zero
    # e.g. 
    # [
    #   [1, 0, 0]
    #   [0.5, 0.5, 0]
    #   [0.33, 0.33, 0.33]
    # ] 
    
    b = torch.randint(0,10,(3,2)).float()
    c = a @ b
    print('a=', '\n', a)
    print('--')
    print('b=', '\n', b)
    print('--')
    print('c=','\n',c)

# Note: Attention is a communication mechanism. Nodes in a directed graph, and every node has a vector of information.
# It can aggergate information via a weighted sum of all the nodes that point to it (in a data dependent manner)
# Nodes = # tokens
# Think of the graph as like:
    # 1 block_size = 8 tokens, each token is a node
    # The first node only has it pointed to itself (cannot see the future)
    # The eight node would have all other nodes pointed to it, as well as pointing to itself
# There is no notion of space here. Attention acts over a set of vectors, hence why we need to positionally encode tokens.
    
def self_attention_head(B: torch.Tensor, T: torch.Tensor, C: torch.Tensor, head_size: int):
    x = torch.randn(B,T,C)
    key = nn.Linaer(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Liner(C, head_size, bias=False)

    k, q = key(x), query(x) # (B, T, 16)
    wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) => (B, T, T)
    tril = torch.tril(torch.ones(T,T))
    wei = wei.masked_fill(tril == 0, float('-inf')) # This line prevents future nodes from communication with the previous ones
    wei = F.softmax(wei, dim=-1)

    v = value(x)
    out = wei @ v
    # out = wei @ x

    print(out.shape)