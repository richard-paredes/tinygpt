import torch
import torch.nn as nn

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
              block_size: int, 
              batch_size: int, 
              split: str,
              device: str):
    # We generate a small batch of data of inputs X and targets Y
    data = training_data if split == 'train' else validation_data
    # This will get random offsets within the training set
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Torch.stack will stack the 1-dimensional tensors of tokens as rows in a 4x8 tensor
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    # The result is 32 independent samples packed into a single batch, within X and the corresponding outputs in Y
    return x, y

# Best practice: Indicate to PyTorch when we will not execute back-propagation
@torch.no_grad() # What this means: Everything that happens inside this function, we will not call .backward() on (means more memory efficient as no intermediate vars are stored)
def estimate_loss(model: nn.Module, 
                device: str,
                training_data: torch.Tensor, 
                validation_data: torch.Tensor, 
                block_size: int,
                batch_size: int, 
                eval_iters: int):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(training_data, validation_data, block_size, batch_size, split, device)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def create_optimizer(model: nn.Module, learning_rate: int):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)

def train(model: nn.Module, 
          device: str,
          optimizer: torch.optim.Optimizer,
          training_data: torch.Tensor, 
          validation_data: torch.Tensor, 
          block_size: int,
          batch_size: int,
          max_iters: int, 
          eval_iters: int,
          eval_interval: int):
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, device, training_data, validation_data, block_size, batch_size, eval_iters)
            print(f"Step: {iter}: training loss - {losses['train']:.4f}, validation loss - {losses['val']:.4f}")
    
    # Sample
    xb, yb = get_batch(training_data, validation_data, block_size, batch_size, 'train', device)

    # Eval loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

def evaluate_model(device: str, model: nn.Module, max_new_tokens: int, decoder):
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decoder(model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()))