import torch
import torch.nn as nn
from torch.nn import functional as F
from models.BigramLanguageModel import BigramLanguageModel
from utils.util import read_file, get_distinct_chars, create_encoder, create_decoder, split_data, create_optimizer, train, evaluate_model, self_attention_inefficient, trivial_matrix_multiplication_example

# --- begin Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERATIONS = 5000
EVALUATION_ITERATIONS = 500
EVALUATION_INTERVAL = 200
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_NEW_TOKENS = 500
# --- end Hyperparameters

torch.manual_seed(1337)

def execute_training():
    text = read_file('input.txt')
    distinct_chars = get_distinct_chars(text)
    vocab_size = len(distinct_chars)
    encoder = create_encoder(distinct_chars)
    decoder = create_decoder(distinct_chars)
    train_data, val_data = split_data(encoder, text)
    model = BigramLanguageModel(vocab_size, DEVICE, BATCH_SIZE, BLOCK_SIZE)
    model = model.to(DEVICE)
    optimizer = create_optimizer(model, LEARNING_RATE)
    print('Training')
    train(model, DEVICE, optimizer, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, MAX_ITERATIONS, EVALUATION_ITERATIONS, EVALUATION_INTERVAL)
    print('Evaluating')
    evaluate_model(DEVICE, model, MAX_NEW_TOKENS, decoder)


def execute_self_attention():
    B,T,C = 4,8,32 # batch, time, channels
    x = torch.randn(B,T,C)
    tril = torch.tril(torch.ones(T,T))
    wei = torch.zeros((T,T))
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    out = wei @ x

    self_attention_inefficient(B,T,C)


execute_training()
    