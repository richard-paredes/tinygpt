import torch
import torch.nn as nn
from torch.nn import functional as F

import hyperparameters
from models.BigramLanguageModel import BigramLanguageModel
from utils.util import read_file, get_distinct_chars, create_encoder, create_decoder, split_data, create_optimizer, train, evaluate_model, self_attention_inefficient

torch.manual_seed(1337)

def execute_training():
    text = read_file('input.txt')
    distinct_chars = get_distinct_chars(text)
    vocab_size = len(distinct_chars)
    encoder = create_encoder(distinct_chars)
    decoder = create_decoder(distinct_chars)
    train_data, val_data = split_data(encoder, text)
    model = BigramLanguageModel(vocab_size)
    model = model.to(hyperparameters.DEVICE)
    optimizer = create_optimizer(model, hyperparameters.LEARNING_RATE)
    print('Training')
    train(model, optimizer, train_data, val_data)
    print('Evaluating')
    evaluate_model(hyperparameters.DEVICE, model, hyperparameters.MAX_NEW_TOKENS, decoder)


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
    