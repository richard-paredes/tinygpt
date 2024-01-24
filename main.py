import torch
import torch.nn as nn
from torch.nn import functional as F
from models.BigramLanguageModel import BigramLanguageModel
from utils.util import read_file, get_distinct_chars, create_encoder, create_decoder, split_data, create_optimizer, train, evaluate_model

# --- begin Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERATIONS = 3000
EVALUATION_ITERATIONS = 200
EVALUATION_INTERVAL = 300
LEARNING_RATE = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_NEW_TOKENS = 500
# --- end Hyperparameters

torch.manual_seed(1337)

text = read_file('input.txt')
distinct_chars = get_distinct_chars(text)
vocab_size = len(distinct_chars)
encoder = create_encoder(distinct_chars)
decoder = create_decoder(distinct_chars)
train_data, val_data = split_data(encoder, text)
model = BigramLanguageModel(vocab_size)
model = model.to(DEVICE)
optimizer = create_optimizer(model, LEARNING_RATE)
train(model, DEVICE, optimizer, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, MAX_ITERATIONS, EVALUATION_ITERATIONS, EVALUATION_INTERVAL)
evaluate_model(DEVICE, model, MAX_NEW_TOKENS, decoder)