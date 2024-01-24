import torch
from models.BigramLanguageModel import BigramLanguageModel

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Examine the entire character length of the dataset
print("Length of the dataset in characters: ", len(text))

# Examine the first 1K characters
print(text[:1000])

# Retrieve all the distinct characters occurring in this text
distinct_characters = sorted(list(set(text)))
vocabulary_size = len(distinct_characters)
print(''.join(distinct_characters))
print("Total number of distinct characters: ", vocabulary_size)

# Need to tokenize the input
# Tokenize: Convert the raw charater input to some sequence of numbers based on some categorization strategy
# There is a tradeoff between the tokenizer strategies done. There may be a large vocabulary with small sequence of integers
# Or there may be a small vocabulary with a large sequence of integers

# Our strategy: Convert the individual characters to integers.
characterToInteger = { ch:i for i,ch in enumerate(distinct_characters) }
integerToCharacter = { i:ch for i,ch in enumerate(distinct_characters) }
encode = lambda inputString: [characterToInteger[c] for c in inputString] # encoder: Take in a string, output a list of integers
decode = lambda token: ''.join([integerToCharacter[i] for i in token]) # decoder: take a list of integers, output a string

print(encode("Hello there")) # [20, 43, 50, 50, 53, 1, 58, 46, 43, 56, 43]
print(decode(encode("Hii there"))) # Hii there

# Different strategies for coming up with tokenizer strategies.
# Example: SentencePiece by Google: github.com/google/sentencepiece

# Now encode the dataset and store into a torch.tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) 

# Let's split the data into train and validation sets
n = int(0.9*len(data))
training_data = data[:n]
validation_data = data[n:]

# When training the transformer, we chunk and feed the model batches of data
block_size = 8
sample_chunk = training_data[:block_size+1]
print(sample_chunk)

# Note: We want the transformer to be useful for seeing "contexts" from as little as one token input, all the way to block_size tokens
# This code sample demonstrates the target value, given the context of the inputs so far (which is a sample of the text)
x = training_data[:block_size] # inputs to the transformer, all block_size characters
y = training_data[1:block_size+1] # next blocksize characters, the targets for each position in the input
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context}, the target is: {target}")

# Note: we want to process batches of the dataset concurrently for efficiency reasons. GPUs excel at concurrent processing
torch.manual_seed(1337)
batch_size = 4 # Decides how many independent sequences will we process in parallel
block_size = 8 # Decides what the maximum context length is for predictions

def get_batch(split: str):
    # We generate a small batch of data of inputs X and targets Y
    data = training_data if split == 'train' else validation_data
    # This will get random offsets within the training set
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Torch.stack will stack the 1-dimensional tensors of tokens as rows in a 4x8 tensor
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # The result is 32 independent samples packed into a single batch, within X and the corresponding outputs in Y
    return x, y

xb, yb = get_batch('train')
print('----')
print(f'Inputs: {xb.shape} | {xb}')
print(f'Targets: {yb.shape} | {yb}')
print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"When input is {context.tolist()}, the target is: {target}")

model = BigramLanguageModel(vocabulary_size)
outputs = model(xb, yb)
print(outputs.shape)