import math
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field, BucketIterator

from cleartext.data import WikiSmall
from cleartext.utils import format_time, train_step, eval_step, count_parameters, init_weights
from cleartext.models import EncoderDecoder

EOS_TOKEN = '<eos>'
SOS_TOKEN = '<sos>'
PAD_TOKEN = '<pad>'
CLIP = 1

# usage:
# >>> python -m test.py NUM_EPOCHS [NUM_EXAMPLES] [EMBED_DIM] [NUM_TOKENS] [VERBOSE]
if __name__ == '__main__':
    # parse arguments
    num_epochs = int(sys.argv[1])
    num_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    embed_dim = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    num_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    verbose = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # load data
    print('Preparing data')
    SOURCE = Field(tokenize='basic_english', init_token=SOS_TOKEN, eos_token=EOS_TOKEN, lower=True, pad_token=PAD_TOKEN)
    TARGET = Field(tokenize='basic_english', init_token=SOS_TOKEN, eos_token=EOS_TOKEN, lower=True, pad_token=PAD_TOKEN)
    train_data, valid_data, test_data = data = WikiSmall.splits(fields=(SOURCE, TARGET), num_examples=num_examples)
    print(f'Loaded {len(train_data)} training examples')

    # preprocess data
    # todo: error when actual vocabulary loaded is less than max size
    embed_vectors = f'glove.6B.{embed_dim}d'
    SOURCE.build_vocab(train_data, min_freq=2, vectors=embed_vectors, max_size=num_tokens)
    TARGET.build_vocab(train_data, min_freq=2, vectors=embed_vectors, max_size=num_tokens)
    train_iter, valid_iter, test_iter = BucketIterator.splits((train_data, valid_data, test_data),
                                                              batch_size=32,
                                                              device=device)
    print(f'Source vocabulary size: {len(SOURCE.vocab)}')
    print(f'Target vocabulary size: {len(TARGET.vocab)}')

    print('Building model')
    # model = build_model(device, TARGET.vocab.vectors)
    model = EncoderDecoder(device, SOURCE.vocab.vectors, TARGET.vocab.vectors, 100, 50, 0.2).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TARGET.vocab.stoi[PAD_TOKEN])
    trainable, total = count_parameters(model)
    print(f'Trainable parameters: {trainable} / Total parameters: {total}')

    print(f'Training model for {num_epochs} epochs')
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_step(model, train_iter, criterion, optimizer, verbose=verbose)
        valid_loss = eval_step(model, valid_iter, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = format_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTraining loss: {train_loss:.3f}\t| Training perplexity: {math.exp(train_loss):7.3f}')
        print(f'\tValidation Loss: {valid_loss:.3f}\t| Validation perplexity: {math.exp(valid_loss):7.3f}')

    print('Testing model')
    test_loss = eval_step(model, test_iter, criterion)
    print(f'Test loss: {test_loss:.3f} | Test perplexity: {math.exp(test_loss):7.3f}')
