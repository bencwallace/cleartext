import math
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator

import cleartext.utils as utils
from cleartext.data import WikiSmall
from cleartext.models import EncoderDecoder


# arbitrary choices
EOS_TOKEN = '<eos>'
SOS_TOKEN = '<sos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

# fixed choices
BATCH_SIZE = 32
MIN_FREQ = 2
NUM_SAMPLES = 4

# defaults for CLI arguments
NUM_TOKENS = 10_000
EMBED_DIM = 50
MAX_EXAMPLES = 30_000
RNN_UNITS = 100
ATTN_UNITS = 50
DROPOUT = 0.2
CLIP = 1

# usage:
# >>> python -m train NUM_EPOCHS [MAX_EXAMPLES] [EMBED_DIM] [NUM_TOKENS] [VERBOSE]
if __name__ == '__main__':
    # parse arguments
    num_epochs = int(sys.argv[1])
    max_examples = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_EXAMPLES
    embed_dim = int(sys.argv[3]) if len(sys.argv) > 3 else EMBED_DIM
    num_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else NUM_TOKENS
    verbose = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # load data
    print('Loading data')
    field_args = {'tokenize': 'spacy',
                  'tokenizer_language': 'en_core_web_sm',
                  'init_token': SOS_TOKEN,
                  'eos_token': EOS_TOKEN,
                  'lower': True,
                  'pad_token': PAD_TOKEN,
                  'unk_token': UNK_TOKEN,
                  'preprocessing': utils.preprocess}
    FIELD = Field(**field_args)
    train_data, valid_data, test_data = data = WikiSmall.splits(fields=(FIELD, FIELD), max_examples=max_examples)
    print(f'Loaded {len(train_data)} training examples')

    # load embeddings and prepare data
    print(f'Loading {embed_dim}-dimensional GloVe vectors')
    proj_root = utils.get_proj_root()
    vectors_path = proj_root / '.vector_cache'
    embed_vectors = f'glove.6B.{embed_dim}d'
    # todo: fix error when actual vocabulary loaded is less than max size
    FIELD.build_vocab(train_data, min_freq=MIN_FREQ, vectors=embed_vectors, max_size=num_tokens, vectors_cache=vectors_path)
    iters = BucketIterator.splits((train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)
    train_iter, valid_iter, test_iter = iters
    print(f'Source vocabulary size: {len(FIELD.vocab)}')
    print(f'Target vocabulary size: {len(FIELD.vocab)}')

    print('Building model')
    model = EncoderDecoder(device, FIELD.vocab.vectors, FIELD.vocab.vectors, RNN_UNITS, ATTN_UNITS, DROPOUT).to(device)
    model.apply(utils.init_weights)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=FIELD.vocab.stoi[PAD_TOKEN])
    trainable, total = utils.count_parameters(model)
    print(f'Trainable parameters: {trainable} | Total parameters: {total}')

    print(f'Training model for {num_epochs} epochs')
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = utils.train_step(model, train_iter, criterion, optimizer, verbose=verbose)
        valid_loss = utils.eval_step(model, valid_iter, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = utils.format_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTraining loss: {train_loss:.3f}\t| Training perplexity: {math.exp(train_loss):7.3f}')
        print(f'\tValidation Loss: {valid_loss:.3f}\t| Validation perplexity: {math.exp(valid_loss):7.3f}')

    print('Testing model')
    test_loss = utils.eval_step(model, test_iter, criterion)
    print(f'Test loss: {test_loss:.3f} | Test perplexity: {math.exp(test_loss):7.3f}')

    print('Model sample')
    ignore = list(map(lambda s: FIELD.vocab.stoi[s], [UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN]))
    source, output, target = utils.sample(model, test_iter, ignore)
    for i in torch.randint(0, len(source), (NUM_SAMPLES,)):
        print('> ', utils.seq_to_sentence(source.T[i].tolist(), FIELD.vocab, [PAD_TOKEN]))
        print('= ', utils.seq_to_sentence(target.T[i].tolist(), FIELD.vocab, [PAD_TOKEN]))
        print('< ', utils.seq_to_sentence(output.T[i].tolist(), FIELD.vocab, [PAD_TOKEN]))
        print()
