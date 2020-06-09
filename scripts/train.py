#!/usr/bin/env python3
import click
import math
import signal
import sys
import time
from click import Choice

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torchtext.data import BucketIterator, Field, Iterator
from torchtext.data.metrics import bleu_score

import cleartext.utils as utils
from cleartext import PROJ_ROOT
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
CLIP = 1


@click.command()
@click.option('--num_epochs', '-e', default=10, type=int, help='Number of epochs')
@click.option('--max_examples', '-n', default=50_000, type=int, help='Max number of training examples')
@click.option('--embed_dim', '-d', default='50', type=Choice(['50', '100', '200', '300']), help='Embedding dimension')
@click.option('--trg_vocab', '-t', default=2_000, type=int, help='Max target vocabulary size')
@click.option('--rnn_units', '-r', default=100, type=int, help='Number of RNN units')
@click.option('--attn_units', '-a', default=100, type=int, help='Number of attention units')
@click.option('--dropout', '-p', default=0.3, type=float, help='Dropout probability')
def main(num_epochs: int, max_examples: int,
         embed_dim: str, trg_vocab: int,
         rnn_units: int, attn_units: int,
         dropout: float) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # load data
    print('Loading data')
    field_args = {
        'tokenize': 'spacy', 'tokenizer_language': 'en_core_web_sm',
        'init_token': SOS_TOKEN, 'eos_token': EOS_TOKEN, 'pad_token': PAD_TOKEN, 'unk_token': UNK_TOKEN,
        'lower': True, 'preprocessing': utils.preprocess
    }
    src = Field(**field_args)
    trg = Field(**field_args)
    train_data, valid_data, test_data = WikiSmall.splits(fields=(src, trg), max_examples=max_examples)
    print(f'Loaded {len(train_data)} training examples')

    # load embeddings and build vocabulary
    print(f'Loading {embed_dim}-dimensional GloVe vectors')
    vectors_path = PROJ_ROOT / '.vector_cache'
    glove = f'glove.6B.{embed_dim}d'
    vocab_args = {'min_freq': MIN_FREQ, 'vectors': glove, 'vectors_cache': vectors_path}
    src.build_vocab(train_data, **vocab_args)
    trg.build_vocab(train_data, max_size=trg_vocab, **vocab_args)

    # batch data
    iters = BucketIterator.splits((train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)
    train_iter, valid_iter, test_iter = iters
    print(f'Source vocabulary size: {len(src.vocab)}')
    print(f'Target vocabulary size: {len(trg.vocab)}')

    # build model
    print('Building model')
    model = EncoderDecoder(device, src.vocab.vectors, trg.vocab.vectors, rnn_units, attn_units, dropout).to(device)
    model.apply(utils.init_weights)
    trainable, total = utils.count_parameters(model)
    print(f'Trainable parameters: {trainable} | Total parameters: {total}')

    # prepare optimizer and loss
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=trg.vocab.stoi[PAD_TOKEN])

    # define register signal handler
    def signal_handler(_signal, _frame):
        finalize(device, model, src, trg, test_data, test_iter, criterion, NUM_SAMPLES)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # start training cycle -- todo: update and use checkpoints
    print(f'Training model for {num_epochs} epochs')
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = utils.train_step(model, train_iter, criterion, optimizer)
        valid_loss = utils.eval_step(model, valid_iter, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = utils.format_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        utils.print_loss(train_loss, 'Train')
        utils.print_loss(valid_loss, 'Valid')

    # run tests
    test(model, src, trg, test_iter, criterion)
    print_samples(device, model, src, trg, test_data, NUM_SAMPLES)


def finalize(device, model, src, trg, test_data, test_iter, criterion, num_examples):
    test(model, src, trg, test_iter, criterion)
    print_samples(device, model, src, trg, test_data, num_examples)


def print_samples(device, model, src, trg, test_data, num_examples):
    sources, targets, outputs = sample(device, model, src, trg, test_data, num_examples)
    for source, target, output in zip(sources, targets, outputs):
        print('> ', ' '.join(source))
        print('= ', ' '.join(target))
        print('< ', ' '.join(output))

    # compute bleu score
    sources, targets, outputs = sample(device, model, src, trg, test_data, len(test_data))
    score = 0
    for target, output in zip(targets, outputs):
        # space tokens kill bleu score for some reason
        target = ' '.join(target).split()
        output = ' '.join(output).split()
        score += bleu_score([target], [[output]])
    score /= len(targets)
    print(f'Avg BLEU score: {score:.3f}')


def sample(device, model, src, trg, test_data, num_examples):
    model.eval()
    # todo: randomize examples
    sources, targets = zip(*((example.src, example.trg) for example in test_data[:num_examples]))

    # run model with dummy target
    source_tensor = src.process(sources).to(device)
    dummy = torch.zeros(source_tensor.shape, dtype=int, device=device)
    dummy.fill_(trg.vocab[SOS_TOKEN])

    # select most likely tokens (ignoring non-word tokens)
    output = model(source_tensor, dummy, 0)[1:]
    # todo: vectorize masking
    for i in map(trg.vocab.stoi.get, [SOS_TOKEN, UNK_TOKEN, PAD_TOKEN]):
        output.data[:, :, i] = 0
    output = output.argmax(dim=2)

    # trim past eos token and denumericalize
    output = output.T.tolist()
    trimmed = []
    for out in output:
        try:
            eos_index = out.index(trg.vocab[EOS_TOKEN])
        except ValueError:
            eos_index = len(out)
        out = out[:eos_index]
        out = list(map(lambda i: trg.vocab.itos[i], out))
        trimmed.append(out)

    return sources, targets, trimmed


def test(model: Module, src: Field, trg: Field, test_iter: Iterator, criterion: Module):
    print('\nTesting model')
    test_loss = utils.eval_step(model, test_iter, criterion)
    # todo: use print_loss
    print(f'Test loss: {test_loss:.3f} | Test perplexity: {math.exp(test_loss):7.3f}')


if __name__ == '__main__':
    main()
