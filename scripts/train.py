#!/usr/bin/env python3
import click
import signal
import sys
import time
from click import Choice

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator, Field
from torchtext.data.metrics import bleu_score

import cleartext.utils as utils
from cleartext import PROJ_ROOT
from cleartext.data import WikiSmall
from cleartext.models import EncoderDecoder
from cleartext.utils.run import sample

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
MODELS_ROOT = PROJ_ROOT / 'models'


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
    # define register signal handler
    filename = f'gru-{max_examples}-{embed_dim}-{trg_vocab}-{rnn_units}-{attn_units}.pt'
    def signal_handler(_signal, _frame):
        finalize(device, model, src, trg, test_data, test_iter, criterion, NUM_SAMPLES, filename)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # set device
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
    vectors_path = PROJ_ROOT / 'vectors' / 'glove'
    glove = f'glove.6B.{embed_dim}d'
    vocab_args = {'min_freq': MIN_FREQ, 'vectors': glove, 'vectors_cache': vectors_path}
    src.build_vocab(train_data, **vocab_args)
    trg.build_vocab(train_data, max_size=trg_vocab, **vocab_args)
    # todo: check if files exists
    torch.save(src, MODELS_ROOT / f'src-{embed_dim}d.pt')
    torch.save(trg, MODELS_ROOT / f'trg-{embed_dim}d-{trg_vocab}')

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

    # start training cycle -- todo: update and use checkpoints
    print(f'Training model for {num_epochs} epochs')
    best_valid_loss = float('inf')
    valid_losses = [best_valid_loss]
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = utils.train_step(model, train_iter, criterion, optimizer)
        valid_loss = utils.eval_step(model, valid_iter, criterion)
        end_time = time.time()

        epoch_mins, epoch_secs = utils.format_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        utils.print_loss(train_loss, 'Train')
        utils.print_loss(valid_loss, 'Valid')

        # save everything
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, MODELS_ROOT / filename)
            with open(str(MODELS_ROOT / filename) + '.epoch', 'w') as f:
                f.write(f'Epoch {epoch} ({epoch_mins}m {epoch_secs}s)\n')
        elif valid_loss > valid_losses[-1] and valid_loss > valid_losses[-2]:
            finalize(device, model, src, trg, test_data, test_iter, criterion, NUM_SAMPLES, filename)
        valid_losses.append(valid_loss)

    finalize(device, model, src, trg, test_data, test_iter, criterion, NUM_SAMPLES, filename)


def finalize(device, model, src, trg, test_data, test_iter, criterion, num_examples, filename):
    def exit(_signal, _frame):
        sys.exit(0)
    signal.signal(signal.SIGINT, exit)
    print_diagnostics(device, model, criterion, src, trg, test_data, test_iter, num_examples, filename)


def print_diagnostics(device, model, criterion, src, trg, test_data, test_iter, num_examples, filename):
    # Compute and print test loss
    print('\nTesting model')
    test_loss = utils.eval_step(model, test_iter, criterion)
    utils.print_loss(test_loss, 'Test')

    # Generate and print sample translations
    ignore_tokens = [UNK_TOKEN, PAD_TOKEN]
    sources, targets, outputs = sample(device, model, src, trg, test_data, num_examples, SOS_TOKEN, EOS_TOKEN, ignore_tokens)
    source_outs = []
    target_outs = []
    output_outs = []
    for source, target, output in zip(sources, targets, outputs):
        source_out = '> ' + ' '.join(source)
        target_out = '= ' + ' '.join(target)
        output_out = '< ' + ' '.join(output)

        source_outs.append(source_out)
        target_outs.append(target_out)
        output_outs.append(output_out)

        print(source_out)
        print(target_out)
        print(output_out)

    # Compute and print BLEU score
    sources, targets, outputs = sample(device, model, src, trg, test_data, len(test_data), SOS_TOKEN, EOS_TOKEN, ignore_tokens)
    score = 0
    for target, output in zip(targets, outputs):
        # kill whitespace tokens, which crash bleu_score for some reason
        target = ' '.join(target).split()
        output = ' '.join(output).split()

        score += bleu_score([target], [[output]])
    score /= len(targets)
    print(f'Average BLEU score: {score:.3f}')

    with open(str(MODELS_ROOT / filename) + '.test', 'a') as f:
        f.write(f'Test loss: {test_loss}\n')
        f.write(f'BLEU score: {score}\n')
        for source_out, target_out, output_out in zip(source_outs, target_outs, output_outs):
            # todo: this isn't working right
            f.write(source_out + '\n')
            f.write(target_out + '\n')
            f.write(output_out + '\n')


if __name__ == '__main__':
    main()
