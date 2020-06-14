#!/usr/bin/env python3
import click
from click import Choice

import mlflow
import torch

import cleartext.utils as utils
from cleartext import PROJ_ROOT
from cleartext.data import WikiSmall
from cleartext.pipeline import Pipeline

# arbitrary choices
EOS_TOKEN = '<eos>'
SOS_TOKEN = '<sos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

# fixed choices
MIN_FREQ = 2
NUM_SAMPLES = 4
CLIP = 1
MODELS_ROOT = PROJ_ROOT / 'models'


@click.command()
@click.option('--num_epochs', '-e', default=10, type=int, help='Number of epochs')
@click.option('--max_examples', '-n', default=0, type=int, help='Max number of training examples')
@click.option('--batch_size', '-b', default=32, type=int, help='Batch size')
@click.option('--embed_dim', '-d', default='50', type=Choice(['50', '100', '200', '300']), help='Embedding dimension')
@click.option('--trg_vocab', '-t', default=0, type=int, help='Max target vocabulary size')
@click.option('--rnn_units', '-r', default=100, type=int, help='Number of RNN units')
@click.option('--attn_units', '-a', default=100, type=int, help='Number of attention units')
@click.option('--dropout', '-p', default=0.3, type=float, help='Dropout probability')
@click.option('--alpha', default=0.5, type=float, help='Beam search regularization')
def main(num_epochs: int, max_examples: int, batch_size: int,
         embed_dim: str, trg_vocab: int, rnn_units: int, attn_units: int,
         dropout: float, alpha: float) -> None:
    # initialize pipeline
    pipeline = Pipeline()
    print(f'Using {pipeline.device}')
    print()

    # load data
    print('Loading data')
    max_examples = max_examples if max_examples else None
    train_len, _, _ = pipeline.load_data(WikiSmall, max_examples)
    print(f'Loaded {train_len} training examples')
    print()

    # load embeddings
    print(f'Loading {embed_dim}-dimensional GloVe vectors')
    trg_vocab = trg_vocab if trg_vocab else None
    src_vocab_size, trg_vocab_size = pipeline.load_vectors(int(embed_dim), trg_vocab)
    print(f'Source vocabulary size: {src_vocab_size}')
    print(f'Target vocabulary size: {trg_vocab_size}')
    print()

    # prepare data
    pipeline.prepare_data(batch_size)

    # build model and prepare optimizer and loss
    print('Building model')
    trainable, total = pipeline.build_model(rnn_units, attn_units, dropout)
    print(f'Trainable parameters: {trainable} | Total parameters: {total}')
    print()

    # run training loop
    print(f'Training model for {num_epochs} epochs')
    pipeline.train(num_epochs)

    # reload last checkpoint (without losing dataset)
    pl_dict = torch.load(pipeline.root / f'model{pipeline.model_index:02}.pt', map_location=pipeline.device)
    pipeline.model.load_state_dict(pl_dict['model_state_dict'])
    pipeline.model.to(pipeline.device)

    # evaluate and save/print results
    print('\nEvaluating model')
    train_loss, valid_loss, test_loss, bleu = pipeline.evaluate(alpha=alpha)
    mlflow.log_metrics({
        'train_loss_final': train_loss,
        'valid_loss_final': valid_loss,
        'test_loss_final': test_loss,
        'bleu_score': bleu
    })
    utils.print_loss(train_loss, 'Train')
    utils.print_loss(valid_loss, 'Valid')
    utils.print_loss(test_loss, 'Test')
    print(f'\tBLEU score:\t{bleu:.3f}\t')

    # Generate and print samples
    examples = pipeline.test_data[:NUM_SAMPLES]
    sources, targets = zip(*((e.src, e.trg) for e in examples))
    outputs = [pipeline.beam_search(s, 10, 30) for s in sources]
    source_print = []
    target_print = []
    output_print = []
    for i in range(len(examples)):
        source_out = '> ' + ' '.join(sources[i])
        target_out = '= ' + ' '.join(targets[i])
        output_out = '< ' + ' '.join(outputs[i])

        source_print.append(source_out)
        target_print.append(target_out)
        output_print.append(output_out)

        print(source_out)
        print(target_out)
        print(output_out)

    # save sample outputs
    path = str(pipeline.model_path) + '.txt'
    with open(path, 'w') as f:
        for source_out, target_out, output_out in zip(source_print, target_print, output_print):
            f.write(source_out + '\n')
            f.write(target_out + '\n')
            f.write(output_out + '\n')


if __name__ == '__main__':
    main()
