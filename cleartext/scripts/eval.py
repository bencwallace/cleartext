#!/usr/bin/env python3
import click
import warnings

from .. import PROJ_ROOT, utils
from cleartext.data import WikiSmall, WikiLarge
from cleartext.pipeline import Pipeline


@click.command()
@click.argument('name', default='pl', required=False, type=str)
@click.argument('dataset', default='wikilarge', type=str)
@click.option('-b', '--beam_size', default=10, type=int, help='Beam size')
@click.option('-l', '--max_len', required=False, type=str, help='Max length')
@click.option('-a', '--alpha', default=0.5, type=float, help='Beam search regularization')
@click.option('--batch_size', '-b', default=64, type=int, help='Batch size')
def main(name: str, dataset: str, beam_size: int, max_len: str, alpha: float, batch_size: int):
    warnings.filterwarnings('ignore', category=DeprecationWarning, lineno=6)

    # parse/validate arguments
    if dataset.lower() == 'wikismall':
        dataset = WikiSmall
    elif dataset.lower() == 'wikilarge':
        dataset = WikiLarge
    else:
        raise ValueError(f'Unknown dataset "{dataset}"')

    # deserialize pipeline
    MODELS_ROOT = PROJ_ROOT / 'models'
    path = MODELS_ROOT / name
    print(f'Loading {name}')
    pipeline = Pipeline.deserialize(path)
    print()

    # load data (only validation and test sets)
    print(f'Loading {dataset.__name__} data')
    _, eval_len, test_len = pipeline.load_data(dataset, 1000)
    print(f'Loaded {eval_len} evaluation examples and {test_len} test examples')
    pipeline.prepare_data(batch_size)
    print()

    # evaluate and save/print results
    print('\nEvaluating model')
    _, _, _, bleu = pipeline.evaluate(beam_size, max_len, alpha)
    print(f'\tBLEU score:\t{bleu:.3f}\t')


if __name__ == '__main__':
    main()
