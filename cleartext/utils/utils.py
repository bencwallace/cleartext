import math
import re
import unicodedata
from typing import Iterator, List, Tuple

import torch
from torch import nn as nn
from torch.nn import Module
from torchtext.vocab import Vocab


def count_parameters(model: Module) -> Tuple[int, int]:
    """Count the number of parameters in a model.

    :param model: Module
        A PyTorch model.
    :return: Tuple[int, int]
        The number trainable parameters and the total number of parameters, respectively.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, trainable + fixed


def format_time(elapsed: float) -> Tuple[int, int]:
    """Convert a quantity of time, in seconds, to minutes and seconds.

    :param elapsed: float
        A quantity of time, in seconds.
    :return: Tuple[int, int]
        The number of minutes in `elapsed` and the remaining number of seconds, respectively.
    """
    mins = int(elapsed / 60)
    secs = int(elapsed - (mins * 60))
    return mins, secs


def init_weights_(model: Module) -> None:
    """Initialize a model's weights.

    Initializes weights containing the word "weight" in their name using Xavier/Glorot initialization and initializes
    the remaining weights, assumed to be biases, to zero. Performs initialization in-place.

    :param model: Module
        A PyTorch model, whose weights are to be initialized.
    :return: None
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param.data)
        else:
            nn.init.constant_(param.data, 0)


# todo: use tokenizer to remove space around punctuation, etc.
def seq_to_sentence(seq: Iterator[int], vocab: Vocab, ignore: Iterator[int]) -> str:
    """Convert a sequence of integers to a string of (space-separated) words according to a vocabulary.

    :param seq: Iterator[int]
        A sequence of integers (tokens) to be converted.
    :param vocab: Vocab
        A Torchtext Vocab object containing a mapping from integers to strings (words).
    :param ignore: Iterator[int]
        A sequence of integers representing "special tokens" to ignore (convert as blanks).
    :return: str
        The resulting sentence.
    """
    return ' '.join(vocab.itos[i] if vocab.itos[i] not in ignore else '' for i in seq).strip()


def preprocess(strings: Iterator[str]) -> List[str]:
    """Preprocess a collection of strings.

    The following preprocessing steps are taken: Convert to lowercase, normalize unicode, and remove special
    characters (non-alphanumeric characters, except for '.', '!', and '?').

    :param strings: Iterator[str]
        A sequence of un-processed strings.
    :return: List[str]
        A list of preprocessed strings.
    """
    def preprocess_string(s: str) -> str:
        s = s.lower().strip()
        s = ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn')
        s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
        return s
    return [preprocess_string(s) for s in strings]


def print_loss(loss: float, name: str = '') -> None:
    """Format and print a loss and corresponding perplexity.

    :param loss: float
        The value of the loss to print.
    :param name: str, optional
        The name of the loss to print (e.g. "Train", "Test", etc.). Defaults to the empty string.
    :return: None
    """
    print(f'\t{name} loss:\t{loss:.3f}\t| {name} perplexity:\t{math.exp(loss):7.3f}')


def get_device() -> torch.device:
    """Detect and retrieve the available processing device.

    Returns cuda if available. Otherwise, returns cpu.

    :return: torch.device
        A PyTorch device.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
