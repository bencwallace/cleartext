import math
import re
import unicodedata
from typing import Iterator, List, Tuple

import torch
from torch import nn as nn
from torch.nn import Module
from torchtext.vocab import Vocab


def count_parameters(model: Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, trainable + fixed


def format_time(elapsed: float) -> Tuple[int, int]:
    mins = int(elapsed / 60)
    secs = int(elapsed - (mins * 60))
    return mins, secs


def init_weights_(model: Module) -> None:
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param.data)
        else:
            nn.init.constant_(param.data, 0)


# todo: use tokenizer to remove space around punctuation, etc.
def seq_to_sentence(seq, vocab: Vocab, ignore: Iterator[int]) -> str:
    return ' '.join(vocab.itos[i] if vocab.itos[i] not in ignore else '' for i in seq).strip()


def preprocess(strings: Iterator[str]) -> List[str]:
    def preprocess_string(s: str) -> str:
        s = s.lower().strip()
        s = ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn')
        s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
        return s
    return [preprocess_string(s) for s in strings]


def print_loss(loss: float, name: str = '') -> None:
    print(f'\t{name} loss:\t{loss:.3f}\t| {name} perplexity:\t{math.exp(loss):7.3f}')


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
