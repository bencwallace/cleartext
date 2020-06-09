import math
import re
import unicodedata
from pathlib import Path
from typing import Tuple

from torch import nn as nn
from torch.nn import Module
from torchtext.vocab import Vocab


def count_parameters(model: Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, trainable + fixed


def format_time(start: float, stop: float) -> Tuple[int, int]:
    elapsed = stop - start
    mins = int(elapsed / 60)
    secs = int(elapsed - (mins * 60))
    return mins, secs


def init_weights(model: Module) -> None:
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param.data)
        else:
            nn.init.constant_(param.data, 0)


# todo: use tokenizer to remove space around punctuation, etc.
def seq_to_sentence(seq, vocab: Vocab, ignore) -> str:
    def itos(i):
        s = vocab.itos[i]
        return '' if s in ignore else s
    return ' '.join(list(map(itos, seq)))


def preprocess_string(s: str) -> str:
    s = s.lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s


def preprocess(strings):
    return list(map(preprocess_string, strings))


def print_loss(loss: float, name: str = '') -> None:
    print(f'\t{name} loss:\t{loss:.3f}\t| {name} perplexity:\t{math.exp(loss):7.3f}')
