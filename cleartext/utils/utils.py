from pathlib import Path

from torch import nn as nn


def get_proj_root():
    return Path(__file__).parent.parent.parent


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fixed = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, trainable + fixed


def format_time(start, stop):
    elapsed = stop - start
    mins = int(elapsed / 60)
    secs = int(elapsed - (mins * 60))
    return mins, secs


def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param.data)
        else:
            nn.init.constant_(param.data, 0)


def seq_to_sentence(seq, vocab, pad_token):
    def itos(i):
        s = vocab.itos[i]
        return '' if s == pad_token else s
    return ' '.join(list(map(itos, seq)))
