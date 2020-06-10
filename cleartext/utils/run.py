import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchtext.data import Iterator


def train_step(model: Module, iterator: Iterator, criterion: Module, optimizer: Optimizer, clip: int = 1) -> float:
    model.train()
    epoch_loss = 0
    for batch in iterator:
        source = batch.src
        target = batch.trg

        optimizer.zero_grad()
        output = model(source, target)
        output = output[1:].view(-1, output.shape[-1])
        target = target[1:].view(-1)
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def eval_step(model: Module, iterator: Iterator, criterion: Module) -> float:
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            source = batch.src
            target = batch.trg

            output = model(source, target, 0)
            output = output[1:].view(-1, output.shape[-1])
            target = target[1:].view(-1)
            loss = criterion(output, target)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
