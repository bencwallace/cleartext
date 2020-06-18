import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchtext.data import Iterator


CLIP = 1


def train(model: Module, iterator: Iterator, criterion: Module, optimizer: Optimizer) -> float:
    """Perform a single training step (epoch).

    :param model: Module
        The PyTorch model being trained.
    :param iterator: Iterator
        A Torchtext Iterator on batches of data.
    :param criterion: Module
        A PyTorch loss function.
    :param optimizer: Optimizer
        A PyTorch optimizer.
    :return: float
        The average epoch loss with respect to the loss function over all batches.
    """
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: Module, iterator: Iterator, criterion: Module) -> float:
    """Evaluate a model over a batched dataset.

    :param model: Module
        The PyTorch model to evaluate.
    :param iterator: Iterator
        A Torchtext iterator on batches of data.
    :param criterion: Module
        A PyTorch loss function.
    :return: float
        The average model loss with respect to the loss function on the given data.
    """
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
