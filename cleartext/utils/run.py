import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchtext.data import Iterator

from scripts.train import SOS_TOKEN, EOS_TOKEN


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


def sample(device, model, src, trg, test_data, num_examples, ignore_tokens):
    model.eval()
    # todo: randomize examples
    sources, targets = zip(*((example.src, example.trg) for example in test_data[:num_examples]))

    # run model with dummy target
    source_tensor = src.process(sources).to(device)
    dummy = torch.zeros(source_tensor.shape, dtype=int, device=device)
    dummy.fill_(trg.vocab[SOS_TOKEN])

    # select most likely tokens (ignoring non-word tokens)
    output = model(source_tensor, dummy, 0)[1:]
    ignore_indices = map(trg.vocab.stoi.get, ignore_tokens)
    for i in ignore_indices:
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