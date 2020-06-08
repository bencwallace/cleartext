import torch


def train_step(model, iterator, criterion, optimizer, clip=1, verbose=False):
    model.train()
    epoch_loss = 0
    num_batches = len(iterator)
    for i, batch in enumerate(iterator):
        if verbose:
            print(f'Batch {i} of {num_batches}')

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


def eval_step(model, iterator, criterion):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            source = batch.src
            target = batch.trg

            output = model(source, target, 0)
            output = output[1:].view(-1, output.shape[-1])
            target = target[1:].view(-1)
            loss = criterion(output, target)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)