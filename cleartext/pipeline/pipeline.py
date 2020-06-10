import pathlib
import signal
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field
from torchtext.data import BucketIterator
from torchtext.data.metrics import bleu_score

import cleartext.utils as utils
from cleartext import PROJ_ROOT
from cleartext.models import EncoderDecoder


class Pipeline(object):
    EOS_TOKEN = '<eos>'
    SOS_TOKEN = '<sos>'
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'

    MIN_FREQ = 2
    NUM_EXAMPLES = 4

    VECTORS_ROOT = PROJ_ROOT / 'vectors'
    MODELS_ROOT = PROJ_ROOT / 'models'

    @classmethod
    def deserialize(cls):
        # todo
        pass

    def __init__(self, name='pl'):
        self.name = name
        (self.MODELS_ROOT / name).mkdir(exist_ok=True)
        self.root = self.MODELS_ROOT / name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_index = 0

        self.src = None
        self.trg = None

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.train_iter = None
        self.valid_iter = None
        self.test_iter = None

        self.model_path = None
        self.model = None
        self.optimizer = None
        self.criterion = None

    def load_data(self, dataset_cls, max_examples):
        # reset iterators to force client to run `prepare_data`
        self.train_iter = None
        self.valid_iter = None
        self.test_iter = None

        field_args = {
            'tokenize': 'spacy', 'tokenizer_language': 'en_core_web_sm',
            'init_token': self.SOS_TOKEN, 'eos_token': self.EOS_TOKEN,
            'pad_token': self.PAD_TOKEN, 'unk_token': self.UNK_TOKEN,
            'lower': True, 'preprocessing': utils.preprocess
        }
        self.src = Field(**field_args)
        self.trg = Field(**field_args)

        data = dataset_cls.splits(fields=(self.src, self.trg), max_examples=max_examples)
        self.train_data, self.valid_data, self.test_data = data

        return list(map(len, data))

    def prepare_data(self, embed_dim, trg_vocab, batch_size):
        # reset model, optimizer, and loss to force client to run `build_model`
        self.model_index = 0
        self.model_path = None
        self.model = None

        vectors_dir = self.VECTORS_ROOT / 'glove'
        vectors_path = f'glove.6B.{embed_dim}d'
        vocab_args = {'min_freq': self.MIN_FREQ, 'vectors': vectors_path, 'vectors_cache': vectors_dir}
        self.src.build_vocab(self.train_data, **vocab_args)
        self.trg.build_vocab(self.train_data, max_size=trg_vocab, **vocab_args)

        # todo: check if necessary that serialization depend on `embed_dim`
        torch.save(self.src, self.root / f'src-{embed_dim}d.pt')
        torch.save(self.trg, self.root / f'trg-{embed_dim}d-{trg_vocab}')

        iterators = BucketIterator.splits((self.train_data, self.valid_data, self.test_data),
                                          batch_size=batch_size, device=self.device)
        self.train_iter, self.valid_iter, self.test_iter = iterators

        return len(self.src.vocab), len(self.trg.vocab)

    def build_model(self, rnn_units, attn_units, dropout):
        self.model_index += 1
        self.model_path = self.root / f'model{self.model_index:02}.pt'
        self.model = EncoderDecoder(self.device, self.src.vocab.vectors, self.trg.vocab.vectors,
                                    rnn_units, attn_units, dropout).to(self.device)
        self.model.apply(utils.init_weights)
        trainable, total = utils.count_parameters(self.model)

        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg.vocab.stoi[self.PAD_TOKEN])

        return trainable, total

    def train(self, num_epochs):
        best_valid_loss = float('inf')
        train_hist = [best_valid_loss, best_valid_loss]
        valid_hist = [best_valid_loss, best_valid_loss]
        for epoch in range(num_epochs):
            # perform step
            start_time = time.time()
            train_loss = utils.train_step(self.model, self.train_iter, self.criterion, self.optimizer)
            valid_loss = utils.eval_step(self.model, self.valid_iter, self.criterion)
            end_time = time.time()

            # update histories
            train_hist.append(train_loss)
            valid_hist.append(valid_loss)

            # print epoch diagnostics
            epoch_mins, epoch_secs = utils.format_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            utils.print_loss(train_loss, 'Train')
            utils.print_loss(valid_loss, 'Valid')

            # checkpoint, finalize, or continue depending on validation loss
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save({
                    'epoch': epoch,
                    'epoch_start': start_time,
                    'epoch_end': end_time,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_hist': train_hist,
                    'valid_hist': valid_hist
                }, self.model_path)
            elif valid_hist[-1] > valid_hist[-2] > valid_hist[-3]:
                self._finalize()
                break
        else:
            self._finalize()

    def _finalize(self):
        # reset default signal handler
        def default_handler(_signal, _frame):
            sys.exit(0)
        signal.signal(signal.SIGINT, default_handler)

        self.print_diagnostics()

    def print_diagnostics(self):
        # Compute and print test loss
        print('\nTesting model')
        test_loss = utils.eval_step(self.model, self.test_iter, self.criterion)
        utils.print_loss(test_loss, 'Test')

        # Generate and print samples
        sources, targets, outputs = self.sample()
        source_outs = []
        target_outs = []
        output_outs = []
        for source, target, output in zip(sources, targets, outputs):
            source_out = '> ' + ' '.join(source)
            target_out = '= ' + ' '.join(target)
            output_out = '< ' + ' '.join(output)

            source_outs.append(source_out)
            target_outs.append(target_out)
            output_outs.append(output_out)

            print(source_out)
            print(target_out)
            print(output_out)

        # Compute and print BLEU score
        sources, targets, outputs = self.sample()
        score = 0
        for target, output in zip(targets, outputs):
            # kill whitespace tokens, which crash BLEU score for some reason
            target = ' '.join(target).split()
            output = ' '.join(output).split()

            score += bleu_score([target], [[output]])
        score /= len(targets)
        print(f'Average BLEU score: {score:.3f}')

        # save summary data
        path = str(self.model_path) + '.txt'
        with open(path, 'w') as f:
            f.write(f'Test loss: {test_loss}\n')
            f.write(f'BLEU score: {score}\n')
            for source_out, target_out, output_out in zip(source_outs, target_outs, output_outs):
                f.write(source_out + '\n')
                f.write(target_out + '\n')
                f.write(output_out + '\n')

    def sample(self, num_examples=NUM_EXAMPLES):
        self.model.eval()

        # todo: randomize examples
        # run model with dummy target
        sources, targets = zip(*((example.src, example.trg) for example in self.test_data[:num_examples]))
        source_tensor = self.src.process(sources).to(self.device)
        _dummy = torch.zeros(source_tensor.shape, dtype=int, device=self.device)
        _dummy.fill_(self.trg.vocab[self.SOS_TOKEN])

        # select most likely tokens (ignoring non-word tokens)
        output = self.model(source_tensor, _dummy, 0)[1:]
        ignore_indices = map(self.trg.vocab.stoi.get, [self.SOS_TOKEN, self.UNK_TOKEN, self.PAD_TOKEN])
        for i in ignore_indices:
            output.data[:, :, i] = 0
        output = output.argmax(dim=2)

        # trim past eos token and denumericalize
        output = output.T.tolist()
        trimmed = []
        for out in output:
            try:
                eos_index = out.index(self.trg.vocab[self.EOS_TOKEN])
            except ValueError:
                eos_index = len(out)
            out = out[:eos_index]
            out = list(map(lambda i: self.trg.vocab.itos[i], out))
            trimmed.append(out)

        return sources, targets, trimmed
