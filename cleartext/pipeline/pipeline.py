import time
from pathlib import Path
from typing import List, Tuple, Union

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field
from torchtext.data import BucketIterator
from torchtext.data.metrics import bleu_score

from .. import utils
from .. import PROJ_ROOT
from ..models import EncoderDecoder


class Pipeline(object):
    EOS_TOKEN = '<eos>'
    SOS_TOKEN = '<sos>'
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'

    MIN_FREQ = 2
    NUM_EXAMPLES = 10

    VECTORS_ROOT = PROJ_ROOT / 'vectors'
    MODELS_ROOT = PROJ_ROOT / 'models'

    @classmethod
    def deserialize(cls, path: Union[str, Path], index=1):
        # assume saved on gpu
        device = utils.get_device()

        path = Path(path)
        pl_dict = torch.load(path / f'model{index:02}.pt', map_location=device)
        name = pl_dict['name']
        pipeline = cls(name)

        # load preprocessing
        pipeline.src = torch.load(path / 'src.pt')
        pipeline.trg = torch.load(path / 'trg.pt')

        # load model
        src_vectors = pipeline.src.vocab.vectors = pipeline.src.vocab.vectors.to(device)
        trg_vectors = pipeline.trg.vocab.vectors = pipeline.trg.vocab.vectors.to(device)
        pipeline.model = EncoderDecoder(device, src_vectors, trg_vectors,
                                        pl_dict['rnn_units'], pl_dict['attn_units'], pl_dict['dropout'])
        pipeline.model.load_state_dict(pl_dict['model_state_dict'])
        pipeline.model.to(device)

        # load optimizer
        pipeline.optimizer = optim.Adam(pipeline.model.parameters())
        pipeline.optimizer.load_state_dict(pl_dict['optimizer_state_dict'])

        # load loss
        pipeline.criterion = nn.CrossEntropyLoss(ignore_index=pipeline.trg.vocab.stoi[cls.PAD_TOKEN])
        pipeline.criterion.load_state_dict(pl_dict['loss_state_dict'])
        pipeline.criterion.to(device)

        return pipeline

    def __init__(self, name: str = 'pl'):
        (self.MODELS_ROOT / name).mkdir(exist_ok=True)
        self.root = self.MODELS_ROOT / name
        self.device = utils.get_device()
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

    def load_data(self, dataset_cls: type, max_examples: Union[int, None] = None):
        field_args = {
            'tokenize': 'spacy', 'tokenizer_language': 'en_core_web_sm',
            'init_token': self.SOS_TOKEN, 'eos_token': self.EOS_TOKEN,
            'pad_token': self.PAD_TOKEN, 'unk_token': self.UNK_TOKEN,
            'lower': True, 'preprocessing': utils.preprocess
        }
        if self.src is None:
            self.src = Field(**field_args)
        if self.trg is None:
            self.trg = Field(**field_args)

        data = dataset_cls.splits(fields=(self.src, self.trg), max_examples=max_examples)
        self.train_data, self.valid_data, self.test_data = data

        return [len(dataset) for dataset in data]

    def load_vectors(self, embed_dim: int, trg_vocab: Union[None, int]) -> Tuple[int, int]:
        vectors_dir = self.VECTORS_ROOT / 'glove'
        vectors_path = f'glove.6B.{embed_dim}d'
        vocab_args = {'min_freq': self.MIN_FREQ, 'vectors': vectors_path, 'vectors_cache': vectors_dir}
        self.src.build_vocab(self.train_data, **vocab_args)
        self.trg.build_vocab(self.train_data, max_size=trg_vocab, **vocab_args)

        torch.save(self.src, self.root / f'src.pt')
        torch.save(self.trg, self.root / f'trg.pt')

        return len(self.src.vocab), len(self.trg.vocab)

    def prepare_data(self, batch_size: int) -> None:
        iterators = BucketIterator.splits((self.train_data, self.valid_data, self.test_data),
                                          batch_size=batch_size, device=self.device)
        self.train_iter, self.valid_iter, self.test_iter = iterators

    def build_model(self, rnn_units: int, attn_units: int, dropout: float) -> Tuple[int, int]:
        self.model_index += 1
        self.model_path = self.root / f'model{self.model_index:02}.pt'
        self.model = EncoderDecoder(self.device, self.src.vocab.vectors, self.trg.vocab.vectors,
                                    rnn_units, attn_units, dropout).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg.vocab.stoi[self.PAD_TOKEN])

        return utils.count_parameters(self.model)

    def train(self, num_epochs: int) -> None:
        best_valid_loss = float('inf')
        train_hist = [best_valid_loss, best_valid_loss]
        valid_hist = [best_valid_loss, best_valid_loss]
        times_hist = [0, 0]
        for epoch in range(num_epochs):
            # perform step
            start_time = time.time()
            train_loss = utils.train_step(self.model, self.train_iter, self.criterion, self.optimizer)
            valid_loss = utils.eval_step(self.model, self.valid_iter, self.criterion)
            end_time = time.time()
            elapsed = end_time - start_time

            # update histories
            valid_hist.append(valid_loss)
            mlflow.log_metrics({
                'train_loss': train_loss,
                'valid_loss': valid_loss
            }, step=epoch)

            # print epoch diagnostics
            epoch_mins, epoch_secs = utils.format_time(elapsed)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            utils.print_loss(train_loss, 'Train')
            utils.print_loss(valid_loss, 'Valid')

            # checkpoint, finalize, or continue depending on validation loss
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save({
                    'device': self.device,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_state_dict': self.criterion.state_dict()
                }, self.model_path)
            elif valid_hist[-1] > valid_hist[-2] > valid_hist[-3]:
                break

    def evaluate(self, beam_size: int = 10, max_len: int = 30, alpha: float = 1) -> Tuple[float, float, float, float]:
        # Compute losses
        train_loss = utils.eval_step(self.model, self.train_iter, self.criterion)
        valid_loss = utils.eval_step(self.model, self.valid_iter, self.criterion)
        test_loss = utils.eval_step(self.model, self.test_iter, self.criterion)

        # Run beam search on test data
        sources, targets = zip(*((e.src, e.trg) for e in self.test_data))
        outputs = [self.beam_search(s, beam_size, max_len, alpha) for s in sources]

        # Compute average (test) BLEU score
        bleu = 0
        for target, output in zip(targets, outputs):
            # kill whitespace tokens, which crash BLEU score for some reason
            target = ' '.join(target).split()
            output = ' '.join(output).split()
            bleu += bleu_score([target], [[output]])
        bleu /= len(targets)

        return train_loss, valid_loss, test_loss, bleu

    def beam_search(self, source, beam_size: int, max_len: int, alpha: float = 1) -> List[str]:
        """
        :param source:
            Iterator over tokens
        :param beam_size: int
        :param max_len: int
        :param alpha: float
            Length regularization parameter
        :return:
            List of tokens
        """
        sos_index = self.src.vocab.stoi[self.SOS_TOKEN]
        eos_index = self.trg.vocab.stoi[self.EOS_TOKEN]

        # run beam search
        source_tensor = self.src.process([source]).to(self.device)
        source_tensor = source_tensor.squeeze(1)
        beam_search_results = self.model.beam_search(source_tensor, beam_size, sos_index, max_len)
        output_tensor, scores = beam_search_results                     # (max_len, beam_size), (beam_size,)

        # find ends of sequences
        lengths = torch.empty(beam_size, dtype=torch.long, device=self.device)
        for beam in range(beam_size):
            try:
                index = output_tensor[:, beam].tolist().index(eos_index)
                lengths[beam] = index
            except ValueError:
                lengths[beam] = max_len

        # normalize scores and select winning sequence
        scores = scores / lengths ** alpha                              # (beam_size,)
        idx = torch.argmax(scores)
        winner_len = lengths[idx]
        winner = output_tensor[:winner_len, idx]

        # todo: use utils.seq_to_sentence -- first figure out why not getting <unk>
        result = [self.trg.vocab.itos[d] for d in winner]
        return result
