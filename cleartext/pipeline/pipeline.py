import random
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

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
    """Simplification pipeline.

    Can be used to load and prepare data, build, train, and evaluate a model, and decoder sequences using beam search.
    """
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
        """Deserialize a pipeline consisting of tokenizers and a model.

        :param path: str or Path
            Path to the directory containing tokenizers and model.
        :param index: int
            Model index.
        :return: Pipeline
        """
        # assume saved on gpu
        device = utils.get_device()

        path = Path(path)
        pl_dict = torch.load(path / f'model{index:02}.pt', map_location=device)
        name = pl_dict.get('name')
        if name is None:
            name = path.name
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
        """Initialize a pipeline with a given name.

        :param name: str
            Pipeline name.
        """
        (self.MODELS_ROOT / name).mkdir(exist_ok=True, parents=True)
        self.root = self.MODELS_ROOT / name
        self.device = utils.get_device()
        self.model_index = 0
        self.name = name

        self.src = None
        self.trg = None

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.train_iter = None
        self.valid_iter = None
        self.test_iter = None

        self.rnn_units = None
        self.attn_units = None
        self.dropout = None

        self.model_path = None
        self.model = None
        self.optimizer = None
        self.criterion = None

    def load_data(self, dataset_cls: type, max_examples: Optional[int] = None) -> List[int]:
        """Load a dataset.

        :param dataset_cls: type
            Dataset type (WikiSmall or WikiLarge).
        :param max_examples: int, optional
            Maximum number of training examples to load. Default is to load all examples.
        :return: List[int]
            Sizes of training, validation, and test sets, respectively.
        """
        field_args = {
            'tokenize': 'spacy', 'tokenizer_language': 'en_core_web_sm',
            'init_token': self.SOS_TOKEN, 'eos_token': self.EOS_TOKEN,
            'pad_token': self.PAD_TOKEN, 'unk_token': self.UNK_TOKEN,
            'lower': True, 'preprocessing': utils.preprocess
        }
        warnings.filterwarnings('ignore', category=DeprecationWarning, lineno=6)
        if self.src is None:
            self.src = Field(**field_args)
        if self.trg is None:
            self.trg = Field(**field_args)

        data = dataset_cls.splits(fields=(self.src, self.trg), max_examples=max_examples)
        self.train_data, self.valid_data, self.test_data = data

        return [len(dataset) for dataset in data]

    def load_vectors(self, embed_dim: int, src_vocab: Optional[int], trg_vocab: Optional[int]) -> Tuple[int, int]:
        """Load word embedding vectors.

        Must be called after calling `load_data`.

        :param embed_dim: int
            Embedding dimension.
        :param src_vocab: int
            Maximum source vocabulary size.
        :param trg_vocab: int
            Maximum target vocabulary size.
        :return: Tuple[int, int]
            Source and target vocabulary sizes, respectively.
        """
        vectors_dir = self.VECTORS_ROOT / 'glove'
        glove = f'glove.6B.{embed_dim}d'
        vocab_args = {'min_freq': self.MIN_FREQ, 'vectors': glove, 'vectors_cache': vectors_dir}
        self.src.build_vocab(self.train_data, max_size=src_vocab, **vocab_args)
        self.trg.build_vocab(self.train_data, max_size=trg_vocab, **vocab_args)

        torch.save(self.src, self.root / f'src.pt')
        torch.save(self.trg, self.root / f'trg.pt')

        return len(self.src.vocab), len(self.trg.vocab)

    def prepare_data(self, batch_size: int, seed: Optional[int] = None) -> None:
        """Prepare loaded dataset into batches.

        Must be called after calling `load_data`.

        :param batch_size: int
            Batch size.
        :param seed: int, optional
            Random seed.
        :return: None
        """
        # `BucketIterator` makes use of global `random` state
        if seed:
            random.seed(seed)
        iterators = BucketIterator.splits((self.train_data, self.valid_data, self.test_data),
                                          batch_size=batch_size, device=self.device)
        self.train_iter, self.valid_iter, self.test_iter = iterators

    def build_model(self, rnn_units: int, attn_units: int, num_layers: int, dropout: float) -> Tuple[int, int]:
        """Build an encoder-decoder model for simplification.

        Must be called after calling `load_vectors`.

        :param rnn_units: int
            RNN state dimensionality.
        :param attn_units: int
            Attention state dimensionality.
        :param num_layers: int
            Number of encoder layers.
        :param dropout: float
            Dropout probability.
        :return: Tuple[int, int]
            Number of trainable parameters and total number of model parameters, respectively.
        """
        warnings.filterwarnings('ignore', category=UserWarning, lineno=50)

        self.rnn_units = rnn_units
        self.attn_units = attn_units
        self.dropout = dropout

        self.model_index += 1
        self.model_path = self.root / f'model{self.model_index:02}.pt'
        self.model = EncoderDecoder(self.device, self.src.vocab.vectors, self.trg.vocab.vectors,
                                    rnn_units, attn_units, num_layers, dropout).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters())
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg.vocab.stoi[self.PAD_TOKEN])

        return utils.count_parameters(self.model)

    def train(self, num_epochs: int) -> int:
        """Train model.

        Trains the model for at most `num_epochs` epochs, stopping early if the validation loss increases for two
        consecutive epochs. Whenever the validation loss achieves a minimum, the model is checkpointed. The best model
        is re-loaded at the end of the training cycle.

        Must be called after calling `prepare_data` and `build_model`.

        :param num_epochs: int
            Maximum number of epochs.
        :return: int
            Number of epochs spent training.
        """
        best_valid_loss = float('inf')
        valid_hist = [best_valid_loss, best_valid_loss]
        for epoch in range(num_epochs):
            # perform step
            start_time = time.time()
            train_loss = utils.train(self.model, self.train_iter, self.criterion, self.optimizer)
            valid_loss = utils.evaluate(self.model, self.valid_iter, self.criterion)
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
                    'name': self.name,
                    'device': self.device,
                    'rnn_units': self.rnn_units,
                    'attn_units': self.attn_units,
                    'dropout': self.dropout,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_state_dict': self.criterion.state_dict()
                }, self.model_path)
            elif valid_hist[-1] > valid_hist[-2] > valid_hist[-3]:
                return epoch + 1

    def evaluate(self, beam_size: int = 10, max_len: int = 30, alpha: float = 1) -> Tuple[float, float, float, float]:
        """Evaluate the model.

        :param beam_size: int
            Beam size used in beam search.
        :param max_len: int
            Maximum beam search output length.
        :param alpha: float
            Beam search regularization parameter.
        :return: Tuple[float, float, float, float]
            Training loss, validation loss, test loss, and BLEU score, respectively.
        """
        # Compute losses
        train_loss = utils.evaluate(self.model, self.train_iter, self.criterion)
        valid_loss = utils.evaluate(self.model, self.valid_iter, self.criterion)
        test_loss = utils.evaluate(self.model, self.test_iter, self.criterion)

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

    def beam_search(self, source: List[str], beam_size: int,
                    max_len: Optional[int] = None, alpha: float = 1) -> List[str]:
        """Perform beam search on a tokenized source sequence.

        :param source: List[str]
            List of tokens.
        :param beam_size: int
            Beam size.
        :param max_len: int
            Maximum output length.
        :param alpha: float
            Length regularization parameter
        :return: List[str]
            List of output tokens.
        """
        with torch.no_grad():
            sos_index = self.trg.vocab.stoi[self.SOS_TOKEN]
            eos_index = self.trg.vocab.stoi[self.EOS_TOKEN]
            unk_index = self.trg.vocab.stoi[self.UNK_TOKEN]
            if max_len is None:
                max_len = 2 * len(source)

            # run beam search
            source_tensor = self.src.process([source]).to(self.device)
            source_tensor = source_tensor.squeeze(1)
            beam_search_results = self.model.beam_search(source_tensor, beam_size, sos_index, unk_index, max_len)
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

            result = [self.trg.vocab.itos[d] for d in winner]
            return result
