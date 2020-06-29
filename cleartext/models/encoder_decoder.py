from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch import Tensor

from .components import Encoder, Attention, Decoder
from .. import utils


class EncoderDecoder(nn.Module):
    """Encoder-decoder (or sequence-to-sequence) model.

    Attributes
    ----------
    device: torch.device
        Device on which model is loaded.
    trg_vocab_size: int
        Size of the target vocabulary.
    encoder: Module
        Encoder sub-module.
    decoder: Module
        Decoder sub-module.
    attention: Module
        Attention sub-module.
    fc_hidden: Module
        Fully-connected layer. Used to perform dimensionality reduction on final (bidirectional) encoder hidden state,
        which will be used to initialize the (unidirectional) hidden state of the decoder.
    fc_cell: Module
        Fully-connected layer. Purpose similar to that of fc_hidden, but for the cell state.
    """
    def __init__(self, device: torch.device,
                 embed_weights_src: Tensor, embed_weights_trg: Tensor,
                 rnn_units: int, attn_units: int,
                 num_layers: int,
                 dropout: float) -> None:
        """Initialize model.

        :param device: torch.device
            Device on which to load the model.
        :param embed_weights_src: Tensor
            Embedding weights of shape (src_vocab_size, embed_dim)
        :param rnn_units: int
            Number of hidden units in all sub-module RNNs.
        :param attn_units: int
            Number of hidden units in attention layer.
        :param num_layers: int
            Number of encoder layers.
        :param dropout: float
            Dropout probability.
        """
        super().__init__()
        self.device = device
        self.trg_vocab_size = embed_weights_trg.shape[0]
        self.num_layers = num_layers
        self.rnn_units = rnn_units

        self.encoder = Encoder(embed_weights_src, rnn_units, num_layers, dropout)
        self.decoder = Decoder(embed_weights_trg, rnn_units, rnn_units, num_layers, dropout)
        self.attention = Attention(rnn_units, rnn_units, attn_units, dropout)

        self.fc_hidden = nn.Linear(2 * rnn_units, rnn_units)
        self.fc_cell = nn.Linear(2 * rnn_units, rnn_units)

        utils.init_weights_(self.fc_hidden)
        utils.init_weights_(self.fc_cell)

    def forward(self, source: Tensor, target: Tensor, teacher_forcing: float = 0.3) -> Tensor:
        """Translates a source sequence into an output sequence using teacher forcing.

        Teacher forcing is used to randomly select, at each time-step, whether or not to use the target output or the
        previous output as input to the decoder.

        :param source: Tensor
            Source sequence of shape (src_len, batch_size).
        :param target:
            Target sequence of shape (trg_len, batch_size).
        :param teacher_forcing: float
            Teacher forcing probability.
        :return: Tensor
            Output tensor of shape (trg_len, batch_size).
        """
        batch_size = source.shape[1]
        max_len = target.shape[0]
        enc_outputs, state = self.encoder(source)
        hidden, cell = self._reduce(state)

        outputs = torch.zeros(max_len, batch_size, self.trg_vocab_size, device=self.device)
        out = target[0, :]
        for t in range(1, max_len):
            context = self._compute_context(hidden, enc_outputs)
            out, (hidden, cell) = self.decoder(out, context, hidden, cell)
            outputs[t] = out
            teacher_force = torch.bernoulli(torch.tensor(teacher_forcing, dtype=torch.float)).item()
            out = (target[t] if teacher_force else out.max(1)[1])

        return outputs

    def beam_search(self, source: Tensor, beam_size: int, trg_sos: int, trg_unk: int,
                    max_len: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Translate a source sequence into an output sequence using beam search.

        :param source: Tensor
            Source sequence of shape (src_len,).
        :param beam_size: int
            The beam size.
        :param trg_sos: int
            Target start-of-sentence index, used to initialize output.
        :param max_len: int, optional
            Maximum output length. Defaults to None, meaning the maximum length is computed from the input length.
        :return: Tuple[Tensor, Tensor]
            Un-trimmed output sequences of shape (max_len, beam_size) and *unnormalized* scores of shape (beam_size,).
        """
        self.eval()

        # run encoder
        enc_outputs, state = self.encoder(source.unsqueeze(1))
        hidden, cell = self._reduce(state)

        # initialize distribution over first word
        context = self._compute_context(hidden, enc_outputs)
        token = torch.tensor([trg_sos], dtype=torch.long, device=self.device)
        out, (hidden, cell) = self.decoder(token, context, hidden, cell)        # (1, vocab), (layers, 1, dec_units)

        # compute log-likelihood scores
        probs = softmax(out, dim=1)
        scores = torch.log(probs)

        # initialize scores, sequences, and states
        scores, sequences = torch.topk(scores, beam_size, dim=1)                # (1, beam_size), (seq_len=1, beam_size)
        scores = scores.squeeze(0)                                              # (beam_size,)
        hidden_states = hidden.unsqueeze(0).repeat(beam_size, 1, 1, 1)             # (beam_size, 1, dec_units)
        cell_states = cell.unsqueeze(0).repeat(beam_size, 1, 1, 1)

        # main loop over time steps
        for _ in range(1, max_len):
            # generate scores for next time step
            all_scores = torch.empty(0, device=self.device)
            for i, seq in enumerate(sequences.permute(1, 0)):                   # (seq_len,)
                # run decoder
                context = self._compute_context(hidden_states[i], enc_outputs)
                token = seq[-1].unsqueeze(0)                                    # (1,)

                # (1, vocab_size), (1, dec_units)
                out, (hidden_states[i], cell_states[i]) = self.decoder(token, context, hidden_states[i], cell_states[i])

                # apply softmax
                probs = softmax(out, dim=1)                                     # (1, vocab_size)

                # mask out <unk>
                # probs[0, trg_unk] = 0

                # update scores
                curr_score = scores[i].item()
                new_scores = curr_score + torch.log(probs)                      # (1, vocab_size)
                all_scores = torch.cat((all_scores, new_scores), dim=0)         # (i + 1, vocab_size)

            # all_scores has shape (beam_size, vocab_size)
            all_scores = all_scores.flatten()                                   # (beam_size * vocab_size,)
            scores, indices = torch.topk(all_scores, beam_size)                 # (beam_size,)

            # create placeholder for new sequences and scores
            new_sequences = torch.empty(0, dtype=torch.long, device=self.device)
            # loop through `beam_size` number of candidates and build each one
            for idx in indices:
                # convert idx into corresponding sequence and additional token (notice all_scores.shape above)
                seq_index = idx // self.trg_vocab_size
                vocab_index = idx % self.trg_vocab_size

                # add new sequence
                new_seq = torch.cat((sequences[:, seq_index], vocab_index.unsqueeze(0)))     # (_ + 1,)
                new_sequences = torch.cat((new_sequences, new_seq.unsqueeze(1)), dim=1)      # (_ + 1, num_iterations)
            # update sequences tensor
            sequences = new_sequences

        return sequences, scores

    def _compute_context(self, dec_state, enc_outputs):
        """Computes a context vector using the attention module.

        :param dec_state: Tensor
            Decoder state of shape (num_layers, batch_size, dec_units).
        :param enc_outputs: Tensor
            Encoder outputs of shape (seq_len, batch_size, 2 * enc_units).
        :return:
            Context vector of shape (1, batch_size, 2 * enc_units).
        """
        weights = self.attention(dec_state[-1], enc_outputs).unsqueeze(1)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        context = torch.bmm(weights, enc_outputs).permute(1, 0, 2)
        return context

    def _reduce(self, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        :param state: Tuple[Tensor, Tensor]
            Final encoder hidden/cell states of shape (2 * num_layers, batch_size, enc_units).
        :return: Tuple[Tensor, Tensor]
            Initial decoder hidden/cell states of shape (num_layers, batch_size, dec_units).
        """
        hidden, cell = state
        hidden = hidden.view(self.num_layers, 2, -1, self.rnn_units)
        cell = cell.view(self.num_layers, 2, -1, self.rnn_units)

        # combine and reshape bidirectional states for compatibility with (unidirectional) decoder
        hidden_combined = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        hidden = torch.tanh(self.fc_hidden(hidden_combined))

        cell_combined = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)
        cell = torch.tanh(self.fc_cell(cell_combined))

        return hidden, cell
