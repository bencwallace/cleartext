from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch import Tensor

from .. import utils


class Encoder(nn.Module):
    def __init__(self, embed_weights: Tensor, units: int, dropout: float) -> None:
        """
        :param embed_weights: Tensor
            Embedding weights of shape (src_vocab_size, embed_dim)
        :param units: int
        :param dropout: float
        """
        super().__init__()
        self.units = units
        self.embed_dim = embed_weights.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embed_weights)
        self.gru = nn.GRU(self.embed_dim, units, bidirectional=True)
        self.fc = nn.Linear(units * 2, units)
        self.dropout = nn.Dropout(dropout)

        utils.init_weights_(self.gru)
        utils.init_weights_(self.fc)

    def forward(self, source: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param source: Tensor
            Source sequence of shape (seq_len, batch_size)
        :return: Tuple[Tensor, Tensor]
            Outputs of shape (seq_len, batch_size, units) and state of shape (batch_size, 2 * units)
        """
        embedded = self.embedding(source)
        outputs, state = self.gru(embedded)

        # combine and reshape bidirectional states for compatibility with (unidirectional) decoder
        combined = torch.cat((state[-2, :, :], state[-1, :, :]), dim=1)
        combined = self.dropout(combined)
        state = torch.tanh(self.fc(combined))

        return outputs, state


class Attention(nn.Module):
    def __init__(self, state_dim: int, units: int, dropout: float = 0) -> None:
        """
        :param state_dim: int
        :param units: int
        :param dropout: float
        """
        super().__init__()
        self.attn_in = state_dim * 3

        self.fc = nn.Linear(self.attn_in, units)
        self.dropout = nn.Dropout(dropout)

        utils.init_weights_(self.fc)

    def forward(self, dec_state: Tensor, enc_outputs: Tensor) -> Tensor:
        """
        :param dec_state: Tensor
            Previous or initial decoder state of shape (batch_size, dec_units)
        :param enc_outputs: Tensor
            Encoder outputs of shape (source_len, batch_size, 2 * enc_units)
        :return:
            Attention weights of shape (batch_size, source_len)
        """
        source_len = enc_outputs.shape[0]

        # vectorize computation of Bahdanau attention scores for all encoder outputs
        dec_state = dec_state.unsqueeze(1).repeat(1, source_len, 1)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        combined = torch.cat((dec_state, enc_outputs), dim=2)
        combined = self.dropout(combined)
        scores = torch.tanh(self.fc(combined))
        # todo: shouldn't this be fully connected?
        scores = torch.sum(scores, dim=2)

        weights = softmax(scores, dim=1)
        return weights


class Decoder(nn.Module):
    def __init__(self, embed_weights: Tensor, units: int, dropout: float, enc_units: int, vocab_size: int) -> None:
        """
        :param embed_weights: Tensor
            Embedding weights of shape (trg_vocab_size, embed_dim)
        :param units: int
        :param dropout: float
        """
        super().__init__()
        self.vocab_size = vocab_size
        embed_dim = embed_weights.shape[1]

        self.embedding = nn.Embedding.from_pretrained(embed_weights)
        self.rnn = nn.GRU((units * 2) + embed_dim, units)
        self.fc = nn.Linear(units + 2 * enc_units + embed_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)

        utils.init_weights_(self.rnn)
        utils.init_weights_(self.fc)

    def forward(self, token: Tensor, dec_state: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param token: Tensor
            Token of shape (batch_size,)
        :param dec_state:
            Decoder state of shape (batch_size, dec_units)
        :param context:
            Context vector of shape (1, batch_size, 2 * enc_units)
        :return: Tuple[Tensor, Tensor]
            Vocabulary scores of shape (batch_size, vocab_size) and decoder state of shape (batch_size, dec_units)
        """
        token = token.unsqueeze(0)
        embedded = self.embedding(token)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, dec_state = self.rnn(rnn_input, dec_state.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        context = context.squeeze(0)

        # compute output using gru output, context vector, and embedding of previous output
        combined = torch.cat((output, context, embedded), dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)

        # return logits (rather than softmax activations) for compatibility with cross-entropy loss
        dec_state = dec_state.squeeze(0)
        return output, dec_state


class EncoderDecoder(nn.Module):
    def __init__(self, device: torch.device,
                 embed_weights_src: Tensor, trg_vocab_size: int,
                 rnn_units: int, attn_units: int,
                 dropout: float) -> None:
        """
        :param device: torch.device
        :param embed_weights_src: Tensor
            Embedding weights of shape (src_vocab_size, embed_dim)
        :param trg_vocab_size:
        :param rnn_units: int
        :param attn_units: int
        :param dropout: float
        """
        super().__init__()
        self.device = device

        self.encoder = Encoder(embed_weights_src, rnn_units, dropout)
        self.attention = Attention(rnn_units, attn_units)
        self.decoder = Decoder(embed_weights_src, rnn_units, dropout, rnn_units, trg_vocab_size)

    def forward(self, source: Tensor, target: Tensor, teacher_forcing: float = 0.3) -> Tensor:
        """
        :param source: Tensor
            Source sequence of shape (src_len, batch_size)
        :param target:
            Target sequence of shape (trg_len, batch_size)
        :param teacher_forcing: float
        :return:
        """
        batch_size = source.shape[1]
        max_len = target.shape[0]
        enc_outputs, state = self.encoder(source)

        outputs = torch.zeros(max_len, batch_size, self.target_vocab_size, device=self.device)
        out = target[0, :]
        for t in range(1, max_len):
            context = self._compute_context(state, enc_outputs)
            out, state = self.decoder(out, state, context)
            outputs[t] = out
            teacher_force = torch.bernoulli(torch.tensor(teacher_forcing, dtype=torch.float)).item()
            out = (target[t] if teacher_force else out.max(1)[1])

        return outputs

    def beam_search(self, source: Tensor, beam_size: int, trg_sos: int, max_len: int) -> Tuple[Tensor, Tensor]:
        """
        :param source: Tensor
            Source sequence of shape (src_len,)
        :param beam_size: int
        :param trg_sos: int
        :param max_len: int
        :return: Tuple[Tensor, Tensor]
            Un-trimmed output sequences of shape (max_len, beam_size) and *unnormalized* scores of shape (beam_size,)
        """
        self.eval()

        # run encoder
        enc_outputs, state = self.encoder(source.unsqueeze(1))

        # initialize distribution over first word
        context = self._compute_context(state, enc_outputs)
        token = torch.tensor([trg_sos], dtype=torch.long, device=self.device)
        out, state = self.decoder(token, state, context)                        # (1, vocab_size), (1, dec_units)

        # compute log-likelihood scores
        probs = softmax(out, dim=1)
        scores = torch.log(probs)

        # initialize scores, sequences, and states
        scores, sequences = torch.topk(scores, beam_size, dim=1)                # (1, beam_size), (seq_len=1, beam_size)
        scores = scores.squeeze(0)                                              # (beam_size,)
        states = state.unsqueeze(0).repeat(beam_size, 1, 1)                     # (beam_size, 1, dec_units)

        # main loop over time steps
        for t in range(1, max_len):
            # generate scores for next time step
            all_scores = torch.empty(0, device=self.device)
            for i, seq in enumerate(sequences.permute(1, 0)):                   # (seq_len,)
                # run decoder
                context = self._compute_context(states[i], enc_outputs)
                token = seq[-1].unsqueeze(0)                                    # (1,)
                out, states[i] = self.decoder(token, states[i], context)        # (1, vocab_size), (1, dec_units)
                # apply softmax
                probs = softmax(out, dim=1)                                     # (1, vocab_size)

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
                idx.item()
                # convert idx into corresponding sequence and additional token (notice all_scores.shape above)
                seq_index = idx // self.target_vocab_size
                vocab_index = idx % self.target_vocab_size

                # add new sequence
                new_seq = torch.cat((sequences[:, seq_index], vocab_index.unsqueeze(0)))     # (t + 1,)
                new_sequences = torch.cat((new_sequences, new_seq.unsqueeze(1)), dim=1)      # (t + 1, num_iterations)
            # update sequences tensor
            sequences = new_sequences

        return sequences, scores

    def _compute_context(self, dec_state, enc_outputs):
        """
        :param dec_state: Tensor
            Decoder state of shape (batch_size, dec_units)
        :param enc_outputs: Tensor
            Encoder outputs of shape (seq_len, batch_size, 2 * enc_units)
        :return:
            Context vector of shape (1, batch_size, 2 * enc_units)
        """
        weights = self.attention(dec_state, enc_outputs).unsqueeze(1)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        context = torch.bmm(weights, enc_outputs).permute(1, 0, 2)
        return context
