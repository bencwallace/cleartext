import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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

    def forward(self, source: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param source: Tensor
            Source sequence of shape (seq_len, batch_size)
        :return: Tuple[Tensor, Tensor]
            Outputs of shape (seq_len, batch_size, units) and state of shape (batch_size, 2 * units)
        """
        embedded = self.embedding(source)
        # todo: ensure state properly initialized
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
        print('Decoder state shape: ', dec_state.shape)
        input()
        dec_state = dec_state.unsqueeze(1).repeat(1, source_len, 1)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        combined = torch.cat((dec_state, enc_outputs), dim=2)
        combined = self.dropout(combined)
        scores = torch.tanh(self.fc(combined))
        # todo: shouldn't this be fully connected?
        scores = torch.sum(scores, dim=2)

        weights = F.softmax(scores, dim=1)
        return weights


class Decoder(nn.Module):
    def __init__(self, embed_weights: Tensor, units: int, dropout: float, enc_units) -> None:
        """
        :param embed_weights: Tensor
            Embedding weights of shape (trg_vocab_size, embed_dim)
        :param units: int
        :param dropout: float
        """
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embed_weights)
        self.vocab_size, embed_dim = embed_weights.shape
        self.rnn = nn.GRU((units * 2) + embed_dim, units)
        self.fc = nn.Linear(units + 2 * enc_units + embed_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token: Tensor, dec_state: Tensor, context) -> Tuple[Tensor, Tensor]:
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
                 embed_weights_src: Tensor, embed_weights_trg: Tensor,
                 rnn_units: int, attn_units: int,
                 dropout: float) -> None:
        """
        :param device: torch.device
        :param embed_weights_src: Tensor
            Embedding weights of shape (src_vocab_size, embed_dim)
        :param embed_weights_trg:
            Embedding  weights of shape (trg_vocab_size, embed_dim)
        :param rnn_units: int
        :param attn_units: int
        :param dropout: float
        """
        super().__init__()
        self.encoder = Encoder(embed_weights_src, rnn_units, dropout)
        self.attention = Attention(rnn_units, attn_units)
        self.decoder = Decoder(embed_weights_trg, rnn_units, dropout, rnn_units)

        self.device = device
        self.target_vocab_size = self.decoder.vocab_size

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

        outputs = torch.zeros(max_len, batch_size, self.target_vocab_size).to(self.device)
        out = target[0, :]
        for t in range(1, max_len):
            context = self._compute_context(state, enc_outputs)
            out, state = self.decoder(out, state, context)
            outputs[t] = out
            teacher_force = random.random() < teacher_forcing
            out = (target[t] if teacher_force else out.max(1)[1])

        return outputs

    # todo: extend to batches
    def beam_search_decode(self, source: Tensor, beam_size: int, trg_sos: int, max_len: int) -> Tensor:
        """
        :param source: Tensor
            Source sequence of shape (src_len, batch_size)
        :param beam_size: int
        :param trg_sos: int
        :param max_len: int
        :return:
        """
        batch_size = source.shape[1]
        # run encoder
        enc_outputs, state = self.encoder(source)

        # infer distribution over first word
        context = self._compute_context(state, enc_outputs)
        token = torch.Tensor(batch_size)
        token.fill_(trg_sos)                                                    # (batch_size,)
        out, state = self.decoder(token, state, context)                        # (batch_size, vocab_size), (batch_size, dec_units)
        probs, tokens = torch.topk(out, beam_size, dim=1)                       # (batch_size, beam_size), (batch_size, beam_size)
        scores = -torch.log(probs)                                              # (batch_size, beam_size)
        sequences = tokens.squeeze(1)                                           # (batch_size, seq_len=1, beam_size)

        # beam search main loop
        for t in range(max_len):
            context = self._compute_context(state, enc_outputs)
            for seq in sequences.permute(2, 0, 1):                              # (batch_size, seq_len)
                token = seq[:, -1]                                              # (batch_size,)
                out, state = self.decoder(token, state, context)
                probs, tokens = torch.topk(out, beam_size, dim=1)
                # todo

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
