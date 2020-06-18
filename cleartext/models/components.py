from typing import Tuple

import torch
from torch import nn as nn, Tensor
from torch.nn.functional import softmax

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
    def __init__(self, enc_units: int, dec_units: int, units: int, dropout: float = 0) -> None:
        """
        :param enc_units: int
        :param dec_units: int
        :param units: int
        :param dropout: float
        """
        super().__init__()
        self.attn_in = 2 * enc_units + dec_units

        self.fc = nn.Linear(self.attn_in, units)
        self.fc2 = nn.Linear(units, 1)
        self.dropout = nn.Dropout(dropout)

        utils.init_weights_(self.fc)
        utils.init_weights_(self.fc2)

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
        scores = torch.tanh(self.fc(combined))                                          # (len, batch, units)
        scores = self.fc2(scores).squeeze(-1)
        # scores = torch.sum(scores, dim=2)

        weights = softmax(scores, dim=1)
        return weights


class Decoder(nn.Module):
    def __init__(self, embed_weights: Tensor, units: int, dropout: float, enc_units: int) -> None:
        """
        :param embed_weights: Tensor
            Embedding weights of shape (trg_vocab_size, embed_dim)
        :param units: int
        :param dropout: float
        """
        super().__init__()
        self.vocab_size, embed_dim = embed_weights.shape

        self.embedding = nn.Embedding.from_pretrained(embed_weights)                    #
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