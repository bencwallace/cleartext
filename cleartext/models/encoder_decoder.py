import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embed_weights, units, dropout):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embed_weights)
        embed_dim = embed_weights.shape[1]
        self.gru = nn.GRU(embed_dim, units, bidirectional=True)
        self.fc = nn.Linear(units * 2, units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source):
        embedded = self.embedding(source)
        # todo: ensure state properly initialized
        outputs, state = self.gru(embedded)

        # combine and reshape bidirectional states for compatibility with (unidirectional) decoder
        combined = torch.cat((state[-2, :, :], state[-1, :, :]), dim=1)
        combined = self.dropout(combined)
        state = torch.tanh(self.fc(combined))

        return outputs, state


class Attention(nn.Module):
    def __init__(self, state_dim, units, dropout=0.2):
        super().__init__()
        self.attn_in = state_dim * 3
        self.fc = nn.Linear(self.attn_in, units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_state, enc_outputs):
        source_len = enc_outputs.shape[0]

        # vectorize computation of Bahdanau attention scores for all encoder outputs
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
    def __init__(self, embed_weights, units, dropout, attention):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embed_weights)
        self.vocab_size, embed_dim = embed_weights.shape
        self.rnn = nn.GRU((units * 2) + embed_dim, units)
        self.fc = nn.Linear(attention.attn_in + embed_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, token, dec_state, enc_outputs):
        token = token.unsqueeze(0)
        embedded = self.embedding(token)

        context = self._compute_context(dec_state, enc_outputs)
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
        return output, dec_state.squeeze(0)

    def _compute_context(self, dec_state, enc_outputs):
        weights = self.attention(dec_state, enc_outputs)
        weights = weights.unsqueeze(1)

        enc_outputs = enc_outputs.permute(1, 0, 2)
        context = torch.bmm(weights, enc_outputs)
        context = context.permute(1, 0, 2)
        return context


class EncoderDecoder(nn.Module):
    def __init__(self, device, embed_weights_src, embed_weights_trg, rnn_units, attn_units, dropout):
        super().__init__()
        self.encoder = Encoder(embed_weights_src, rnn_units, dropout)
        self.attention = Attention(rnn_units, attn_units)
        self.decoder = Decoder(embed_weights_trg, rnn_units, dropout, self.attention)

        self.device = device
        self.target_vocab_size = self.decoder.vocab_size

    # todo: implement beam search
    def forward(self, source, target, teacher_forcing=0.5):
        batch_size = source.shape[1]
        # todo: change max_len
        max_len = target.shape[0]
        enc_outputs, state = self.encoder(source)

        outputs = torch.zeros(max_len, batch_size, self.target_vocab_size).to(self.device)
        out = target[0, :]
        for t in range(1, max_len):
            out, state = self.decoder(out, state, enc_outputs)
            # todo: break if EOS found
            outputs[t] = out
            teacher_force = random.random() < teacher_forcing
            out = (target[t] if teacher_force else out.max(1)[1])

        return outputs
