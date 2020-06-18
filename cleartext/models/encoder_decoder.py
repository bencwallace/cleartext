from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch import Tensor

from .components import Encoder, Attention, Decoder


class EncoderDecoder(nn.Module):
    def __init__(self, device: torch.device,
                 embed_weights_src: Tensor, embed_weights_trg: Tensor,
                 rnn_units: int, attn_units: int,
                 dropout: float) -> None:
        """
        :param device: torch.device
        :param embed_weights_src: Tensor
            Embedding weights of shape (src_vocab_size, embed_dim)
        :param rnn_units: int
        :param attn_units: int
        :param dropout: float
        """
        super().__init__()
        self.device = device
        self.trg_vocab_size = embed_weights_trg.shape[0]

        self.encoder = Encoder(embed_weights_src, rnn_units, dropout)
        self.decoder = Decoder(embed_weights_trg, rnn_units, dropout, rnn_units)
        self.attention = Attention(rnn_units, rnn_units, attn_units)

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

        outputs = torch.zeros(max_len, batch_size, self.trg_vocab_size, device=self.device)
        out = target[0, :]
        for t in range(1, max_len):
            context = self._compute_context(state, enc_outputs)
            out, state = self.decoder(out, state, context)
            outputs[t] = out
            teacher_force = torch.bernoulli(torch.tensor(teacher_forcing, dtype=torch.float)).item()
            out = (target[t] if teacher_force else out.max(1)[1])

        return outputs

    def beam_search(self, source: Tensor, beam_size: int, trg_sos: int, max_len: int, trg_unk: int) -> Tuple[Tensor, Tensor]:
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
                # mask out <unk>
                probs[0, trg_unk] = 0

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
