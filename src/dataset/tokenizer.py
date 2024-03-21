import numpy as np
import torch
from torch.nn import functional as F

from ..constants import START_TOKEN, AA_DICT


class Tokenizer:
    def __init__(
        self,
        max_seq_len: int,
        one_hot=True,
        prepend_start_token=False,
        base_vocab:dict=None
    ):
        self.max_seq_len = max_seq_len
        self.prepend_start_token = prepend_start_token

        if prepend_start_token:
            self.enc_dict = {
                letter: idx for idx, letter in enumerate([START_TOKEN] + list(AA_DICT.keys()))
            }
            # self.max_seq_len += 1
            self.dec_dict = {
                idx: amino_acid for amino_acid, idx in AA_DICT.items()
            }
        else:
            self.enc_dict = AA_DICT
        self.dec_dict = {idx: amino_acid for amino_acid, idx in self.enc_dict.items()}
        self.one_hot = one_hot

    def encode(self, sequence, is_output):
        enc = []
        # sequence = sequence.ljust(self.max_seq_len, "-")
        # for aa in sequence[: self.max_seq_len]:
        if self.prepend_start_token and is_output:
            enc.append(self.enc_dict[sequence[0]])
            sequence = sequence[1:]

        for a in [sequence[i] for i in range(0, len(sequence))]:
            enc.append(self.enc_dict[a])

        enc += [self.enc_dict[""]] * (self.max_seq_len - len(enc)) # Add padding

        # print(len(enc))

        if self.one_hot:
            return F.one_hot(torch.tensor(enc), len(self.enc_dict)).float()
        else:
            return torch.tensor(enc)
        
    def batch_encode(self, batch):
        # TODO make efficient
        out = []
        for seq in batch:
            out.append(self.encode(seq))
        return out

    def decode(self, batch):
        sequence_size = batch.size()
        batch_size = sequence_size[0]
        seq_len = sequence_size[1]

        batch_seq = []

        if self.one_hot:
            h = torch.max(batch, dim=-1).indices
        else:
            h = batch

        for batch_idx in range(batch_size):
            seq = ""
            for seq_idx in range(seq_len):
                seq += self.dec_dict[int(h[batch_idx][seq_idx])]
            batch_seq.append(seq)

        return batch_seq
