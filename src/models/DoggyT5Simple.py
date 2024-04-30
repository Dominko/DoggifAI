import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import TransformerDecoder, TransformerEncoder
from torch.nn import functional as F
from tqdm import tqdm

from src.configs import ModelConfigs
from src.constants import AA_DICT
from src.utils.model_utils import generate_square_subsequent_mask

import numpy as np

class DoggyT5Simple(nn.Module):
    def __init__(self, model_configs: ModelConfigs, device=None, **kwargs):
        super().__init__()
        self.input_length = model_configs.hyperparameters.max_seq_len
        self.output_length = kwargs["target_length"]

        self.embedding_dim = model_configs.hyperparameters.embedding_dim
        self.hidden_dim = model_configs.hyperparameters.hidden_dim
        self.nhead = model_configs.hyperparameters.nhead
        self.num_layers = model_configs.hyperparameters.num_layers
        self.dropout = model_configs.hyperparameters.dropout

        self.vocab_size = kwargs["vocab_size"]

        self.padding_idx = kwargs["padding_idx"]
        self.start_idx = kwargs["start_idx"]
        self.end_idx = kwargs["end_idx"]

        self.device = device

        self._build_model()

    def _build_model(self):
        self.input_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)

        self.output_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)

        self.input_positional_embedding = PositionalEncoder(
            self.hidden_dim, self.dropout, self.input_length, self.device
        )

        self.output_positional_embedding = PositionalEncoder(
            self.hidden_dim, self.dropout, self.input_length, self.device
        )
        
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            num_decoder_layers=self.num_layers,
            num_encoder_layers=self.num_layers,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            batch_first=True
        )

    def forward(self, input_sequence, output_sequence, mask=None):

        input_embedded = self.input_embedding(input_sequence)
        input_positioned = self.input_positional_embedding(input_embedded)

        output_embedded = self.input_embedding(output_sequence)
        output_positioned = self.output_positional_embedding(output_embedded)

        output_decoded = self.transformer(
            input_positioned,
            output_positioned,
            tgt_mask=mask
        )

        out = output_decoded @ self.output_embedding.weight.T

        return out

    def step(self, input_sequence, output_sequence):
        input = input_sequence[:, :-1]
        output = output_sequence[:, :-1]
        target = output_sequence[:, 1:].contiguous().view(-1)
        mask = generate_square_subsequent_mask(input.size(1)).to(self.device)

        generated_sequences = self.forward(input, output, mask)
        generated_sequences = generated_sequences.view(-1, self.vocab_size)

        # print(generated_sequences)
        # print(target)

        # raise Exception()

        loss = F.cross_entropy(generated_sequences, target, ignore_index=self.padding_idx)

        return {"loss": loss, 
                "perplexity": torch.exp(loss)}

    def generate_sequences(
        self, num_sequences, inputs, temperature=1.0, batch_size=None, topk=5, trim_to_eos=True
    ):
        self.eval()
        # padding is all ones
        samples = torch.ones(num_sequences, self.input_length).to(self.device)

        if batch_size is None:
            batch_size = num_sequences

        if batch_size > num_sequences:
            batch_size = num_sequences

        for idx in range(0, num_sequences, batch_size):
            if self.start_idx == 0:
                input_sequences = torch.LongTensor([self.start_idx] * batch_size).unsqueeze(
                    dim=1
                )
            else:
                input_sequences = inputs[:,0].unsqueeze(
                    dim=1
                )
            input_sequences = input_sequences.to(self.device)

            inputs = inputs.to(self.device)

            for i in range(self.input_length):
                out = self.forward(inputs, input_sequences)
                out = out[:, -1, :] / temperature
                out = F.softmax(out, dim=-1)

                out = torch.topk(out, topk)

                new_input_sequences_i = torch.multinomial(out.values, num_samples=1)
                new_input_sequences = out.indices[0, new_input_sequences_i]

                samples[idx : idx + batch_size, i] = new_input_sequences.squeeze()
                input_sequences = torch.cat(
                    (input_sequences, new_input_sequences), dim=1
                )

        if trim_to_eos:
            samples = samples.detach().cpu().numpy()
            for i in range(0, len(samples)):
                # Fidn the eos token
                indices = np.argwhere(samples[i]==self.end_idx)
                if len(indices) == 0:
                    continue
                idx = indices[0][0]
                samples[i][idx+1:] = self.padding_idx
            
            samples = torch.from_numpy(samples).to(self.device)

        return samples

class PositionalEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout, max_seq_len=1300, device=None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position = torch.arange(max_seq_len).unsqueeze(1)
        self.positional_encoding = torch.zeros(1, max_seq_len, hidden_dim)

        _2i = torch.arange(0, hidden_dim, step=2).float()
        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 0::2] = torch.sin(
            self.position / (10000 ** (_2i / hidden_dim))
        )
        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 1::2] = torch.cos(
            self.position / (10000 ** (_2i / hidden_dim))
        )

        self.device = device

    def forward(self, x):
        # batch_size, input_len, embedding_dim
        batch_size, seq_len, _ = x.size()

        position_encoding = self.positional_encoding[:batch_size, :seq_len, :].to(
            self.device
        )

        x += position_encoding

        return self.dropout(x)