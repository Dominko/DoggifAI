import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import TransformerDecoder, TransformerEncoder
from torch.nn import functional as F
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src.configs import ModelConfigs
from src.constants import AA_DICT
from src.utils.model_utils import generate_square_subsequent_mask

import random

# import torchaudio
# from torchaudio.models.decoder import ctc_decoder

from src.models.decoders.beam_search import beam_search

import math

import numpy as np

class DoggyT5Simple(nn.Module):
    def __init__(self, model_configs: ModelConfigs, tokenizer: PreTrainedTokenizer, device=None, **kwargs):
        super().__init__()
        self.input_length = model_configs.hyperparameters.max_seq_len
        self.output_length = kwargs["target_length"]

        self.embedding_dim = model_configs.hyperparameters.embedding_dim
        self.hidden_dim = model_configs.hyperparameters.hidden_dim
        self.nhead = model_configs.hyperparameters.nhead
        self.num_layers = model_configs.hyperparameters.num_layers
        self.dropout = model_configs.hyperparameters.dropout
        self.ff_size = model_configs.hyperparameters.ff_size

        self.tokenizer = tokenizer

        # self.vocab_size = kwargs["vocab_size"] - 1
        self.vocab_size = kwargs["vocab_size"]

        self.padding_idx = kwargs["padding_idx"]
        print(self.padding_idx)
        self.bos_idx = kwargs["bos_idx"]
        self.eos_idx = kwargs["eos_idx"]

        self.device = device

        self._build_model()

    def _build_model(self):
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx = self.padding_idx)
        # self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)

        # self.output_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)

        self.positional_embedding = PositionalEncoder(
            self.hidden_dim, self.dropout, self.input_length, self.device
        )

        # self.output_positional_embedding = PositionalEncoder(
        #     self.hidden_dim, self.dropout, self.input_length, self.device
        # )
        
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            num_decoder_layers=self.num_layers,
            num_encoder_layers=self.num_layers,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            batch_first=True
        )

        self.linear = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, input_sequence, output_sequence, mask=None):

        input_embedded = self.embedding(input_sequence) 
        input_embedded =  math.sqrt(self.hidden_dim) * input_embedded
        input_positioned = self.positional_embedding(input_embedded)

        output_embedded = self.embedding(output_sequence) * math.sqrt(self.hidden_dim)
        output_positioned = self.positional_embedding(output_embedded)

        input_pad_mask = input_sequence == self.padding_idx
        output_pad_mask = output_sequence == self.padding_idx

        mask = nn.Transformer.generate_square_subsequent_mask(output_sequence.size(1)).to(self.device)

        output_decoded = self.transformer(
            input_positioned,
            output_positioned,
            tgt_mask=mask,
            src_key_padding_mask=input_pad_mask,
            tgt_key_padding_mask=output_pad_mask
        )

        out = self.linear(output_decoded)

        # out = output_decoded @ self.output_embedding.weight.T

        return out

    def step(self, input_sequence, output_sequence, debug=False):
        input = input_sequence[:, :-1]
        output = output_sequence[:, :-1]
        target = output_sequence[:, 1:].contiguous().view(-1)
        # mask = generate_square_subsequent_mask(input.size(1)).to(self.device)
        mask = nn.Transformer.generate_square_subsequent_mask(input.size(1)).to(self.device)

        # print(input[0])
        # print(output[0])
        # print(mask[0])
        generated_sequences = self.forward(input, output, mask)
        # print(generated_sequences[0])
        generated_sequences = generated_sequences.view(-1, self.vocab_size)
        # print(generated_sequences[0])

        # if(debug):
        # print(generated_sequences.shape)
        # print(target.shape)

        # raise Exception()

        # loss = F.cross_entropy(generated_sequences, target, ignore_index=self.padding_idx)
        loss = F.cross_entropy(generated_sequences, target, ignore_index=self.padding_idx)
        # print(loss)
        # raise Exception()

        return {"loss": loss, 
                "perplexity": torch.exp(loss)}

    def generate_sequences(
        self, num_sequences_per_input, inputs, y_init=None, temperature=1.0, batch_size=None, topk=1, beam_width=5, trim_to_eos=True, sample_method="topk"
    ):
        self.eval()
        
        if y_init == None:
            y_init = torch.ones((len(inputs), 1), dtype=torch.long) * self.bos_idx

        if sample_method == "topk":
            samples, probabilities = self.topk_generator(num_sequences_per_input, 
                                       inputs, 
                                       y_init,
                                       temperature = temperature, 
                                       batch_size = batch_size, 
                                       topk = topk, 
                                       trim_to_eos = trim_to_eos
                                       )
        elif sample_method == "beam":
            samples, probabilities = self.beam_generator(num_sequences_per_input, 
                                       inputs, 
                                       y_init,
                                       temperature = temperature, 
                                       batch_size = batch_size, 
                                       beam_width = beam_width, 
                                       trim_to_eos = trim_to_eos
                                       )
        else:
            raise NotImplementedError()
        
        if trim_to_eos:
            samples = samples.detach().cpu().numpy()
            for i in range(0, len(samples)):
                # Fidn the eos token
                indices = np.argwhere(samples[i]==self.eos_idx)
                if len(indices) == 0:
                    continue
                idx = indices[0][0]
                samples[i][idx+1:] = self.padding_idx
            
            samples = torch.from_numpy(samples).to(self.device)

        return samples, probabilities
    
    def beam_generator(
            self, num_sequences_per_input, inputs, y_inits, temperature=1.0, batch_size=None, beam_width=1, trim_to_eos=True
    ):
        y_inits = y_inits.to(self.device)
        inputs = inputs.to(self.device)
        outputs = None
        probabilities = None
        for i in range(len(inputs)):
            input = inputs[i].repeat(num_sequences_per_input, 1)
            y_init = y_inits[i].repeat(num_sequences_per_input, 1)
            
            out, probability = beam_search(self, input, Y_init = y_init, predictions=self.input_length, beam_width=beam_width, temperature=temperature)
            out = out.squeeze(0)
            probability = probability.squeeze(0)

            if outputs == None:
                outputs = out[:num_sequences_per_input]
                probabilities = probability[:num_sequences_per_input]
            else:
                outputs = torch.cat((outputs, out[:num_sequences_per_input]), axis=0)
                probabilities = torch.cat((probabilities, probability[:num_sequences_per_input]), axis=0)

        return outputs, probabilities

    def topk_generator(
            self, num_sequences_per_input, inputs, y_init, temperature=1.0, batch_size=None, topk=1, trim_to_eos=True
    ):
        with torch.no_grad():
            num_sequences = num_sequences_per_input * len(inputs)
            # padding is all ones
            samples = torch.ones(num_sequences, self.input_length, dtype=torch.int32).to(self.device)
            probabilities = torch.zeros(num_sequences, 1, dtype=torch.float).to(self.device)

            if batch_size is None:
                batch_size = num_sequences

            if batch_size > num_sequences:
                batch_size = num_sequences

            for idx in range(0, num_sequences, batch_size):
                input_sequences = y_init.to(self.device)
                # input_sequences = inputs[:,0].unsqueeze(
                #     dim=1
                # )
                # input_sequences = input_sequences.to(self.device)

                inputs = inputs.to(self.device)
                for i in range(self.input_length):
                    # print(random.getstate())
                    out = self.forward(inputs, input_sequences)
                    out = out[:, -1, :] / temperature
                    out = F.softmax(out, dim=-1)
                    # out = F.log_softmax(out, dim=-1)

                    out = torch.topk(out, topk)

                    new_input_sequences_i = torch.multinomial(out.values, num_samples=1)
                    
                    # new_input_sequences = out.indices[0, new_input_sequences_i]
                    new_input_sequences = out.indices[np.arange(out.indices.shape[0]), new_input_sequences_i.squeeze()]
                    probability = torch.log(out.values[np.arange(out.values.shape[0]), new_input_sequences_i.squeeze()])

                    new_input_sequences = new_input_sequences.reshape(-1, 1)
                    probability = probability.reshape(-1, 1)
                    
                    samples[idx : idx + batch_size, i] = new_input_sequences.squeeze()
                    probabilities[idx : idx + batch_size] += probability
                    input_sequences = torch.cat(
                        (input_sequences, new_input_sequences), dim=1
                    )
            
            return samples, probabilities

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