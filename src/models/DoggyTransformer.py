import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import TransformerDecoder, TransformerEncoder
from torch.nn import functional as F
from tqdm import tqdm

from src.configs import ModelConfigs
from src.constants import AA_DICT
from src.utils.model_utils import generate_square_subsequent_mask

class DoggyTransformer(nn.Module):
    def __init__(self, model_configs: ModelConfigs, device=None, **kwargs):
        super().__init__()
        self.max_seq_len = model_configs.hyperparameters.max_seq_len
        self.embedding_dim = model_configs.hyperparameters.embedding_dim
        self.hidden_dim = model_configs.hyperparameters.hidden_dim
        self.nhead = model_configs.hyperparameters.nhead
        self.num_layers = model_configs.hyperparameters.num_layers
        self.dropout = model_configs.hyperparameters.dropout

        tokenizer = model_configs.tokenizer

        # include start token in the vocab size
        if tokenizer == "Base":
            self.vocab_size = len(AA_DICT) + 1
        elif tokenizer == "BPE":
            tokenizer = Tokenizer.from_file(model_configs.tokenizer_path)
            self.vocab_size = len(tokenizer.get_vocab())
        else:
            raise NotImplementedError("Tokeniser not implemented")

        # self.padding_idx = kwargs["padding_idx"]
        self.start_idx = kwargs["start_idx"]

        self.device = device

        self._build_model()

    def _build_model(self):
        self.input_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)

        self.output_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)

        # transformer_input_dim = self.hidden_dim + self.immunogenicity_size * self.nhead
        transformer_input_dim = self.hidden_dim

        transformer_output_dim = self.hidden_dim
        # self.immunogenicity_embedding = nn.Embedding(
        #     self.immunogenicity_size, transformer_input_dim
        # )
        # self.immunogenicity_value_embedding = nn.Embedding(
        #     self.immunogenicity_size, self.immunogenicity_size * self.nhead
        # )

        self.input_positional_embedding = PositionalEncoder(
            self.hidden_dim, self.dropout, self.max_seq_len, self.device
        )

        self.output_positional_embedding = PositionalEncoder(
            self.hidden_dim, self.dropout, self.max_seq_len, self.device
        )

        
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim, nhead=self.nhead, batch_first=True
        )

        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_output_dim, nhead=self.nhead, batch_first=True
        )

        input_layer_norm = nn.LayerNorm(transformer_input_dim)
        
        self.transformer_encoder = TransformerEncoder(
            transformer_encoder_layer, num_layers=self.num_layers, norm=input_layer_norm
        )

        output_layer_norm = nn.LayerNorm(transformer_input_dim)

        self.transformer_decoder = TransformerDecoder(
            transformer_decoder_layer, num_layers=self.num_layers, norm=output_layer_norm
        )

        self.projection = nn.Linear(transformer_input_dim, self.hidden_dim)

    def forward(self, input_sequence, output_sequence, mask=None):

        input_embedded = self.input_embedding(input_sequence)
        input_positioned = self.input_positional_embedding(input_embedded)

        output_embedded = self.input_embedding(output_sequence)
        output_positioned = self.output_positional_embedding(output_embedded)

        input_encoded = self.transformer_encoder(
            input_positioned, mask=mask
        )

        output_decoded = self.transformer_decoder(
            output_positioned, memory=input_encoded, tgt_mask=mask
        )
        output_decoded = self.projection(output_decoded)
        out = output_decoded @ self.output_embedding.weight.T

        return out

    def step(self, input_sequence, output_sequence):
        input = input_sequence[:, :-1]
        output = output_sequence[:, :-1]
        target = output_sequence[:, 1:].contiguous().view(-1)
        mask = generate_square_subsequent_mask(input.size(1)).to(self.device)

        generated_sequences = self.forward(input, output, mask)
        generated_sequences = generated_sequences.view(-1, self.vocab_size)

        loss = F.cross_entropy(generated_sequences, target)
        return {"loss": loss, "perplexity": torch.exp(loss)}

    def generate_sequences(
        self, num_sequences, inputs, temperature=1.0, batch_size=None, topk=5
    ):
        self.eval()
        # padding is all ones
        samples = torch.ones(num_sequences, self.max_seq_len).to(self.device)

        if batch_size is None:
            batch_size = num_sequences

        if batch_size > num_sequences:
            batch_size = num_sequences

        for idx in tqdm(range(0, num_sequences, batch_size)):

            input_sequences = torch.LongTensor([self.start_idx] * batch_size).unsqueeze(
                dim=1
            )
            input_sequences = input_sequences.to(self.device)

            inputs = inputs.repeat(batch_size, 1).to(self.device)

            for i in tqdm(range(self.max_seq_len)):
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