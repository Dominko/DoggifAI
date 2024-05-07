import numpy as np
import torch

from ..configs import DatasetConfigs
from ..constants import START_TOKEN, AA_DICT
from .bpe_tokenizer import BPE_Tokenizer
from .tokenizer import Tokenizer

from transformers import PreTrainedTokenizer, DefaultDataCollator
import datasets

def load_sequences_file(filename):
    with open(filename, "r") as file:
        viral_seqs = file.readlines()
    return [viral_seq.replace("\n", "") for viral_seq in viral_seqs]

class SequenceDataset:
    def __init__(
        self,
        dataset_configs: DatasetConfigs,
        tokenizer: PreTrainedTokenizer,
        split: str,
        max_seq_len: int,
        target_length: int,
        sequence_one_hot: bool = True,
        label_one_hot: bool = True,
        prepend_start_token: bool = False,
    ):
        self.max_seq_len = max_seq_len
        self.sequence_one_hot = sequence_one_hot
        self.label_one_hot = label_one_hot
        self.prepend_start_token = prepend_start_token
        self.tokenizer = tokenizer

        # Intialise datasets
        if split == "train":
            input_sequences_path = dataset_configs.train.input_sequences_path
            output_sequences_path = dataset_configs.train.output_sequences_path
        elif split == "val":
            input_sequences_path = dataset_configs.val.input_sequences_path
            output_sequences_path = dataset_configs.val.output_sequences_path
        elif split == "test":
            input_sequences_path = dataset_configs.test.input_sequences_path
            output_sequences_path = dataset_configs.test.output_sequences_path

        self.input_sequences = datasets.load_dataset("csv", data_files=input_sequences_path)["train"]
        self.output_sequences = datasets.load_dataset("csv", data_files=output_sequences_path)["train"]

        # self.input_sequences = load_sequences_file(input_sequences_path)
        # self.output_sequences = load_sequences_file(output_sequences_path)

        self.sequences = self.input_sequences.map(
                tokenize_function,
                batched=True,
                fn_kwargs={
                    'tokenizer': self.tokenizer,
                    'prepend_start_token': False,
                    'max_seq_len': self.max_seq_len,
                },
                remove_columns=['text'],
                load_from_cache_file=False
            )
        
        self.output_sequences = self.output_sequences.map(
                tokenize_function,
                batched=True,
                fn_kwargs={
                    'tokenizer': self.tokenizer,
                    'prepend_start_token': self.prepend_start_token,
                    'max_seq_len': self.max_seq_len,
                },
                remove_columns=['text'],
                load_from_cache_file=False
            )
        
        self.sequences = self.sequences.add_column("output_ids", self.output_sequences["input_ids"])

        self.data_collator = DefaultDataCollator()

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def getCollator(self):
        return self.data_collator
    
def tokenize_function(examples, tokenizer, prepend_start_token, max_seq_len):
    tokenizer_out = tokenizer(
        text=examples["text"],
        return_attention_mask=False
    )

    # Pad
    for i in range(0, len(tokenizer_out["input_ids"])):
        # Remove fantom whitespace
        tokenizer_out["input_ids"][i] = tokenizer_out["input_ids"][i][1:]

        if prepend_start_token:
            tokenizer_out["input_ids"][i].insert(0, tokenizer.bos_token_id)
        
        padding_len = max_seq_len - len(tokenizer_out["input_ids"][i])
        tokenizer_out["input_ids"][i] = list(np.pad(tokenizer_out["input_ids"][i], pad_width=(0, padding_len), mode="constant", constant_values=tokenizer.pad_token_id))

    return tokenizer_out