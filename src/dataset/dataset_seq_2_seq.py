import numpy as np
import torch
import datasets

from ..configs import DatasetConfigs
from ..constants import START_TOKEN, AA_DICT
from .bpe_tokenizer import BPE_Tokenizer
from .tokenizer import Tokenizer
from ..utils.DataCollatorForT5MLM import DataCollatorForT5MLM


from transformers.utils.generic import PaddingStrategy

from transformers import PreTrainedTokenizer

def load_sequences_file(filename):
    with open(filename, "r") as file:
        viral_seqs = file.readlines()
    return np.array([viral_seq.replace("\n", "") for viral_seq in viral_seqs])

class SequenceDatasetS2S:
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
            output_sequences_path = dataset_configs.train.output_sequences_path
        elif split == "val":
            output_sequences_path = dataset_configs.val.output_sequences_path
        elif split == "test":
            output_sequences_path = dataset_configs.test.output_sequences_path
        elif split == "sample":
            output_sequences_path = dataset_configs.sample.output_sequences_path

        # self.sequences = load_sequences_file(output_sequences_path)
        self.sequences = datasets.load_dataset("csv", data_files=output_sequences_path)["train"]


        # print(self.sequences)

        # print(target_length)
        # print(self.sequences[0]["text"])

        self.sequences = self.sequences.map(
                tokenize_function,
                batched=True,
                fn_kwargs={
                    'tokenizer': self.tokenizer,
                    'prepend_start_token': self.prepend_start_token,
                },
                remove_columns=['text']
            )
        
        # print(self.sequences[0]["input_ids"])
        # print("\r\n" + self.tokenizer.decode(self.sequences[0]["input_ids"]) + "\r\n")
        # raise NotImplementedError()

        # TODO: Consider whether we want the target length to be max_seq_len or the "target lenght" at PT

        # self.data_collator = DataCollatorForT5MLM(
        #     tokenizer=self.tokenizer,
        #     noise_density=dataset_configs.corrupted_percentage,
        #     mean_noise_span_length=dataset_configs.mean_noise_span_length,
        #     input_length=self.max_seq_len,
        #     target_length=target_length,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     decoder_start_token_id=self.tokenizer.bos_token_id,
        # )


        self.data_collator = DataCollatorForT5MLM(
            tokenizer=self.tokenizer,
            noise_density=dataset_configs.corrupted_percentage,
            mean_noise_span_length=dataset_configs.mean_noise_span_length,
            input_length=self.max_seq_len,
            target_length=self.max_seq_len,
            pad_token_id=self.tokenizer.pad_token_id,
            max_sentinel_token_idx = self.tokenizer.additional_special_tokens_ids[0],
            decoder_start_token_id=self.tokenizer.bos_token_id,
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def getCollator(self):
        return self.data_collator
        
def tokenize_function(examples, tokenizer, prepend_start_token):
        tokenizer_out = tokenizer(
            text=examples["text"],
            return_attention_mask=False,
            # max_length=in_length,
            # pad_to_multiple_of=in_length,
            # padding=PaddingStrategy.DO_NOT_PAD,
        )
        

        for i in range(0, len(tokenizer_out["input_ids"])):
            # Remove fantom whitespace
            tokenizer_out["input_ids"][i] = tokenizer_out["input_ids"][i][1:]

            if prepend_start_token:
                tokenizer_out["input_ids"][i].insert(0, tokenizer.bos_token_id)


        # NOTE: This seems to squish text to match the desired length but without regard for sequence coherence

        # input_ids = tokenizer_out["input_ids"]

        # concatenated_ids = np.concatenate(input_ids)

        # total_length = concatenated_ids.shape[0]
        # total_length = (total_length // in_length) * in_length

        # concatenated_ids = concatenated_ids[:total_length].reshape(-1, in_length)
        # result = {"input_ids": concatenated_ids}

        # return result

        return tokenizer_out