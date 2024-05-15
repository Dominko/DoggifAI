import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.autograd import Variable
import wandb

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

from src.configs import TestingConfigs, TrainingConfigs
from src.constants import START_TOKEN
from src.dataset.dataset import SequenceDataset
from src.dataset.dataset_seq_2_seq import SequenceDatasetS2S
from src.models.DoggyTransformer import DoggyTransformer
# from src.models.vaxformer import Vaxformer
from src.utils import common_utils, model_utils

from transformers import T5Tokenizer

from src.utils.DataCollatorForT5MLM import compute_t5_input_and_target_lengths

from src.sampler import Sampler

def argument_parser():
    parser = argparse.ArgumentParser(
        description="Doggy AI project"
    )
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--sequences_per_input", type=int, required=True)
    parser.add_argument("--log_to_wandb", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    run_name = os.path.basename(args.config_filepath).replace(".yaml", "")
    configs = TestingConfigs(**common_utils.load_yaml(args.config_filepath))
    train_configs = TrainingConfigs(
        **common_utils.load_yaml(configs.pretrained_model_configs_path)
    )

    print(configs.top_k)

    sequences_per_input = args.sequences_per_input

    common_utils.setup_random_seed(configs.random_seed)
    outputs_dir = common_utils.setup_experiment_folder(
        os.path.join(os.getcwd(), configs.outputs_dir)
    )
    device = common_utils.setup_device(configs.device)
    print(f"Running on {device}")

    wandb.init(
        project=train_configs.training_configs.wandb_project,
        name=train_configs.training_configs.wandb_name,
        entity="dominik-grabarczyk",
        mode="online" if args.log_to_wandb else "disabled",
    )
    wandb.config.update(train_configs.dict())
    wandb.config.update(
        {"outputs_dir": outputs_dir, "device_count": torch.cuda.device_count()}
    )

    full_filename = os.path.join(
            outputs_dir, f"{run_name}_samples"
        )

    if train_configs.model_configs.model_type in [
        "DoggyTransformer", "t5", "t5_simple"
    ]:
        sequence_one_hot = False
        label_one_hot = False
        if train_configs.training_configs.training_type == "ft":
            prepend_start_token = False
        else: 
            prepend_start_token = False

    expanded_inputs_length, target_length = compute_t5_input_and_target_lengths(
            inputs_length=train_configs.model_configs.hyperparameters.max_seq_len,
            noise_density=train_configs.dataset_configs.corrupted_percentage,
            mean_noise_span_length=train_configs.dataset_configs.mean_noise_span_length,
        )
    
    run_validation = configs.run_validation
    
    # Intialise tokeniser
    if train_configs.model_configs.tokenizer == "Base":
        tokenizer = T5Tokenizer(
            vocab_file=train_configs.model_configs.tokenizer_path,
            extra_ids=train_configs.dataset_configs.extra_ids,
            legacy=True,
            bos_token="<s>"
        )
    if tokenizer == "BPE":
        # TODO: Implement
        raise NotImplementedError()

    if tokenizer == "ProtGPT2":
        # TODO: Implement
        raise NotImplementedError()

    sample_dataset = setupDataset(
        configs.sample_conf,
        tokenizer,
        "sample",
        train_configs.model_configs.hyperparameters.max_seq_len,
        target_length,
        sequence_one_hot,
        label_one_hot,
        prepend_start_token,
        train_configs.training_configs.training_type,
    )

    sampler = Sampler(
        configs,
        train_configs,
        full_filename,
        sequences_per_input,
        run_validation,
        sample_dataset,
        outputs_dir,
        prepend_start_token,
        tokenizer,
        target_length,
        device=device,
    )

    sampler.sample()

def setupDataset(dataset_configs, 
                 tokenizer, 
                 split, 
                 max_seq_len, 
                 target_length,
                 sequence_one_hot,
                 label_one_hot,
                 prepend_start_token,
                 mode
                 ):
    if mode == "pt":
        return SequenceDatasetS2S(
            dataset_configs,
            tokenizer,
            split,
            max_seq_len,
            target_length,
            sequence_one_hot,
            label_one_hot,
            prepend_start_token=prepend_start_token,
        )
    else:
        return SequenceDataset(
            dataset_configs,
            tokenizer,
            split,
            max_seq_len,
            target_length,
            sequence_one_hot,
            label_one_hot,
            prepend_start_token=prepend_start_token,
        )


if __name__ == "__main__":
    main()
