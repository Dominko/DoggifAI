import argparse
import os
import sys

import torch

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv("env/.env")

import wandb
from src.configs import TrainingConfigs
from src.dataset.dataset import SequenceDataset
from src.dataset.dataset_seq_2_seq import SequenceDatasetS2S
from src.trainer import Trainer
from src.utils import common_utils

from transformers import T5Tokenizer

from src.utils.DataCollatorForT5MLM import compute_t5_input_and_target_lengths


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Doggy AI project"
    )
    parser.add_argument("--config_filepath", type=str, required=True)
    parser.add_argument("--log_to_wandb", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    configs = TrainingConfigs(**common_utils.load_yaml(args.config_filepath))

    common_utils.setup_random_seed(configs.training_configs.random_seed)
    outputs_dir = common_utils.setup_experiment_folder(
        os.path.join(os.getcwd(), configs.training_configs.outputs_dir)
    )
    common_utils.save_training_configs(configs, outputs_dir)
    device = common_utils.setup_device(configs.training_configs.device)
    print(f"Running on {device}")

    wandb.init(
        project=configs.training_configs.wandb_project,
        name=configs.training_configs.wandb_name,
        mode="online" if args.log_to_wandb else "disabled",
    )
    if configs.model_configs.model_id == None:
        wandb.init(
            project=configs.training_configs.wandb_project,
            name=configs.training_configs.wandb_name,
            mode="online" if args.log_to_wandb else "disabled",
        )
    else:
        wandb.init(
            project=configs.training_configs.wandb_project,
            name=configs.training_configs.wandb_name,
            mode="online" if args.log_to_wandb else "disabled",
            id=configs.model_configs.model_id, 
            resume="must"
        )
    
    wandb.config.update(configs.dict())
    wandb.config.update(
        {"outputs_dir": outputs_dir, "device_count": torch.cuda.device_count()}
    )

    if configs.model_configs.model_type in [
        "DoggyTransformer", "t5", "t5_simple"
    ]:
        sequence_one_hot = False
        label_one_hot = False
        if configs.training_configs.training_type == "ft":
            prepend_start_token = False
        else: 
            prepend_start_token = False

    expanded_inputs_length, target_length = compute_t5_input_and_target_lengths(
            inputs_length=configs.model_configs.hyperparameters.max_seq_len,
            noise_density=configs.dataset_configs.corrupted_percentage,
            mean_noise_span_length=configs.dataset_configs.mean_noise_span_length,
        )

    # Intialise tokeniser
    if configs.model_configs.tokenizer == "Base":
        tokenizer = T5Tokenizer(
            vocab_file=configs.model_configs.tokenizer_path,
            extra_ids=configs.dataset_configs.extra_ids,
            legacy=False,
            bos_token="<s>"
        )
    if tokenizer == "BPE":
        # TODO: Implement
        raise NotImplementedError()

    if tokenizer == "ProtGPT2":
        # TODO: Implement
        raise NotImplementedError()

    train_dataset = setupDataset(
        configs.dataset_configs,
        tokenizer,
        "train",
        configs.model_configs.hyperparameters.max_seq_len,
        target_length,
        sequence_one_hot,
        label_one_hot,
        prepend_start_token,
        configs.training_configs.training_type,
    )
    val_dataset = setupDataset(
        configs.dataset_configs,
        tokenizer,
        "val",
        configs.model_configs.hyperparameters.max_seq_len,
        target_length,
        sequence_one_hot,
        label_one_hot,
        prepend_start_token,
        configs.training_configs.training_type,
    )
    test_dataset = setupDataset(
        configs.dataset_configs,
        tokenizer,
        "test",
        configs.model_configs.hyperparameters.max_seq_len,
        target_length,
        sequence_one_hot,
        label_one_hot,
        prepend_start_token,
        configs.training_configs.training_type,
    )

    data_stats = {
        "train_num_sequences": len(train_dataset),
        "val_num_sequences": len(val_dataset),
        "test_num_sequences": len(test_dataset),
    }
    wandb.config.update(data_stats)

    trainer = Trainer(
        configs,
        train_dataset,
        val_dataset,
        test_dataset,
        outputs_dir,
        prepend_start_token,
        tokenizer,
        target_length,
        device=device,
    )

    trainer.train()

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
