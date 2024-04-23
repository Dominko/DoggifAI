import os

import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

import wandb

from .configs import ModelConfigs, TrainingConfigs
from .constants import START_TOKEN
from .dataset.dataset import SequenceDataset
from .models.DoggyTransformer import DoggyTransformer
from .models.DoggyT5 import DoggyT5
from .models.DoggyT5Simple import DoggyT5Simple
from transformers import PreTrainedTokenizer
from transformers import BatchEncoding

from .utils.common_utils import reconstruct_sequence, strip_tags

import torch_optimizer

from Bio import Align
from Bio.Align import substitution_matrices
from Bio.SeqUtils import ProtParam

import numpy as np

MODELS_MAP = {"DoggyTransformer": DoggyTransformer, "t5": DoggyT5, "t5_simple": DoggyT5Simple}

class Trainer:
    def __init__(
        self,
        configs: TrainingConfigs,
        train_dataset: SequenceDataset,
        val_dataset: SequenceDataset,
        test_dataset: SequenceDataset,
        outputs_dir: str,
        prepend_start_token: bool,
        tokenizer: PreTrainedTokenizer, 
        target_length: int,
        device: torch.device = None,
        verbose: bool = False,
    ):
        """
        A Trainer class that contains necessary components for training and operational

        Args:
            configs (TrainingConfigs): Config file for training
            train_dataset (SequenceDataset): Training Dataset
            val_dataset (SequenceDataset): Validation Dataset
            test_dataset (SequenceDataset): Testing Dataset
            outputs_dir (str): Path to the output directory
            device (torch.device, optional): Device used for the runs. Defaults to None.
            verbose (bool, optional): Regulate TQDM verbosity. Defaults to False.
        """
        # General setup
        self.configs = configs

        # Dataset setup
        # self.padding_idx = train_dataset.tokenizer.enc_dict["-"]
        # if START_TOKEN in train_dataset.tokenizer.enc_dict:
        if prepend_start_token:
            # self.start_idx = train_dataset.tokenizer.enc_dict[START_TOKEN]
            self.start_idx = 1
        else:
            self.start_idx = 0

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Operational setup
        self.outputs_dir = outputs_dir
        self.checkpoint_path = os.path.join(outputs_dir, "checkpoint")
        self.eval_steps = self.configs.training_configs.eval_steps
        self.checkpoint_steps = self.configs.training_configs.checkpoint_steps
        self.verbose = verbose

        # Modelling setup
        self.model_type = configs.model_configs.model_type

        self.device = device

        self.tokenizer = tokenizer
        self.target_length = target_length

        self.model = self.setup_model(configs.model_configs)

        self.optimizer = self.setup_optimizer(
            configs.model_configs.hyperparameters.optimizer
        )
        self.grad_accumulation_step = (
            configs.model_configs.hyperparameters.grad_accumulation_step
        )

        if configs.model_configs.model_state_dict_path:
            self.load_checkpoint(configs.model_configs.model_state_dict_path)

    def setup_model(self, model_configs: ModelConfigs) -> nn.Module:
        """
        Setup model based on the model type, number of entities and relations
        mentioned in the config file

        Args:
            model_configs (dict): configurations

        Returns:
            nn.Module: The model to be trained
        """

        print(self.tokenizer.pad_token_id)

        kwargs = {}
        if self.model_type in ["DoggyTransformer", "t5", "t5_simple"]:
            kwargs.update(
                {"start_idx": self.start_idx,
                 "padding_idx": self.tokenizer.pad_token_id,
                 "vocab_size": self.tokenizer.vocab_size,
                 "target_length": self.target_length}
            )

        if model_configs.model_type not in MODELS_MAP:
            raise NotImplementedError(
                f"Model {model_configs.model_type} not implemented"
            )

        return MODELS_MAP[model_configs.model_type](
            model_configs, self.device, **kwargs
        ).to(self.device)

    def setup_optimizer(self, optimizer: str) -> torch.optim.Optimizer:
        """
        Setup optimizer based on the optimizer name

        Args:
            optimizer (str): optimizer name

        Returns:
            torch.optim.Optimizer: Optimizer class
        """
        if optimizer == "adam":
            return torch.optim.Adam(
                list(self.model.parameters()),
                lr=self.configs.model_configs.hyperparameters.learning_rate,
            )
        elif optimizer == "Adafactor":
            return torch_optimizer.Adafactor(
                list(self.model.parameters()),
                lr=self.configs.model_configs.hyperparameters.learning_rate,
                beta1=self.configs.model_configs.hyperparameters.beta1
            )
        else:
            raise NotImplementedError(f"Optimizer {optimizer} is not implemented")

    def epoch(self, epoch, data_split) -> float:
        """
        Method that represents one epoch (multiple training steps).

        Args:
            dataset (tbd): Dataset object

        Returns:
            float: the averaged loss of the epoch
        """

        # Set model mode to train
        if data_split == "train":
            self.model.train()
            dataset = self.train_dataset
            data_length = dataset.__len__()
        elif data_split == "val":
            self.model.eval()
            dataset = self.val_dataset
            data_length = dataset.__len__()

        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.open_gap_score = self.configs.validation_configs.gap_insertion_penalty
        aligner.extend_gap_score = self.configs.validation_configs.gap_extension_penalty
        aligner.substitution_matrix = substitution_matrices.load(name=self.configs.validation_configs.substitution_matrix)

        ## Create DataLoader for sampling
        # if self.configs.training_configs.training_type == "ft":
        #     data_loader = DataLoader(
        #         dataset,
        #         self.configs.model_configs.hyperparameters.batch_size,
        #         shuffle=True,
        #     )
        # elif self.configs.training_configs.training_type == "pt":
        data_loader = DataLoader(
            dataset,
            self.configs.model_configs.hyperparameters.batch_size,
            collate_fn=dataset.getCollator(),
            shuffle=True,
        )
        
        if self.model_type in ["DoggyTransformer", "t5", "t5_simple"]:
            if data_split == "train":
                total_loss = {"combined_loss": 0, 
                            "perplexity": 0,
                            }
            elif data_split == "val":
                total_loss = {"combined_loss": 0, 
                            "perplexity": 0,
                            "avg_alignment_score": 0,
                            "avg_charge_at_pH7": 0,
                            "avg_gravy": 0,
                            "avg_instability_index": 0,
                            "avg_molecular_weight": 0,
                            "avg_charge_at_pH7_dev": 0,
                            "avg_gravy_dev": 0,
                            "avg_instability_index_dev": 0,
                            "avg_molecular_weight_dev": 0,
                            }

        total_examples = 0
        for iteration, batch in tqdm.tqdm(
            enumerate(data_loader),
            desc=f"EPOCH {epoch}, {data_split}, batch ",
            unit="",
            total=len(data_loader),
            disable=self.verbose,
        ):
            # Load data to GPU
            # batch_sequences = batch
            if self.configs.training_configs.training_type == "ft":
                # batch_input_sequences, batch_output_sequences = batch
                batch_input_sequences = batch["input_ids"].to(self.device)
                batch_output_sequences = batch["output_ids"].to(self.device)
            elif self.configs.training_configs.training_type == "pt":
                batch_output_sequences = batch["labels"].to(self.device)
                batch_input_sequences = batch["input_ids"].to(self.device)
            # batch_codon_adaptation_indices = batch_codon_adaptation_indices.to(self.device)

            # print(batch_output_sequences[0])
            # print(self.tokenizer.decode(batch_output_sequences[0]))
            # print(batch_output_sequences[1])
            # print(self.tokenizer.decode(batch_output_sequences[1]))
            # print(batch_output_sequences[2])
            # print(self.tokenizer.decode(batch_output_sequences[2]))

            # Run one step of training
            outputs = self.model.step(batch_input_sequences, batch_output_sequences)
            if self.model_type in ["DoggyTransformer", "t5", "t5_simple"]:
                combined_loss = outputs["loss"]
                perplexity = outputs["perplexity"]

            # Stop training if loss becomes inf or nan
            if torch.isinf(combined_loss):
                raise Exception("Loss is infinity. Stopping training...")
            elif torch.isnan(combined_loss):
                raise Exception("Loss is NaN. Stopping training...")

            # Average by gradient accumulation step if any
            combined_loss = combined_loss / self.grad_accumulation_step

            if data_split == "train":
                # Compute combined_loss
                combined_loss.backward()

                # Run backprop if iteration falls on the gradient accumulation step
                if ((iteration + 1) % self.grad_accumulation_step == 0) or (
                    (iteration + 1) == len(data_loader)
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if data_split == "val":
                samples = self.model.generate_sequences(len(batch_input_sequences), 
                                                        batch_input_sequences, 
                                                        temperature=1.0, 
                                                        topk=self.configs.validation_configs.top_k)
                samples = self.tokenizer.batch_decode(samples)
                targets = self.tokenizer.batch_decode(batch_output_sequences)
                cdrs = self.tokenizer.batch_decode(batch_input_sequences)
                # Compute local alignment scores
                for i in range(len(batch_input_sequences)):                    
                    if self.configs.training_configs.training_type == "ft":
                        target = reconstruct_sequence(targets[i], cdrs[i])
                        sample = reconstruct_sequence(samples[i], cdrs[i])
                    else:
                        target = strip_tags(targets[i])
                        sample = strip_tags(samples[i])

                    if len(sample) == 0:
                        continue
                    alignment = aligner.align(target, 
                                                sample)
                    if len(alignment) == 0:
                        continue
                    alignment_score = alignment[0].score
                    protein_params = ProtParam.ProteinAnalysis(sample)
                    charge_at_pH7 = protein_params.charge_at_pH(7)
                    gravy = protein_params.gravy()
                    instability_index = protein_params.instability_index()
                    molecular_weight = protein_params.molecular_weight()

                    org_protein_params = ProtParam.ProteinAnalysis(target)
                    org_charge_at_pH7 = org_protein_params.charge_at_pH(7)
                    org_gravy = org_protein_params.gravy()
                    org_instability_index = org_protein_params.instability_index()
                    org_molecular_weight = org_protein_params.molecular_weight()

                    total_loss["avg_alignment_score"] += alignment_score / data_length
                    total_loss["avg_charge_at_pH7"] += charge_at_pH7 / data_length
                    total_loss["avg_gravy"] += gravy / data_length
                    total_loss["avg_instability_index"] += instability_index / data_length
                    total_loss["avg_molecular_weight"] += molecular_weight / data_length

                    total_loss["avg_charge_at_pH7_dev"] += (charge_at_pH7 - org_charge_at_pH7) / data_length
                    total_loss["avg_gravy_dev"] += (gravy - org_gravy) / data_length
                    total_loss["avg_instability_index_dev"] += (instability_index - org_instability_index) / data_length
                    total_loss["avg_molecular_weight_dev"] += (molecular_weight - org_molecular_weight) / data_length

            # Accumulate epoch loss
            num_examples = batch_input_sequences.size(0)
            if self.model_type in ["DoggyTransformer", "t5", "t5_simple"]:
                total_loss["combined_loss"] += combined_loss.item() * num_examples
                total_loss["perplexity"] += perplexity.item() * num_examples
            total_examples += num_examples

        total_loss["combined_loss"] = total_loss["combined_loss"] / total_examples
        total_loss["perplexity"] = total_loss["perplexity"] / total_examples

        return total_loss
        # return {
        #     metrics_name: metrics_value / total_examples
        #     for metrics_name, metrics_value in total_loss.items()
        # }

    def train(self):
        for epoch in range(1, 1 + self.configs.training_configs.epochs):
            # Run one epoch of training
            train_loss = self.epoch(epoch, "train")

            # Run evaluation if it is the eval step
            if epoch % self.eval_steps == 0:
                # Run one epoch of validation without any gradients computation
                with torch.no_grad():
                    val_loss = self.epoch(epoch, "val")

                # Log to WandB
                wandb_logs = {
                    "epoch": epoch,
                }
                wandb_logs.update(
                    {
                        f"train_{metric_name}": value
                        for metric_name, value in train_loss.items()
                    }
                )
                wandb_logs.update(
                    {
                        f"val_{metric_name}": value
                        for metric_name, value in val_loss.items()
                    }
                )
                wandb.log(wandb_logs)

                # Print the log for verbosity
                for metric_name, value in wandb_logs.items():
                    print(f"- {metric_name}: {value:.4f}")

            # Save checkpoint if it is the checkpoint step
            if epoch % self.checkpoint_steps == 0:
                self.save_checkpoint(epoch, wandb_logs)

    def save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """
        Save checkpoints of all training components

        Args:
            epoch (int): Current epoch
            metrics (dict): Current metrics achieved by the model
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        checkpoint.update(metrics)
        torch.save(checkpoint, f"{self.checkpoint_path}_{epoch}.pt")

    def load_checkpoint(self, model_state_dict_path: str) -> None:
        """
        Load a training checkpoint

        Args:
            model_state_dict_path (str): Path to the pretrained model file
        """
        self.model.load_state_dict(
            torch.load(model_state_dict_path, self.device)["model_state_dict"]
        )
        self.optimizer.load_state_dict(
            torch.load(model_state_dict_path, self.device)["optimizer_state_dict"]
        )
