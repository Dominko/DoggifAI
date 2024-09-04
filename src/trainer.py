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

from .utils.common_utils import reconstruct_pt_sequence, reconstruct_ft_sequence

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
        # if prepend_start_token:
        #     # self.start_idx = train_dataset.tokenizer.enc_dict[START_TOKEN]
        #     self.start_idx = 1
        # else:
        #     self.start_idx = 0

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
        self.vocab_size = len(self.tokenizer)
        self.target_length = target_length

        self.model = self.setup_model(configs.model_configs, self.tokenizer)

        self.optimizer = self.setup_optimizer(
            configs.model_configs.hyperparameters.optimizer
        )
        self.grad_accumulation_step = (
            configs.model_configs.hyperparameters.grad_accumulation_step
        )

        self.start_epoch = 1

        if configs.model_configs.model_state_dict_path:
            self.load_checkpoint(configs.model_configs.model_state_dict_path)
            self.start_epoch = configs.model_configs.start_epoch
            

    def setup_model(self, model_configs: ModelConfigs, tokenizer) -> nn.Module:
        """
        Setup model based on the model type, number of entities and relations
        mentioned in the config file

        Args:
            model_configs (dict): configurations

        Returns:
            nn.Module: The model to be trained
        """

        kwargs = {}
        if self.model_type in ["DoggyTransformer", "t5", "t5_simple"]:
            kwargs.update(
                {"bos_idx": self.tokenizer.bos_token_id,
                 "eos_idx": self.tokenizer.eos_token_id,
                 "padding_idx": self.tokenizer.pad_token_id,
                 "vocab_size": self.vocab_size,
                 "target_length": self.target_length}
            )

        if model_configs.model_type not in MODELS_MAP:
            raise NotImplementedError(
                f"Model {model_configs.model_type} not implemented"
            )

        return MODELS_MAP[model_configs.model_type](
            model_configs, tokenizer, self.device, **kwargs
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
        elif optimizer == "SGD":
            return torch.optim.SGD(
                list(self.model.parameters()),
                lr=self.configs.model_configs.hyperparameters.learning_rate
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

        identity_aligner = Align.PairwiseAligner()
        identity_aligner.mode = 'local'
        identity_aligner.open_gap_score = -1
        identity_aligner.extend_gap_score = -1

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
            num_workers=8,
            pin_memory=True
        )
        
        if self.model_type in ["DoggyTransformer", "t5", "t5_simple"]:
            if data_split == "train":
                total_loss = {"combined_loss": 0, 
                            "perplexity": 0,
                            "avg_alignment_score": 0,
                            "avg_alignment_loss": 0,
                            "avg_alignment_identity_mismatch": 0,
                            "avg_alignment_log_probability":0,
                            "avg_charge_at_pH7": 0,
                            "avg_gravy": 0,
                            "avg_instability_index": 0,
                            "avg_molecular_weight": 0,
                            "avg_charge_at_pH7_dev": 0,
                            "avg_gravy_dev": 0,
                            "avg_instability_index_dev": 0,
                            "avg_molecular_weight_dev": 0,
                            "empty_sequences": 0,
                            "not_aligned": 0
                            }
            elif data_split == "val":
                total_loss = {"combined_loss": 0, 
                            "perplexity": 0,
                            "avg_alignment_score": 0,
                            "avg_alignment_loss": 0,
                            "avg_alignment_identity_mismatch": 0,
                            "avg_alignment_log_probability":0,
                            "avg_charge_at_pH7": 0,
                            "avg_gravy": 0,
                            "avg_instability_index": 0,
                            "avg_molecular_weight": 0,
                            "avg_charge_at_pH7_dev": 0,
                            "avg_gravy_dev": 0,
                            "avg_instability_index_dev": 0,
                            "avg_molecular_weight_dev": 0,
                            "empty_sequences": 0,
                            "not_aligned": 0
                            }

        total_examples = 0
        valid_length = 0
        # TEST
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        if(self.configs.training_configs.verbose):
            with open(os.path.join(self.outputs_dir, self.configs.training_configs.samples_output_file), "a") as output_file:
                output_file.write(f"Epoch {epoch}, {data_split} \r\n")
            with open(os.path.join(self.outputs_dir, self.configs.training_configs.verbose_output_file), "a") as output_file:
                output_file.write(f"Epoch {epoch}, {data_split} \r\n")

        for iteration, batch in tqdm.tqdm(
            enumerate(data_loader),
            desc=f"EPOCH {epoch}, {data_split}, batch ",
            unit="",
            total=len(data_loader),
            disable=self.verbose,
        ):
            if data_split == "train":
                self.model.train()
                self.optimizer.zero_grad()
            else:
                self.model.eval()
            # Load data to GPU
            # batch_sequences = batch
            if self.configs.training_configs.training_type == "ft":
                # batch_input_sequences, batch_output_sequences = batch
                batch_input_sequences = batch["input_ids"].to(self.device)
                batch_output_sequences = batch["output_ids"].to(self.device)
            elif self.configs.training_configs.training_type == "pt":
                batch_output_sequences = batch["labels"].to(self.device)
                batch_input_sequences = batch["input_ids"].to(self.device)

            batch_provided_output_sequences = batch_output_sequences[:,:-1]
            batch_expected_output_sequences = batch_output_sequences[:,1:]

            outputs = self.model(batch_input_sequences, batch_provided_output_sequences)
            outputs = outputs.permute(0, 2, 1)
            
            combined_loss = loss_fn(outputs, batch_expected_output_sequences)

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

            # Accumulate epoch loss
            num_examples = batch_input_sequences.size(0)
            if self.model_type in ["DoggyTransformer", "t5", "t5_simple"]:
                total_loss["combined_loss"] += combined_loss.item() * num_examples
                # total_loss["perplexity"] += perplexity.item() * num_examples
            total_examples += num_examples
            
            # if (data_split == "val" or data_split == "train") and iteration < 100 and epoch >= 500:
            
            if (data_split == "val" or data_split == "train") and iteration < 100:
                with torch.no_grad():
                    self.model.eval()
                    # If we do PT, use chain indicator as first token
                    if self.configs.training_configs.training_type == "ft":
                        y_init = batch_input_sequences[:,0].unsqueeze(
                                        dim=1
                                    ).to(self.device)
                    # If we do PT, use tokeniser BoS as first token
                    elif self.configs.training_configs.training_type == "pt":
                        y_init = None
                    samples, probabilities = self.model.generate_sequences(1, 
                                                            batch_input_sequences, 
                                                            y_init = y_init,
                                                            temperature=0.5, 
                                                            batch_size=len(batch_input_sequences),
                                                            sample_method = self.configs.validation_configs.sample_method,
                                                            topk=self.configs.validation_configs.top_k,
                                                            beam_width=self.configs.validation_configs.beam_width)
                    # print(samples)
                    if data_split == "train":
                        self.model.train()
                    else:
                        self.model.eval() 

                    samples = self.tokenizer.batch_decode(samples)
                    targets = self.tokenizer.batch_decode(batch_expected_output_sequences)
                    cdrs = self.tokenizer.batch_decode(batch_input_sequences)

                    # Compute local alignment scores
                    for i in range(len(batch_input_sequences)):      
                        if self.configs.training_configs.training_type == "ft":
                            target = reconstruct_ft_sequence(targets[i], cdrs[i])
                            sample = reconstruct_ft_sequence(samples[i], cdrs[i])
                        else:
                            target = reconstruct_pt_sequence(cdrs[i], targets[i])
                            sample = reconstruct_pt_sequence(cdrs[i], samples[i])

                        # if i == 1 or i == 2:
                        # print(target)
                        # print(sample)

                        if len(sample) == 0:
                            total_loss["empty_sequences"] +=1
                            continue
                        valid_length += 1
                        alignment = aligner.align(target, 
                                                    sample)
                        identity_alignment = identity_aligner.align(target, sample)
                        if len(alignment) == 0 or len(identity_alignment) == 0:
                            alignment_score = 0
                            identity_alignment_score = 0
                            total_loss["not_aligned"] +=1
                        else:
                            alignment_score = alignment[0].score
                            identity_alignment_score = identity_alignment[0].score

                        alignment_target = aligner.align(target, 
                                                    target)
                        identity_alignment_target = identity_aligner.align(target, 
                                                                    target)
                        
                        alignment_target = alignment_target[0].score
                        identity_alignment_target = identity_alignment_target[0].score
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

                        total_loss["avg_alignment_score"] += alignment_score
                        total_loss["avg_alignment_loss"] += (alignment_target - alignment_score)
                        total_loss["avg_alignment_identity_mismatch"] += (identity_alignment_target - identity_alignment_score)
                        total_loss["avg_alignment_log_probability"] += probabilities.sum()
                        total_loss["avg_charge_at_pH7"] += charge_at_pH7
                        total_loss["avg_gravy"] += gravy
                        total_loss["avg_instability_index"] += instability_index
                        total_loss["avg_molecular_weight"] += molecular_weight

                        total_loss["avg_charge_at_pH7_dev"] += (charge_at_pH7 - org_charge_at_pH7)
                        total_loss["avg_gravy_dev"] += (gravy - org_gravy)
                        total_loss["avg_instability_index_dev"] += (instability_index - org_instability_index)
                        total_loss["avg_molecular_weight_dev"] += (molecular_weight - org_molecular_weight)

                        if(self.configs.training_configs.verbose):
                            with open(os.path.join(self.outputs_dir, self.configs.training_configs.samples_output_file), "a") as output_file:
                                output_file.write(target + "\r\n")
                                output_file.write(targets[i] + "\r\n")
                                output_file.write(sample + "\r\n")
                                output_file.write(samples[i] + "\r\n\r\n")
                        
                        if(self.configs.training_configs.verbose and (alignment_target - alignment_score > 150)):
                            with open(os.path.join(self.outputs_dir, self.configs.training_configs.verbose_output_file), "a") as output_file:
                                output_file.write(target + "\r\n")
                                output_file.write(targets[i] + "\r\n")
                                output_file.write(sample + "\r\n")
                                output_file.write(samples[i] + "\r\n\r\n")

        if (valid_length != 0):
            total_loss["avg_alignment_score"] /= valid_length
            total_loss["avg_alignment_loss"] /= valid_length
            total_loss["avg_alignment_identity_mismatch"] /= valid_length
            total_loss["avg_alignment_log_probability"] /= valid_length
            total_loss["avg_charge_at_pH7"] /= valid_length
            total_loss["avg_gravy"] /= valid_length
            total_loss["avg_instability_index"] /= valid_length
            total_loss["avg_molecular_weight"] /= valid_length

            total_loss["avg_charge_at_pH7_dev"] /= valid_length
            total_loss["avg_gravy_dev"] /= valid_length
            total_loss["avg_instability_index_dev"] /= valid_length
            total_loss["avg_molecular_weight_dev"] /= valid_length

        print(total_loss["combined_loss"])
        print(total_examples)
        total_loss["combined_loss"] = total_loss["combined_loss"] / total_examples
        total_loss["perplexity"] = total_loss["perplexity"] / total_examples
        # if data_split == "val":
        #     data_loader = DataLoader(
        #         dataset,
        #         len(dataset),
        #         collate_fn=dataset.getCollator(),
        #         shuffle=True,
        #     )
        #     iter, full_batch = next(enumerate(data_loader))

        #     if self.configs.training_configs.training_type == "ft":
        #         # batch_input_sequences, batch_output_sequences = batch
        #         batch_input_sequences = full_batch["input_ids"].to(self.device)
        #         batch_output_sequences = full_batch["output_ids"].to(self.device)
        #     elif self.configs.training_configs.training_type == "pt":
        #         batch_output_sequences = full_batch["labels"].to(self.device)
        #         batch_input_sequences = full_batch["input_ids"].to(self.device)


            

        return total_loss
        # return {
        #     metrics_name: metrics_value / total_examples
        #     for metrics_name, metrics_value in total_loss.items()
        # }

    def train(self):
        for epoch in range(self.start_epoch, 1 + self.configs.training_configs.epochs):
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
