import json
import os
import pickle

import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

import wandb

from .configs import ModelConfigs, TestingConfigs, TrainingConfigs
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

class Sampler:
    def __init__(
        self,
        configs: TestingConfigs,
        train_configs: TrainingConfigs,
        full_filename: str,
        sequences_per_input: int,
        run_validation: bool,
        sample_dataset: SequenceDataset,
        outputs_dir: str,
        prepend_start_token: bool,
        tokenizer: PreTrainedTokenizer, 
        target_length: int,
        device: torch.device = None,
        verbose: bool = False,
    ):
        """
        A Sample class that contains necessary components for sampling from existing model

        Args:
            configs (TrainingConfigs): Config file for training
            sample_dataset (SequenceDataset): Sampling Dataset
            outputs_dir (str): Path to the output directory
            device (torch.device, optional): Device used for the runs. Defaults to None.
            verbose (bool, optional): Regulate TQDM verbosity. Defaults to False.
        """
        # General setup
        self.configs = configs
        self.train_configs = train_configs

        self.validate = run_validation

        self.full_filename = full_filename

        # Dataset setup
        # self.padding_idx = train_dataset.tokenizer.enc_dict["-"]
        # if START_TOKEN in train_dataset.tokenizer.enc_dict:
        # if prepend_start_token:
            # self.start_idx = train_dataset.tokenizer.enc_dict[START_TOKEN]
            # self.start_idx = 1
        # else:
            # self.start_idx = 0

        self.sample_dataset = sample_dataset
        self.sequences_per_input = sequences_per_input

        # Operational setup
        self.outputs_dir = outputs_dir
        self.eval_steps = self.train_configs.training_configs.eval_steps
        self.checkpoint_steps = self.train_configs.training_configs.checkpoint_steps
        self.verbose = verbose

        # Modelling setup
        self.model_type = train_configs.model_configs.model_type

        self.device = device

        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer)
        self.target_length = target_length

        self.model = self.setup_model(train_configs.model_configs, self.tokenizer)

        self.optimizer = self.setup_optimizer(
            train_configs.model_configs.hyperparameters.optimizer
        )
        self.grad_accumulation_step = (
            train_configs.model_configs.hyperparameters.grad_accumulation_step
        )

        self.model.load_state_dict(
                torch.load(configs.pretrained_model_state_dict_path, device)[
                    "model_state_dict"
                ]
            )

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
                lr=self.train_configs.model_configs.hyperparameters.learning_rate,
            )
        elif optimizer == "Adafactor":
            return torch_optimizer.Adafactor(
                list(self.model.parameters()),
                lr=self.train_configs.model_configs.hyperparameters.learning_rate,
                beta1=self.train_configs.model_configs.hyperparameters.beta1
            )
        elif optimizer == "SGD":
            return torch.optim.SGD(
                list(self.model.parameters()),
                lr=self.train_configs.model_configs.hyperparameters.learning_rate
            )
        else:
            raise NotImplementedError(f"Optimizer {optimizer} is not implemented")

    def sample(self,) -> float:
        """
        Method that represents one epoch (multiple training steps).

        Args:
            dataset (tbd): Dataset object

        Returns:
            float: the averaged loss of the epoch
        """

        # Set model mode to train
        self.model.eval()
        dataset = self.sample_dataset
        if self.configs.fixed_residue_file is not None:
            print("Loading fixed residues")
            print(self.configs.fixed_residue_file)
            with open(self.configs.fixed_residue_file, "rb") as f:
                fixed_residues = pickle.load(f)
        else:
            fixed_residues = None
        data_length = dataset.__len__()

        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.open_gap_score = self.train_configs.validation_configs.gap_insertion_penalty
        aligner.extend_gap_score = self.train_configs.validation_configs.gap_extension_penalty
        aligner.substitution_matrix = substitution_matrices.load(name=self.train_configs.validation_configs.substitution_matrix)

        identity_aligner = Align.PairwiseAligner()
        identity_aligner.mode = 'local'
        identity_aligner.open_gap_score = -1
        identity_aligner.extend_gap_score = -1

        data_loader = DataLoader(
            dataset,
            self.train_configs.model_configs.hyperparameters.batch_size,
            collate_fn=dataset.getCollator(),
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        
        if self.model_type in ["DoggyTransformer", "t5", "t5_simple"]:
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
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        total_examples = 0
        valid_length = 0
        for iteration, batch in tqdm.tqdm(
            enumerate(data_loader),
            desc=f"Sample, batch ",
            unit="",
            total=len(data_loader),
            disable=self.verbose,
        ):
            # Load data to GPU
            # batch_sequences = batch
            if self.train_configs.training_configs.training_type == "ft":
                # batch_input_sequences, batch_output_sequences = batch
                batch_input_sequences = batch["input_ids"].to(self.device)
                if self.validate:
                    batch_output_sequences = batch["output_ids"].to(self.device)
            elif self.train_configs.training_configs.training_type == "pt":
                batch_output_sequences = batch["labels"].to(self.device)
                if self.validate:
                    batch_input_sequences = batch["input_ids"].to(self.device)


            ######## Run one step of training V2
            if self.validate:
                batch_provided_output_sequences = batch_output_sequences[:,:-1]
                batch_expected_output_sequences = batch_output_sequences[:,1:]

                # print(batch_input_sequences[0])
                # print(batch_provided_output_sequences[0])
                # print(batch_expected_output_sequences[0])
                # raise Exception()
                
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

                combined_loss.backward()

                # print(f"combined_loss (early): {combined_loss.item()}")

                # # Run backprop if iteration falls on the gradient accumulation step
                # if ((iteration + 1) % self.grad_accumulation_step == 0) or (
                #     (iteration + 1) == len(data_loader)
                # ):
                #     self.optimizer.step()
                #     self.optimizer.zero_grad()

                print(f"combined_loss: {combined_loss.item()}")
                # Accumulate epoch loss
                num_examples = batch_input_sequences.size(0)
                if self.model_type in ["DoggyTransformer", "t5", "t5_simple"]:
                    total_loss["combined_loss"] += combined_loss.item() * num_examples
                    # total_loss["perplexity"] += perplexity.item() * num_examples
                total_examples += num_examples

            # If we do PT, use chain indicator as first token
            if self.train_configs.training_configs.training_type == "ft":
                y_init = batch_input_sequences[:,0].unsqueeze(
                                dim=1
                            ).to(self.device)
            # If we do PT, use tokeniser BoS as first token
            elif self.train_configs.training_configs.training_type == "pt":
                y_init = None
            # print(self.tokenizer.batch_decode([2]))
            samples, probabilities = self.model.generate_sequences(1, 
                                                    batch_input_sequences, 
                                                    y_init = y_init,
                                                    temperature=0.5, 
                                                    batch_size=len(batch_input_sequences),
                                                    sample_method = self.configs.sample_method,
                                                    topk=self.configs.top_k,
                                                    beam_width=self.configs.beam_width,
                                                    fixed_residues=fixed_residues)
            
            # print(self.tokenizer.eos_token_id)
            # print(batch_provided_output_sequences[0])
            # print(samples[0])

            # print(self.tokenizer.batch_decode(batch_provided_output_sequences)[0])
            # print(self.tokenizer.batch_decode(samples)[0])
            # raise Exception()

            samples = self.tokenizer.batch_decode(samples)
            cdrs = self.tokenizer.batch_decode(batch_input_sequences)
            if self.validate:
                targets = self.tokenizer.batch_decode(batch_expected_output_sequences)
            
            # Compute local alignment scores
            for i in range(len(batch_input_sequences)):
                if self.validate:
                    if self.train_configs.training_configs.training_type == "ft":
                        target = reconstruct_ft_sequence(targets[i], cdrs[i])
                    else:
                        target = reconstruct_pt_sequence(cdrs[i], targets[i])
                    with open(self.full_filename, "a") as output_file:
                        output_file.write("\r\ntarget: " + target + "\r\n")
                else:
                    with open(self.full_filename, "a") as output_file:
                        output_file.write("\r\ninput: " + cdrs[i] + "\r\n")

                for j in range(self.sequences_per_input):       
                    generated = samples[i*self.sequences_per_input+j]
                    if self.train_configs.training_configs.training_type == "ft":
                        sample = reconstruct_ft_sequence(generated, cdrs[i])
                    else:
                        sample = reconstruct_pt_sequence(cdrs[i], generated)

                    if self.validate:
                        if len(sample) == 0:
                            total_loss["empty_sequences"] +=1
                            continue
                        valid_length += 1
                        alignment = aligner.align(target, 
                                                    sample)
                        identity_alignment = identity_aligner.align(target, 
                                                                    sample)
                        
                        if len(alignment) == 0:
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

                        # if(self.configs.training_configs.verbose):
                        #     with open(os.path.join(self.outputs_dir, self.configs.training_configs.samples_output_file), "a") as output_file:
                        #         output_file.write(target + "\r\n")
                        #         output_file.write(targets[i] + "\r\n")
                        #         output_file.write(sample + "\r\n")
                        #         output_file.write(samples[i] + "\r\n\r\n")
                        
                        # if(self.configs.training_configs.verbose and (alignment_target - alignment_score > 150)):
                        #     with open(os.path.join(self.outputs_dir, self.configs.training_configs.verbose_output_file), "a") as output_file:
                        #         output_file.write(target + "\r\n")
                        #         output_file.write(targets[i] + "\r\n")
                        #         output_file.write(sample + "\r\n")
                        #         output_file.write(samples[i] + "\r\n\r\n")
                    with open(self.full_filename, "a") as output_file:
                        if (self.configs.verbose):
                            output_file.write(generated + "\r\n")
                        output_file.write(sample + "\r\n")

        if self.validate:
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

            print(f"Total examples: {total_examples}")
            print(f"combined_loss: {total_loss['combined_loss']}")
            total_loss["combined_loss"] = total_loss["combined_loss"] / total_examples
            total_loss["perplexity"] = total_loss["perplexity"] / total_examples

            with torch.no_grad():
                val_loss = total_loss

            # Log to WandB
            wandb_logs = {
                "epoch": 0,
            }
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

    def load_checkpoint(self, model_state_dict_path: str) -> None:
        """
        Load a training checkpoint

        Args:
            model_state_dict_path (str): Path to the pretrained model file
        """
        # print(torch.load(model_state_dict_path, self.device)["model_state_dict"])
        # raise Exception()

        self.model.load_state_dict(
            torch.load(model_state_dict_path, self.device)["model_state_dict"]
        )
        self.optimizer.load_state_dict(
            torch.load(model_state_dict_path, self.device)["optimizer_state_dict"]
        )