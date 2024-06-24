from typing import List, Optional

from pydantic import BaseModel


class DataSplitConfigs(BaseModel):
    input_sequences_path: Optional[str]
    output_sequences_path: Optional[str]


class DatasetConfigs(BaseModel):
    train: Optional[DataSplitConfigs]
    val: Optional[DataSplitConfigs]
    test: Optional[DataSplitConfigs]
    sample: Optional[DataSplitConfigs]
    corrupted_percentage: Optional[float] = 0.15
    mean_noise_span_length: Optional[float] = 3.0
    extra_ids: Optional[int] = 100

class HyperparametersConfigs(BaseModel):
    # General hyperparameters
    max_seq_len: int
    dropout: float = 0.0
    batch_size: int = 64
    optimizer: str = "adam"
    learning_rate: float = 0.1
    grad_accumulation_step: int = 1
    ff_size: int = 2048
    # If not based on LLM
    embedding_dim: Optional[int] = 32
    hidden_dim: Optional[int] = 512
    nhead: Optional[int] = 8
    num_layers: Optional[int] = 6
    # Adafactor
    beta1: Optional[float] = None

class ModelConfigs(BaseModel):
    model_type: str
    model_state_dict_path: Optional[str]
    hyperparameters: Optional[HyperparametersConfigs]
    extra_attribute: bool = False
    tokenizer:str = "Base"
    tokenizer_path:Optional[str]

class SetupConfigs(BaseModel):
    epochs: int = 20
    eval_steps: int = 1
    checkpoint_steps: int = 1
    device: Optional[int] = 0
    random_seed: int = 1234
    outputs_dir: str = "outputs"
    training_type: str = "pt"
    wandb_project: str = "doggy_ai"
    wandb_name: str = "doggy_ai"
    verbose: bool = False
    verbose_output_file: Optional[str] = "debug_output"
    samples_output_file: Optional[str] = "samples_output"

class ValidationConfigs(BaseModel):
    substitution_matrix: str = "PAM30"
    gap_insertion_penalty: int = -11
    gap_extension_penalty: int = -1
    sample_method="topk"
    top_k: Optional[int] = 1
    beam_width: Optional[int] = 5

class TrainingConfigs(BaseModel):
    dataset_configs: DatasetConfigs
    model_configs: ModelConfigs
    training_configs: SetupConfigs
    validation_configs: ValidationConfigs

class TestingConfigs(BaseModel):
    random_seed: int = 1234
    outputs_dir: str = "test_outputs"
    device: Optional[int] = 0
    pretrained_model_state_dict_path: str
    pretrained_model_configs_path: str
    top_k: Optional[int] = 5
    beam_width: Optional[int] = 5
    sample_method="topk"
    sample_conf: DatasetConfigs
    run_validation: bool = True
    verbose: bool = False
