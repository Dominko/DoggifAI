from typing import List, Optional

from pydantic import BaseModel


class DataSplitConfigs(BaseModel):
    input_sequences_path: Optional[str]
    output_sequences_path: str


class DatasetConfigs(BaseModel):
    train: DataSplitConfigs
    val: DataSplitConfigs
    test: DataSplitConfigs
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

class TrainingConfigs(BaseModel):
    dataset_configs: DatasetConfigs
    model_configs: ModelConfigs
    training_configs: SetupConfigs


class TestingConfigs(BaseModel):
    random_seed: int = 1234
    outputs_dir: str = "test_outputs"
    device: Optional[int] = 0
    pretrained_model_state_dict_path: str
    pretrained_model_configs_path: str
    topk=5
    input_path: str
