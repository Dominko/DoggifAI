# DoggyAI
This repository contains the code related to the paper [DoggifAI: a transformer based approach for antibody caninisation](http://doi.org/10.1101/2025.05.28.656573)

DoggifAI is a canine antibody (Ab) framework region (FR) generation model, conditioned on the complimentarity determining regions (CDRs). The model follows a standard T5 transformer architecture and can be trained with or without semi-supervised pretraining.

## Getting started
Below is a short guide to get started with the code base

### Environment set up
The enviroment should be set up using conda form the environment.yml file.
This can be done using the command
```
conda env create -f environment.yml
conda activate doggifai
```

### Training and logging
Training is started by running the train.py script which takes a config path argument and a flag to log the results to wandb.

The code can be run on slurm based computing clusters by modifying the example train.sh script for the cluster you are going to use.

The code is built for logging using [Weights and Biases](https://wandb.ai/).
In case code is run locally the user will be prompted to log in on the first use of the code.
For use on distributed compute clusters, we recommend exporting the wandb API key as shown in the train.sh example script.

If logging is not used, the outputs will only be shown in the console.

### Inference
Training is started by running the sample.py script which takes a config path argument and a flag for how many sequences to generate per input.

The code can be run on slurm based computing clusters by modifying the example sample.sh script for the cluster you are going to use.

### Example configs
Example configs are provided in the configs folder. These are split between training scripts and test scripts (inference). They provide examples for pretaining and finetuning set ups as well as cases like resuming runs from checkpoints.

### Scripts
The respository also contains most of the scripts used to generate the figures included in the paper as jupyter notebooks. These are not crucial for the working of the model, but can be used to compare outputs to those in our publication.


## Data availability
The canine dataset used to train the model, including the files for light kappa, light lambda, and heavy immunoglobulin chains as well as the trained Large OAS model, is available [HERE](https://doi.org/10.5281/zenodo.15125375). 

The OAS dataset can be provided upon request.