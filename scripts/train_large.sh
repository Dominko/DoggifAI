#!/bin/bash
#SBATCH --qos epsrc
#SBATCH --time 9-23:59:59
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-gpu 4
#SBATCH --mem-per-gpu 32G
#SBATCH --gpus-per-task 4

echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

module purge
module load baskerville
module load Miniforge3/24.1.2-0
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# echo "Setting up bash enviroment"
# source ~/.bashrc
#set -e
#SCRATCH_DISK=/disk/scratch
#SCRATCH_HOME=${SCRATCH_DISK}/${USER}
#mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
# Define the path to your existing Conda environment (modify as appropriate)
CONDA_ENV_NAME=spike_rna
# CONDA_ENV_PATH="/bask/projects/j/jlxi8926-auto-sum/dgrabarczyk/envs/${CONDA_ENV_NAME}"
echo "Activating conda environment: ${CONDA_ENV_NAME}"
mamba activate ${CONDA_ENV_NAME}

echo "Setting up Wandb API key"
key=`cat scripts/wandb_key`
export WANDB_API_KEY=$key

echo "setting huggingface cache dir"
export HF_DATASETS_CACHE="/bask/projects/j/jlxi8926-auto-sum/dgrabarczyk/.cache"

echo "Running experiment"
echo "Config: $1"
# limit of 12 GB GPU is hidden 256 and batch size 256
python scripts/train.py \
--config_filepath=$1 \
--log_to_wandb

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
