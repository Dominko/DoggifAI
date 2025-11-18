#!/bin/bash
#SBATCH --qos epsrc
#SBATCH --time 1:59:59
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-gpu 4
#SBATCH --gpus-per-task 4

echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# Needed to get conda to tun
module purge
module load baskerville
module load Miniforge3/24.1.2-0
eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"

# Activate your conda environment
CONDA_ENV_NAME=doggifai
# CONDA_ENV_PATH="/bask/projects/j/jlxi8926-auto-sum/dgrabarczyk/envs/${CONDA_ENV_NAME}"
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

echo "Setting up Wandb API key"
key=`cat scripts/wandb_key`
export WANDB_API_KEY=$key

echo "setting huggingface cache dir"
export HF_DATASETS_CACHE="/bask/projects/j/jlxi8926-auto-sum/dgrabarczyk/.cache"

echo "Running experiment"
echo "Config: $1"

echo "Running experiment"
python scripts/sample.py \
--config_filepath $1 \
--sequences_per_input 1

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"

