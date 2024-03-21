#!/bin/bash
# # SBATCH -o /home/%u/slogs/sl_%A.out
# # SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=ampere
#SBATCH --account=BMAI-CDT-SL2-GPU
#SBATCH -t 1-00:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-gpu=4

echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

echo "Setting up bash enviroment"
source ~/.bashrc
#set -e
#SCRATCH_DISK=/disk/scratch
#SCRATCH_HOME=${SCRATCH_DISK}/${USER}
#mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
CONDA_ENV_NAME=rnaformer
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

echo "Running experiment"
# limit of 12 GB GPU is hidden 256 and batch size 256
python scripts/sample.py \
--config_filepath $1 \
--num_sequences=33000

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"

