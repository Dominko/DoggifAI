#!/bin/bash
#SBATCH --qos epsrc
#SBATCH --time 1:59:59
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-gpu 4
#SBATCH --gpus-per-task 1

echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

module purge
module load baskerville

# Set up AlphaFold
module load AlphaFold
export ALPHAFOLD_DATA_DIR=${BASK_APPS_DATA}/AlphaFold/20220825

echo "Running analysis"
echo "Fasta Paths file: $1"
echo "Paths to FASTA files, each containing a prediction target 
        that will be folded one after another. If a FASTA file 
        contains multiple sequences, then it will be folded as a
        multimer. Paths should be separated by commas."

echo "Output directory: $2"

alphafold --fasta_paths $1 --output_dir $2 --max_template_date=2022-01-01