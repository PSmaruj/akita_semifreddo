#!/bin/bash

#SBATCH --job-name=ctcfs
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=15:00:00

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_hic

python compare_CTCFs_fold.py 
