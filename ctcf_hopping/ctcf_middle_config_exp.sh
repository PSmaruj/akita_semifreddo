#!/bin/bash

#SBATCH --job-name=left
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=72:00:00
#SBATCH --exclude=b23-18

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_hic

python ctcf_middle_config_exp.py --outer_orient "<<" 
