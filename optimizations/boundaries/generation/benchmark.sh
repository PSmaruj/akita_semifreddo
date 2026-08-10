#!/bin/bash

#SBATCH --job-name=single
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=20:00:00
#SBATCH --exclude=b23-18

eval "$(conda shell.bash hook)"
conda activate pytorch_hic

python benchmark.py \
        --folds 0 1 2 \
        --run_name benchmark \
        --boundary_strength -1.0 \
        --L 125.0 \
