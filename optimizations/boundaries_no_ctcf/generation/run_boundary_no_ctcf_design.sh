#!/bin/bash

#SBATCH --job-name=g1.8e5
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=24:00:00
#SBATCH --exclude=b23-18

eval "$(conda shell.bash hook)"
conda activate pytorch_hic

python run_boundary_no_ctcf_design.py \
        --folds 0 1 2 3 4 5 6 7 \
        --run_name gamma/gamma_1.8e5 \
        --boundary_strength -0.2 \
        --L 0.01 \
        --gamma 1.8e5
