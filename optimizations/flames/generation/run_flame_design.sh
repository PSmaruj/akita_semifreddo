#!/bin/bash

#SBATCH --job-name=fl_l2
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

python run_flame_design.py \
        --folds 0 1 2 3 \
        --run_name lambda/lambda_0.01 \
        --flame_strength 1.0 \
        --L 0.01 \