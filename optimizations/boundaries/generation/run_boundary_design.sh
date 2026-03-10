#!/bin/bash

#SBATCH --job-name=f3_0.1
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=12:00:00
#SBATCH --exclude=b23-18

eval "$(conda shell.bash hook)"
conda activate pytorch_hic

python run_boundary_design.py \
        --fold 3 \
        --run_name lambda/lambda_0.1 \
        --boundary_strength -0.5 \
        --L 0.1 \