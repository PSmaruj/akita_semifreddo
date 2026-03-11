#!/bin/bash

#SBATCH --job-name=f0_e-3
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

# python run_boundary_design.py \
#         --fold 3 \
#         --run_name lambda/lambda_100.0 \
#         --boundary_strength -0.5 \
#         --L 100.0 \


python run_boundary_design.py \
        --fold 0 \
        --run_name epsilon/epsilon_1e-3 \
        --boundary_strength -0.5 \
        --eps 1e-3 \


# python run_boundary_design.py \
#         --fold 0 \
#         --run_name tau/tau_10.0 \
#         --boundary_strength -0.5 \
#         --tau 10.0 \