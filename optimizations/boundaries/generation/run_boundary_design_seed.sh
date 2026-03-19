#!/bin/bash

#SBATCH --job-name=l_s9
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

python run_boundary_design_seed.py \
        --folds 0 1 2 3 \
        --seeds 9 \
        --run_name indep_runs_lambda_0.01/seed9 \
        --boundary_strength -0.5 \
        --L 0.01 \
