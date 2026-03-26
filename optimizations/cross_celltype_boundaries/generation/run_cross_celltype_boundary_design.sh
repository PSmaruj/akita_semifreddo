#!/bin/bash

#SBATCH --job-name=HF_7
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=40:00:00
#SBATCH --exclude=b23-18

eval "$(conda shell.bash hook)"
conda activate pytorch_hic

python run_cross_celltype_boundary_design.py \
        --folds 4 5 6 7 \
        --run_name HFF_strong_neg0p5_H1hESC_weak_neg0p2 \
        --strong_cell_type HFF \
        --strong_strength -0.5 \
        --weak_strength -0.2