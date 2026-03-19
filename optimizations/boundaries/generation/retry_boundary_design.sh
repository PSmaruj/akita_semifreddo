#!/bin/bash

#SBATCH --job-name=retry
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

python retry_boundary_design.py \
        --folds 0 1 2 3 4 5 6 7 \
        --run_name rerun_unsuccessful \
        --boundary_strength -0.5 \
        --L 125.0 \
        --retry_tsv /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/boundaries/unsuccessful_all_folds_-0.5.tsv
