#!/bin/bash

#SBATCH --job-name=val
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=10:00:00
#SBATCH --exclude=b23-18

eval "$(conda shell.bash hook)"
conda activate pytorch_hic

python run_cassette_design.py \
         --n_runs 100 \
         --lam 10.0 \
        --out_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/validation/cassette_design_multiple