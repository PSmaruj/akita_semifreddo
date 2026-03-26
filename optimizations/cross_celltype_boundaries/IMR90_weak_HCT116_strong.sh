#!/bin/bash

#SBATCH --job-name=IMw_HCs
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=3-00:00:00
#SBATCH --exclude=b23-18       # ← Avoid broken GPU node

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_hic

python IMR90_weak_HCT116_strong.py \
  --fold 0 \
  --target "0.5" \
  --input_tsv_dir /scratch1/smaruj/generate_cell_type_specific_features \
  --pt_files_dir /scratch1/smaruj/generate_cell_type_specific_features \
  --boundary_mask_path /scratch1/smaruj/generate_cell_type_specific_features/boundary_indices.pt \
  --max_iter 2000 \
  --early_stopping_iter 2000  \
  --l 10.0