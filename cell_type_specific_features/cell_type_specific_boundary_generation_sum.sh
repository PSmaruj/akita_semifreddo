#!/bin/bash

#SBATCH --job-name=ct_b_sum
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=12:00:00

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

python cell_type_specific_boundary_generation_sum.py \
  --fold 0 \
  --target "0.5" \
  --input_tsv_dir /scratch1/smaruj/genomic_flat_regions/flat_regions_chrom_states_tsv \
  --pt_files_dir /scratch1/smaruj/generate_cell_type_specific_features \
  --boundary_mask_path /scratch1/smaruj/generate_cell_type_specific_features/boundary_indices.pt \
  --max_iter 2000 \
  --early_stopping_iter 2000  \
  --l 10.0
