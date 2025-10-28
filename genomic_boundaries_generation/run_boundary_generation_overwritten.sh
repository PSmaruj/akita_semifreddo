#!/bin/bash

#SBATCH --job-name=e10_f2
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=6:00:00

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

python run_boundary_generation_overwritten.py \
  --fold 2 \
  --target "-10.0" \
  --model_path /home1/smaruj/pytorch_akita/model_0_v2_finetuned_correctly.pt \
  --input_tsv_dir /scratch1/smaruj/genomic_flat_regions/flat_regions_chrom_states_tsv \
  --pt_files_dir /scratch1/smaruj/generate_genomic_boundary \
  --boundary_mask_path /scratch1/smaruj/generate_genomic_boundary/boundary_indices.pt \
  --max_iter 2000 \
  --early_stopping_iter 2000  \
  --l 60.0 
