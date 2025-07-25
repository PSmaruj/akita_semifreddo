#!/bin/bash

#SBATCH --job-name=f_l1_f2
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=7:00:00

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

python run_fountain_generation.py \
  --fold 2 \
  --target "0.5" \
  --model_path /home1/smaruj/pytorch_akita/model_0_v2_finetuned_correctly.pt \
  --input_tsv_dir /scratch1/smaruj/genomic_flat_regions/flat_regions_chrom_states_tsv \
  --pt_files_dir /scratch1/smaruj/generate_genomic_fountain \
  --fountain_mask_path /scratch1/smaruj/generate_genomic_fountain/fountain_indices.pt \
  --max_iter 2000 \
  --early_stopping_iter 2000  \
  --l 1.0
