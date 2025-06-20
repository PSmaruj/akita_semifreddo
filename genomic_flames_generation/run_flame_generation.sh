#!/bin/bash

#SBATCH --job-name=flm0
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=4:00:00

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_cuda11.8

python run_flame_generation.py \
  --fold 0 \
  --target "0.5" \
  --model_path /home1/smaruj/pytorch_akita/model_0_v2_finetuned_correctly.pt \
  --input_dir /scratch1/smaruj/genomic_insertion_loci \
  --output_dir /scratch1/smaruj/generate_genomic_flame \
  --dots_mask_path /scratch1/smaruj/generate_genomic_flame/flame_indices.pt \
  --max_iter 2000 \
  --early_stopping_iter 2000  \
  --seed 5
  