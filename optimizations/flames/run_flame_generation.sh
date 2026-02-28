#!/bin/bash

#SBATCH --job-name=fl7_1.0
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

python run_flame_generation.py \
  --fold 7 \
  --target "1.0" \
  --model_path /scratch1/smaruj/Akita_pytorch_models/finetuned/mouse_models/Hsieh2019_mESC/models/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth \
  --input_tsv_dir /scratch1/smaruj/genomic_flat_regions/flat_regions_chrom_states_tsv \
  --pt_files_dir /scratch1/smaruj/generate_genomic_flame \
  --mask_path /scratch1/smaruj/generate_genomic_flame/flame_indices.pt \
  --max_iter 2000 \
  --early_stopping_iter 2000  \
  --l 240.0  
