#!/bin/bash

#SBATCH --job-name=rerun
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
conda activate pytorch_hic

python rerun_boundary_generation_unsuccessful.py \
  --target "-0.5" \
  --model_path /scratch1/smaruj/Akita_pytorch_models/finetuned/mouse_models/Hsieh2019_mESC/models/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth \
  --pt_files_dir /scratch1/smaruj/generate_genomic_boundary \
  --boundary_mask_path /scratch1/smaruj/generate_genomic_boundary/boundary_indices.pt \
  --max_iter 2000 \
  --early_stopping_iter 2000  \
  --l 60.0 
