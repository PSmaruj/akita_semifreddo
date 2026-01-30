#!/bin/bash

#SBATCH --job-name=g25
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=24:00:00
#SBATCH --exclude=b23-18

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_hic

python run_CTCF_restricted_suppressing.py \
  --fold 0 \
  --target "-0.5" \
  --gamma "25.0" \
  --model_path /scratch1/smaruj/Akita_pytorch_models/finetuned/mouse_models/Hsieh2019_mESC/models/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth \
  --input_tsv_dir /scratch1/smaruj/suppressing_CTCFs \
  --pt_files_dir /scratch1/smaruj/suppressing_CTCFs \
  --boundary_mask_path /scratch1/smaruj/generate_genomic_boundary/boundary_indices.pt \
  --max_iter 2000 \
  --early_stopping_iter 2000  \
  