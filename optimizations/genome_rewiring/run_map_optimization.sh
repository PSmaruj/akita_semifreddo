#!/bin/bash

#SBATCH --job-name=maps
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=72:00:00
#SBATCH --exclude=b23-18

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_hic

python run_map_optimization.py \
  --fold 0 \
  --model_path /scratch1/smaruj/Akita_pytorch_models/finetuned/mouse_models/Hsieh2019_mESC/models/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth \
  --input_dir /scratch1/smaruj/genomic_map_transformation \
  --output_dir /scratch1/smaruj/genomic_map_transformation \
  --max_iter 2000 \
  --early_stopping_iter 2000