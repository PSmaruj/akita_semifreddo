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

eval "$(conda shell.bash hook)"
conda activate pytorch_hic

python run_genome_rewiring.py \
  --fold 0 \
  --model_path /home1/smaruj/akita_pytorch/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth \
  --input_dir  /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/genome_rewiring \
  --output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/genome_rewiring \
  --max_iter            2000 \
  --early_stopping_iter 2000 \
  --l 0.05