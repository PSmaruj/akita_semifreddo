#!/bin/bash

#SBATCH --job-name=fr_bg
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

python run_background_to_genomic.py \
  --fold 0 \
  --background_fasta /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/background_generation/background_sequences_scd30_totvar1300.fasta \
  --model_path /home1/smaruj/akita_pytorch/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth \
  --input_dir  /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/genome_rewiring \
  --output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/genome_rewiring/FROM_BG \
  --max_iter            2000 \
  --early_stopping_iter 2000 \
  --l 0.05