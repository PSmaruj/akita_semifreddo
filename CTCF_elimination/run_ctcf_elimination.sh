#!/bin/bash

#SBATCH --job-name=el_f7
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=19:00:00

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate pytorch_hic

python run_ctcf_elimination.py \
  --fold 7 \
  --target "-0.2" \
  --gamma "300.0" \
  --model_path /home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth \
  --input_tsv_dir /scratch1/smaruj/genomic_flat_regions/flat_regions_chrom_states_tsv \
  --pt_files_dir /scratch1/smaruj/generate_genomic_boundary \
  --boundary_mask_path /scratch1/smaruj/generate_genomic_boundary/boundary_indices.pt \
  --max_iter 2000 \
  --early_stopping_iter 2000  \
  