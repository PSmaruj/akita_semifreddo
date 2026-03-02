#!/bin/bash

#SBATCH --job-name=ctcf_jaccard
#SBATCH --account=fudenber_735
#SBATCH --partition=qcb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000MB
#SBATCH --time=1:00:00

eval "$(conda shell.bash hook)"
conda activate pytorch_hic

python calculate_ctcf_jaccard.py \
  --fold          0 \
  --input_dir     /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/genome_rewiring \
  --results_dir   /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/genome_rewiring/results_fold0 \
  --output        /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/genome_rewiring/ctcf_jaccard_results_fold0.tsv