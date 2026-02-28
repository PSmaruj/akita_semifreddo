#!/bin/bash

#SBATCH --job-name=hmr_bkgrd
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=9:00:00

cd /home1/smaruj/ledidi_akita/homer_ctcf_suppressing

homer2 background -i /scratch1/smaruj/suppressing_CTCFs/results/combined_sequences.fasta -g /project/fudenber_735/genomes/mm10/mm10.fa -o ctcf_supp
