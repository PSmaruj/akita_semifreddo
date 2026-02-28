#!/bin/bash

#SBATCH --job-name=hmr_search
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=9:00:00

cd /home1/smaruj/ledidi_akita/homer_ctcf_elimination_OG

findMotifs.pl ctcf_elim.target.sequences.fasta fasta ./ -fastaBg /scratch1/smaruj/CTCF_elimination/gamma_300.0/OG_combined_sequences.fasta
