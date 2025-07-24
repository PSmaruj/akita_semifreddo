#!/bin/bash

#SBATCH --job-name=stripepy
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
conda activate pytorch_cuda11.8

COOL_PATH="/project/fudenber_735/GEO/Hsieh2019/4DN/mESC_mm10_4DNFILZ1CPT8.mapq_30.8192.cool"
OUT_PATH="/scratch1/smaruj/stripepy_stripes/Hsieh2019.8192.hdf5"

stripepy call $COOL_PATH 8192 -o $OUT_PATH --genomic-belt 2000000 --max-width 50000 -p 4

stripepy call ./mESC_mm10_4DNFILZ1CPT8.mapq_30.8192.cool 8192 -o /scratch1/smaruj/stripepy_stripes/Hsieh2019.8192.hdf5 --genomic-belt 2000000 --max-width 50000 -p 4