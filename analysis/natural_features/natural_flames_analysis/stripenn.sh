#!/bin/bash

#SBATCH --job-name=nat_flames
#SBATCH --account=fudenber_735
#SBATCH --partition=qcbgpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --gpus-per-node=1
#SBATCH --mem=450000MB
#SBATCH --time=10:00:00

# Conda env activation
eval "$(conda shell.bash hook)"
conda activate stripenn_env

stripenn compute --cool /project/fudenber_735/GEO/Hsieh2019/4DN/mESC_mm10_4DNFILZ1CPT8.mapq_30.8192.cool --out /scratch1/smaruj/stripenn_out -n 50
