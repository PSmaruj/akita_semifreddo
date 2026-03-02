#!/usr/bin/env python3
"""
Semifreddo vs Full Akita Model Comparison Script
Compares prediction time, memory usage, and accuracy between full Akita model
and Semifreddo approximation on genomic regions with inserted CTCF sequences.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import time
import psutil
import sys
import os
from pyfaidx import Fasta
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Semifreddo analysis')
parser.add_argument('--fold', type=int, default=0, help='Fold number (default: 0)')
parser.add_argument('--output_dir', type=str, default='./', help='Output directory for results')
args = parser.parse_args()

FOLD = args.fold
OUTPUT_DIR = args.output_dir

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Starting analysis for fold {FOLD}")
print(f"Output directory: {OUTPUT_DIR}")

# Add custom paths to system path
sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from akita_model.model import SeqNN

sys.path.append(os.path.abspath("/home1/smaruj/ledidi/"))
from semifreddo_full_v2_model_mod import Semifreddo

# Load data
print("Loading flat regions data...")
flat_df = pd.read_csv(
    f"/scratch1/smaruj/genomic_flat_regions/flat_regions_chrom_states_tsv/fold{FOLD}_selected_genomic_windows_centered_chrom_states.tsv",
    sep="\t"
)
print(f"Loaded {len(flat_df)} flat regions")

# Load genome
print("Loading genome...")
genome = Fasta("/project2/fudenber_735/genomes/mm10/mm10.fa")

def get_ctcf_forward_seq(chrom, start, end, strand):
    """Extract CTCF sequence and apply reverse complement if on minus strand"""
    seq = genome[chrom][start:end].seq
    if strand == "-":
        complement = str.maketrans("ACGTacgt", "TGCAtgca")
        seq = seq[::-1].translate(complement)
    return seq.upper()

# 100 strongest CTCFs
ctcf_df = pd.read_csv("/scratch1/smaruj/full_akita_vs_semifreddo/top100_ctcfs.csv")

print("Processing CTCF sequences...")
ctcf_df["ctcf_seq"] = ctcf_df.apply(
    lambda row: get_ctcf_forward_seq(row["chrom"], row["start"], row["end"], row["strand"]),
    axis=1
)

# Create Cartesian product of flat regions and CTCFs
print("Creating merged dataset...")
flat_df["key"] = 1
ctcf_df["key"] = 1
merged_df = pd.merge(flat_df, ctcf_df, on="key", suffixes=("_flat", "_ctcf")).drop(columns="key")

merged_df = merged_df.rename(columns={
    "chrom_flat": "chrom",
    "centered_start": "centered_start",
    "centered_end": "centered_end",
    "chrom_ctcf": "ctcf_chrom",
    "start": "ctcf_start",
    "end": "ctcf_end",
    "strand": "ctcf_strand",
    "ctcf_seq": "ctcf_seq"
})
print(f"Created {len(merged_df)} combinations")

# Setup device and load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading Akita model...")
model = SeqNN()
model.load_state_dict(
    torch.load(
        "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth",
        map_location=device
    )
)
model.eval()
print("Model loaded successfully")

def ohe(seq):
    """One-hot encode a DNA sequence"""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    X = torch.zeros(4, len(seq), dtype=torch.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            X[mapping[base], i] = 1.0
    return X

# Parameters
bin_size = 2048
n_bins = 5

# Storage for metrics
pred_times_seqnn = []
pred_mem_seqnn = []
pred_times_sf = []
pred_mem_sf = []
pearsonRs = []

print("\n" + "="*80)
print("RUNNING PREDICTIONS")
print("="*80)

# Main prediction loop
total_rows = len(merged_df)
for idx, row in merged_df.iterrows():
    if idx % 10 == 0:
        print(f"Processing {idx+1}/{total_rows}...")
    
    # 1. Load original sequence X (one-hot)
    X = torch.load(
        f"/scratch1/smaruj/generate_genomic_boundary/ohe_X/{row['fold']}/{row['chrom']}_{row['centered_start']}_{row['centered_end']}_X.pt",
        weights_only=True
    ).squeeze(0)
    
    # 2. One-hot encode CTCF
    ctcf_seq = row["ctcf_seq"]
    ctcf_ohe = ohe(ctcf_seq)
    
    # 3. Insert CTCF in the middle of X
    seq_len = X.shape[1]
    mid = seq_len // 2
    ctcf_len = ctcf_ohe.shape[1]
    start_idx = mid - ctcf_len // 2
    end_idx = start_idx + ctcf_len
    X[:, start_idx:end_idx] = ctcf_ohe
    
    # 4. Full model prediction
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    with torch.no_grad():
        pred_seqnn = model(X.unsqueeze(0).to(device))
    t1 = time.time()
    
    pred_times_seqnn.append(t1 - t0)
    pred_mem_seqnn.append(torch.cuda.max_memory_allocated(device) / 1e6)
    
    # 5. Prepare padded slice for Semifreddo (central 5 bins)
    central_start = mid - bin_size // 2
    central_end = central_start + bin_size
    slice_start = max(0, central_start - 2 * bin_size)
    slice_end = min(seq_len, central_end + 2 * bin_size)
    slice_0_padded_seq = X[:, slice_start:slice_end].clone()
    
    # Edited indices relative to padded slice
    edited_indices_slice_0 = [256]
    slice_0_padded_seq = slice_0_padded_seq.unsqueeze(0)
    
    # 6. Run Semifreddo
    precom_tensor = torch.load(
        f"/scratch1/smaruj/generate_genomic_boundary/tower_outputs/{row['fold']}/{row['chrom']}_{row['centered_start']}_{row['centered_end']}_tower_out.pt",
        weights_only=True
    ).to(device)
    
    semifreddo_instance = Semifreddo(
        model=model,
        slice_0_padded_seq=slice_0_padded_seq,
        edited_indices_slice_0=edited_indices_slice_0,
        precomputed_full_output=precom_tensor,
        cropping_applied=64,
        batch_size=1
    )
    
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    with torch.no_grad():
        pred_sf = semifreddo_instance.forward()
    t1 = time.time()
    
    pred_times_sf.append(t1 - t0)
    pred_mem_sf.append(torch.cuda.max_memory_allocated(device) / 1e6)
    
    # 7. Compute PearsonR between full and semifreddo predictions
    r, _ = pearsonr(pred_seqnn.flatten().cpu().numpy(), pred_sf.flatten().cpu().numpy())
    pearsonRs.append(r)

# Add metrics to dataframe
merged_df["pred_time_seqnn"] = pred_times_seqnn
merged_df["pred_mem_seqnn_MB"] = pred_mem_seqnn
merged_df["pred_time_sf"] = pred_times_sf
merged_df["pred_mem_sf_MB"] = pred_mem_sf
merged_df["pearsonR"] = pearsonRs

# Calculate and save average metrics
avg_metrics = merged_df[["pred_time_seqnn", "pred_mem_seqnn_MB",
                         "pred_time_sf", "pred_mem_sf_MB", "pearsonR"]].mean()
print("\n" + "="*80)
print("AVERAGE METRICS")
print("="*80)
print(avg_metrics)

# Save results
results_file = os.path.join(OUTPUT_DIR, f"fold{FOLD}_results.tsv")
merged_df.to_csv(results_file, sep='\t', index=False)
print(f"\nResults saved to: {results_file}")

# Save summary metrics
summary_file = os.path.join(OUTPUT_DIR, f"fold{FOLD}_summary.txt")
with open(summary_file, 'w') as f:
    f.write("AVERAGE METRICS\n")
    f.write("="*80 + "\n")
    f.write(str(avg_metrics))
print(f"Summary saved to: {summary_file}")

print("\n" + "="*80)
print("LAYER-WISE MEMORY PROFILING")
print("="*80)

# Memory profiling setup
memory_log = []
memory_log_sf = []

def memory_hook(module, input, output, tag=""):
    """Hook to track memory usage during forward pass"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        memory_log.append({
            "layer": f"{tag}{module.__class__.__name__}",
            "allocated_MB": allocated,
            "reserved_MB": reserved,
            "peak_MB": peak
        })

def memory_hook_sf(module, input, output, tag=""):
    """Hook to track memory usage for Semifreddo"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        memory_log_sf.append({
            "layer": f"{tag}{module.__class__.__name__}",
            "allocated_MB": allocated,
            "reserved_MB": reserved,
            "peak_MB": peak
        })

all_mem_logs_seqnn = []
all_mem_logs_sf = []

# Memory profiling loop
for idx, row in merged_df.iterrows():
    if idx % 10 == 0:
        print(f"Memory profiling {idx+1}/{total_rows}...")
    
    # 1. Load original sequence X (one-hot)
    X = torch.load(
        f"/scratch1/smaruj/generate_genomic_boundary/ohe_X/{row['fold']}/{row['chrom']}_{row['centered_start']}_{row['centered_end']}_X.pt",
        weights_only=True
    ).squeeze(0)
    
    # 2. One-hot encode CTCF
    ctcf_seq = row["ctcf_seq"]
    ctcf_ohe = ohe(ctcf_seq)
    
    # 3. Insert CTCF in the middle of X
    seq_len = X.shape[1]
    mid = seq_len // 2
    ctcf_len = ctcf_ohe.shape[1]
    start_idx = mid - ctcf_len // 2
    end_idx = start_idx + ctcf_len
    X[:, start_idx:end_idx] = ctcf_ohe
    
    # Register hooks for full model
    for name, module in model.named_modules():
        module.register_forward_hook(
            lambda m, i, o, name=name: memory_hook(m, i, o, tag=f"{name}: ")
        )
    
    # 4. Full model prediction
    memory_log.clear()
    with torch.no_grad():
        pred_seqnn = model(X.unsqueeze(0).to(device))
    df_mem = pd.DataFrame(memory_log)
    
    # 5. Prepare padded slice for Semifreddo
    central_start = mid - bin_size // 2
    central_end = central_start + bin_size
    slice_start = max(0, central_start - 2 * bin_size)
    slice_end = min(seq_len, central_end + 2 * bin_size)
    slice_0_padded_seq = X[:, slice_start:slice_end].clone()
    
    edited_indices_slice_0 = [256]
    slice_0_padded_seq = slice_0_padded_seq.unsqueeze(0)
    
    memory_log_sf.clear()
    
    # 6. Run Semifreddo
    precom_tensor = torch.load(
        f"/scratch1/smaruj/generate_genomic_boundary/tower_outputs/{row['fold']}/{row['chrom']}_{row['centered_start']}_{row['centered_end']}_tower_out.pt",
        weights_only=True
    ).to(device)
    
    # Register hooks for Semifreddo
    for name, module in model.named_modules():
        module.register_forward_hook(
            lambda m, i, o, name=name: memory_hook_sf(m, i, o, tag=f"{name}: ")
        )
    
    semifreddo_instance = Semifreddo(
        model=model,
        slice_0_padded_seq=slice_0_padded_seq,
        edited_indices_slice_0=edited_indices_slice_0,
        precomputed_full_output=precom_tensor,
        cropping_applied=64,
        batch_size=1
    )
    
    with torch.no_grad():
        pred_sf = semifreddo_instance.forward()
    df_mem_sf = pd.DataFrame(memory_log_sf)
    
    # Save memory logs with run info
    df_mem["run"] = idx
    df_mem_sf["run"] = idx
    
    all_mem_logs_seqnn.append(df_mem)
    all_mem_logs_sf.append(df_mem_sf)

print("\n" + "="*80)
print("GENERATING MEMORY PLOTS")
print("="*80)

# Concatenate results
df_all_seqnn = pd.concat(all_mem_logs_seqnn, ignore_index=True)
df_all_sf = pd.concat(all_mem_logs_sf, ignore_index=True)

# Average across runs
df_avg_seqnn = df_all_seqnn.groupby("layer", as_index=False)[["allocated_MB", "reserved_MB", "peak_MB"]].mean()
df_avg_sf = df_all_sf.groupby("layer", as_index=False)[["allocated_MB", "reserved_MB", "peak_MB"]].mean()

# Get ordered layer names
ordered_layer_names = [
    f"{name}: {module.__class__.__name__}"
    for name, module in model.named_modules()
    if name != ""
]

set_seq = set(df_avg_seqnn["layer"].unique())
set_sf = set(df_avg_sf["layer"].unique())
common_layers = [name for name in ordered_layer_names if (name in set_seq and name in set_sf)]

df_avg_seqnn_ordered = df_avg_seqnn.set_index("layer").reindex(common_layers).reset_index()
df_avg_sf_ordered = df_avg_sf.set_index("layer").reindex(common_layers).reset_index()

# Plot 1: With layer names
plt.figure(figsize=(20, 5))
x = range(len(common_layers))

plt.plot(x, df_avg_seqnn_ordered["allocated_MB"], marker="o", label="Full Akita")
plt.plot(x, df_avg_sf_ordered["allocated_MB"], marker="o", label="Semifreddo")

plt.xticks(x, df_avg_seqnn_ordered["layer"], rotation=90, fontsize=8)
plt.xlabel("Layer (network order)")
plt.ylabel("Allocated memory (MB)")
plt.title(f"Average allocated memory per layer (n={len(df_all_seqnn['run'].unique())} runs)")
plt.legend()
plt.tight_layout()

plot1_file = os.path.join(OUTPUT_DIR, f"fold{FOLD}_memory_layer_names.svg")
plt.savefig(plot1_file, format="svg")
print(f"Saved plot: {plot1_file}")
plt.close()

# Plot 2: With layer indices
plt.figure(figsize=(20, 3))
x = range(len(common_layers))

plt.plot(x, df_avg_seqnn_ordered["allocated_MB"], marker="o", label="Full Akita")
plt.plot(x, df_avg_sf_ordered["allocated_MB"], marker="o", label="Semifreddo")

plt.xlabel("Layer index (forward order)")
plt.ylabel("Allocated memory (MB)")
plt.title(f"Average allocated memory per layer (n={len(df_all_seqnn['run'].unique())} runs)")
plt.legend()
plt.tight_layout()

plot2_file = os.path.join(OUTPUT_DIR, f"fold{FOLD}_memory_layer_indices.svg")
plt.savefig(plot2_file, format="svg")
print(f"Saved plot: {plot2_file}")
plt.close()

# Save memory profiling data
mem_seqnn_file = os.path.join(OUTPUT_DIR, f"fold{FOLD}_memory_seqnn.csv")
mem_sf_file = os.path.join(OUTPUT_DIR, f"fold{FOLD}_memory_sf.csv")

df_all_seqnn.to_csv(mem_seqnn_file, index=False)
df_all_sf.to_csv(mem_sf_file, index=False)

print(f"Saved memory data: {mem_seqnn_file}")
print(f"Saved memory data: {mem_sf_file}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"All results saved to: {OUTPUT_DIR}")