#!/usr/bin/env python3
"""
Extract SINE B2 sequences and identify which filters are most activated.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
from torch.utils.data import Dataset, DataLoader
from pyfaidx import Fasta
import gzip
import sys
import os

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from akita_model.model import SeqNN


def examine_rmsk_file(rmsk_file, n_lines=20):
    """Look at the structure of the RepeatMasker file."""
    print("="*70)
    print("EXAMINING REPEATMASKER FILE")
    print("="*70)
    print(f"File: {rmsk_file}\n")
    
    opener = gzip.open if rmsk_file.endswith('.gz') else open
    mode = 'rt' if rmsk_file.endswith('.gz') else 'r'
    
    with opener(rmsk_file, mode) as f:
        print("First 20 lines:")
        print("-"*70)
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            print(f"{i:3d}: {line.rstrip()}")
    
    print("\n" + "="*70)


def load_repeatmasker_data(rmsk_file):
    """Load RepeatMasker data and identify SINE B2 elements."""
    print("\nLoading RepeatMasker data...")
    
    if rmsk_file.endswith('.gz'):
        df = pd.read_csv(rmsk_file, sep='\t', compression='gzip', header=None)
    else:
        df = pd.read_csv(rmsk_file, sep='\t', header=None)
    
    print(f"Loaded {len(df)} repeat annotations")
    print(f"Columns: {df.shape[1]}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # UCSC rmsk format
    if df.shape[1] >= 17:
        df.columns = [
            'bin', 'swScore', 'milliDiv', 'milliDel', 'milliIns',
            'genoName', 'genoStart', 'genoEnd', 'genoLeft', 'strand',
            'repName', 'repClass', 'repFamily', 'repStart', 'repEnd', 'repLeft', 'id'
        ]
    
    print(f"\nRepeat classes found:")
    if 'repClass' in df.columns:
        print(df['repClass'].value_counts().head(10))
    
    print(f"\nRepeat names containing 'B2':")
    if 'repName' in df.columns:
        b2_names = df[df['repName'].str.contains('B2', case=False, na=False)]['repName'].value_counts()
        print(b2_names.head(20))
    
    return df


def extract_sine_b2_sequences(rmsk_df, genome_fasta, min_length=150, max_length=250,
                               max_sequences=500, random_seed=42):
    """Extract SINE B2 sequences from the genome."""
    print("\n" + "="*70)
    print("EXTRACTING SINE B2 SEQUENCES")
    print("="*70)
    
    # Filter for TRUE SINE B2 elements only (not LTRs, LINEs, etc. with B2 in name)
    # The three main SINE B2 families in mouse
    true_sine_b2 = ['B2_Mm2', 'B2_Mm1t', 'B2_Mm1a']
    
    sine_b2_mask = rmsk_df['repName'].isin(true_sine_b2)
    
    sine_b2_df = rmsk_df[sine_b2_mask].copy()
    print(f"Found {len(sine_b2_df)} true SINE B2 annotations")
    print(f"  B2_Mm2:  {(sine_b2_df['repName'] == 'B2_Mm2').sum()}")
    print(f"  B2_Mm1t: {(sine_b2_df['repName'] == 'B2_Mm1t').sum()}")
    print(f"  B2_Mm1a: {(sine_b2_df['repName'] == 'B2_Mm1a').sum()}")
    
    # Filter by length
    sine_b2_df['length'] = sine_b2_df['genoEnd'] - sine_b2_df['genoStart']
    sine_b2_df = sine_b2_df[(sine_b2_df['length'] >= min_length) & 
                            (sine_b2_df['length'] <= max_length)]
    print(f"After length filtering ({min_length}-{max_length} bp): {len(sine_b2_df)}")
    
    # Sample if too many
    if len(sine_b2_df) > max_sequences:
        sine_b2_df = sine_b2_df.sample(n=max_sequences, random_state=random_seed)
        print(f"Sampled {max_sequences} sequences")
    
    # Load genome
    print(f"\nLoading genome: {genome_fasta}")
    genome = Fasta(genome_fasta)
    
    # Extract sequences
    sequences = []
    metadata_list = []
    
    print("Extracting sequences...")
    for idx, row in sine_b2_df.iterrows():
        chrom = row['genoName']
        start = int(row['genoStart'])
        end = int(row['genoEnd'])
        
        try:
            seq = str(genome[chrom][start:end].seq).upper()
            if seq.count('N') / len(seq) > 0.1:
                continue
            
            sequences.append(seq)
            metadata_list.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'strand': row['strand'],
                'rep_name': row['repName'],
                'length': len(seq)
            })
            
        except Exception as e:
            continue
        
        if len(sequences) % 100 == 0:
            print(f"  Extracted {len(sequences)} sequences...")
    
    metadata_df = pd.DataFrame(metadata_list)
    
    print(f"\nExtracted {len(sequences)} SINE B2 sequences")
    print(f"Length: mean={metadata_df['length'].mean():.1f}, median={metadata_df['length'].median():.1f}")
    
    # GC content
    gc_contents = [(seq.count('G') + seq.count('C')) / len(seq) for seq in sequences]
    metadata_df['gc_content'] = gc_contents
    print(f"GC content: mean={np.mean(gc_contents):.3f}")
    
    return sequences, metadata_df


def one_hot_encode_sequences(sequences, fixed_length=None):
    """One-hot encode DNA sequences."""
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}
    
    max_length = fixed_length if fixed_length else max(len(s) for s in sequences)
    n_sequences = len(sequences)
    one_hot = torch.zeros((n_sequences, 4, max_length), dtype=torch.float32)
    
    for i, seq in enumerate(sequences):
        seq_to_encode = seq[:max_length] if len(seq) > max_length else seq
        for j, base in enumerate(seq_to_encode):
            idx = base_to_index.get(base, -1)
            if idx >= 0:
                one_hot[i, idx, j] = 1.0
    
    return one_hot


def calculate_sine_b2_activations(sequences, model, device, batch_size=32):
    """Calculate filter activations for SINE B2 sequences."""
    print("\n" + "="*70)
    print("CALCULATING SINE B2 FILTER ACTIVATIONS")
    print("="*70)
    
    print("One-hot encoding sequences...")
    one_hot_seqs = one_hot_encode_sequences(sequences)
    print(f"One-hot shape: {one_hot_seqs.shape}")
    
    dataset = torch.utils.data.TensorDataset(one_hot_seqs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    first_conv = model.conv_block_1.conv
    first_conv.eval()
    
    all_max_activations = []
    
    print("Processing batches...")
    with torch.no_grad():
        for batch_idx, (batch,) in enumerate(dataloader):
            batch = batch.to(device)
            
            x = first_conv(batch)
            if hasattr(model.conv_block_1, 'batch_norm'):
                x = model.conv_block_1.batch_norm(x)
            if hasattr(model.conv_block_1, 'activation'):
                x = model.conv_block_1.activation(x)
            
            max_act = torch.max(x, dim=2)[0]  # [B, 128]
            all_max_activations.append(max_act.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size} sequences...")
    
    activations = np.concatenate(all_max_activations, axis=0)
    print(f"\nActivations shape: {activations.shape}")
    
    return activations


def analyze_top_filters(sine_b2_activations, top_n=20):
    """Identify which filters are most activated by SINE B2."""
    print("\n" + "="*70)
    print("TOP FILTERS ACTIVATED BY SINE B2")
    print("="*70)
    
    n_filters = sine_b2_activations.shape[1]
    
    results = []
    for filter_idx in range(n_filters):
        activations = sine_b2_activations[:, filter_idx]
        results.append({
            'filter_idx': filter_idx,
            'mean': activations.mean(),
            'median': np.median(activations),
            'std': activations.std(),
            'max': activations.max(),
            'pct_active': 100 * (activations > 0.1).sum() / len(activations)
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean', ascending=False)
    
    print(f"\nTop {top_n} filters by mean activation:")
    print(results_df.head(top_n).to_string(index=False))
    
    results_df.to_csv('sine_b2_filter_stats.tsv', sep='\t', index=False)
    print(f"\nSaved: sine_b2_filter_stats.tsv")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bar plot
    top_filters = results_df.head(top_n)
    axes[0, 0].barh(range(len(top_filters)), top_filters['mean'], color='steelblue', alpha=0.7)
    axes[0, 0].set_yticks(range(len(top_filters)))
    axes[0, 0].set_yticklabels([f"F{idx}" for idx in top_filters['filter_idx']])
    axes[0, 0].set_xlabel('Mean Activation', fontweight='bold')
    axes[0, 0].set_title(f'Top {top_n} Filters', fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(alpha=0.3, axis='x')
    
    # Top filter distribution
    top_filter_idx = int(results_df.iloc[0]['filter_idx'])
    top_activations = sine_b2_activations[:, top_filter_idx]
    axes[0, 1].hist(top_activations, bins=30, edgecolor='black', alpha=0.7, color='purple')
    axes[0, 1].axvline(top_activations.mean(), color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Max Activation', fontweight='bold')
    axes[0, 1].set_title(f'Filter {top_filter_idx} Distribution', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Heatmap
    top_20_indices = results_df.head(20)['filter_idx'].values
    n_show = min(100, sine_b2_activations.shape[0])
    sample = np.random.choice(sine_b2_activations.shape[0], n_show, replace=False)
    heatmap_data = sine_b2_activations[np.sort(sample), :][:, top_20_indices].T
    heatmap_z = (heatmap_data - heatmap_data.mean(axis=1, keepdims=True)) / \
                (heatmap_data.std(axis=1, keepdims=True) + 1e-10)
    
    im = axes[1, 0].imshow(heatmap_z, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1, 0].set_yticks(range(len(top_20_indices)))
    axes[1, 0].set_yticklabels([f"F{idx}" for idx in top_20_indices], fontsize=8)
    axes[1, 0].set_xlabel('Sequence Index', fontweight='bold')
    axes[1, 0].set_title('Top 20 Filters (Z-scored)', fontweight='bold')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Activity rate
    top_10 = results_df.head(10)
    axes[1, 1].barh(range(len(top_10)), top_10['pct_active'], color='coral', alpha=0.7)
    axes[1, 1].set_yticks(range(len(top_10)))
    axes[1, 1].set_yticklabels([f"F{idx}" for idx in top_10['filter_idx']])
    axes[1, 1].set_xlabel('% Active (>0.1)', fontweight='bold')
    axes[1, 1].set_title('Activity Rate', fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('sine_b2_top_filters.png', dpi=300, bbox_inches='tight')
    print("Saved: sine_b2_top_filters.png")
    plt.show()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTop 5 filters:")
    for _, row in results_df.head(5).iterrows():
        print(f"  Filter {int(row['filter_idx']):3d}: mean={row['mean']:.3f}, active={row['pct_active']:.1f}%")
    
    print(f"\nFilters 6, 26, 31 ranking:")
    for filter_idx in [6, 26, 31]:
        row = results_df[results_df['filter_idx'] == filter_idx].iloc[0]
        rank = (results_df['filter_idx'] == filter_idx).values.nonzero()[0][0] + 1
        print(f"  Filter {filter_idx}: rank {rank}/{n_filters}, mean={row['mean']:.3f}")
    
    return results_df


def main():
    print("\n" + "="*70)
    print("SINE B2 FILTER ACTIVATION ANALYSIS")
    print("="*70)
    
    RMSK_FILE = "/project2/fudenber_735/genomes/mm10/database/rmsk.txt.gz"
    GENOME_FASTA = "/project2/fudenber_735/genomes/mm10/mm10.fa"
    MODEL_PATH = (
        "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
    )
    
    # Examine file
    print("\nExamining RepeatMasker file...")
    examine_rmsk_file(RMSK_FILE, n_lines=20)
    input("\nPress Enter to continue...")
    
    # Load and extract
    rmsk_df = load_repeatmasker_data(RMSK_FILE)
    sine_b2_seqs, sine_b2_metadata = extract_sine_b2_sequences(
        rmsk_df, GENOME_FASTA, min_length=150, max_length=250, max_sequences=500
    )
    sine_b2_metadata.to_csv('/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/suppressing_CTCFs/results_repeated/sine_b2_metadata.tsv', sep='\t', index=False)
    print("Saved: sine_b2_metadata.tsv")
    
    # Calculate activations
    print("\nLoading model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SeqNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    sine_b2_activations = calculate_sine_b2_activations(sine_b2_seqs, model, device)
    # np.save('sine_b2_activations.npy', sine_b2_activations)
    # print("Saved: sine_b2_activations.npy")
    
    # Analyze
    analyze_top_filters(sine_b2_activations, top_n=20)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()