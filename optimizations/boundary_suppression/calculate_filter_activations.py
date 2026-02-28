#!/usr/bin/env python3
"""
Calculate first convolutional layer filter activations for original vs optimized sequences.
Uses maximum activation per filter (length-independent metric).

Processes all folds and generates comprehensive statistics.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from akita_model.model import SeqNN


# ==========================
# Dataset Classes
# ==========================

class OriginalDataset(Dataset):
    def __init__(self, coord_df, init_seq_path):
        self.coords = coord_df
        self.init_seq_path = init_seq_path
    
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row = self.coords.iloc[idx]
        chrom = row["chrom"]
        start = row["centered_start"]
        end = row["centered_end"]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        X = torch.load(f"{self.init_seq_path}{chrom}_{start}_{end}_X.pt", 
                       weights_only=True, map_location=device)
        X = X.squeeze(0)
        return X
    

class OptimizedDataset(Dataset):
    def __init__(self, coord_df, init_seq_path, slice_path, 
                 slice=256, cropping=64, bin_size=2048):
        self.coords = coord_df
        self.init_seq_path = init_seq_path
        self.slice_path = slice_path
        self.slice = slice
        self.cropping = cropping
        self.bin_size = bin_size
        
    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        row = self.coords.iloc[idx]
        chrom = row["chrom"]
        start = row["centered_start"]
        end = row["centered_end"]
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        X = torch.load(f"{self.init_seq_path}{chrom}_{start}_{end}_X.pt", 
                       weights_only=True, map_location=device)
        slice_tensor = torch.load(f"{self.slice_path}{chrom}_{start}_{end}_slice.pt", 
                                  weights_only=True, map_location=device)
        
        edit_start = (self.slice + self.cropping) * self.bin_size
        edit_end = edit_start + self.bin_size
        
        editedX = X.clone()
        editedX[:, :, edit_start:edit_end] = slice_tensor
        editedX = editedX.squeeze(0)
        
        return editedX


# ==========================
# Activation Calculation
# ==========================

def extract_first_conv_layer(model):
    """
    Extract the first convolutional layer from the model.
    
    The first conv is in conv_block_1, which contains:
    - conv (the actual convolution)
    - batch norm
    - activation
    - pooling
    
    We want activations BEFORE pooling for maximum spatial resolution.
    """
    # Get the conv + batch norm + activation, but NOT pooling
    # This gives us the full spatial resolution before max pooling
    conv_block = model.conv_block_1
    return conv_block


def calculate_max_activations_per_fold(fold, model, device, batch_size=4, successful_seqs_df=None):
    """
    Calculate maximum activation per filter for one fold.
    
    Args:
        successful_seqs_df: DataFrame with all successful sequences
    
    Returns:
        orig_activations: (n_sequences, 128)
        opt_activations: (n_sequences, 128)
        metadata: DataFrame with sequence info
    """
    # Filter successful sequences to this fold
    df = successful_seqs_df[successful_seqs_df['fold'] == fold].copy()
    
    if len(df) == 0:
        print(f"\nProcessing fold {fold}: 0 successful sequences - skipping")
        return None, None, None
    
    print(f"\nProcessing fold {fold}: {len(df)} successful sequences")
    
    # Create datasets
    orig_dataset = OriginalDataset(
        df, 
        f"/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/suppressing_CTCFs/ohe_X/fold{fold}/"
    )
    orig_loader = DataLoader(orig_dataset, batch_size=batch_size, shuffle=False)
    
    opt_dataset = OptimizedDataset(
        df,
        f"/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/suppressing_CTCFs/ohe_X/fold{fold}/",
        f"/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/suppressing_CTCFs/results_repeated/fold{fold}/"
    )
    opt_loader = DataLoader(opt_dataset, batch_size=batch_size, shuffle=False)
    
    # Extract first conv block
    conv_block = extract_first_conv_layer(model)
    conv_block.eval()
    
    # Storage for activations
    all_orig_max = []
    all_opt_max = []
    
    print("  Calculating activations...")
    with torch.no_grad():
        for batch_idx, (orig_batch, opt_batch) in enumerate(zip(orig_loader, opt_loader)):
            orig_batch = orig_batch.to(device)
            opt_batch = opt_batch.to(device)
            
            # Forward through first conv block (up to but not including pooling)
            # We want the activation after conv + batch norm + ReLU
            # Input: [B, 4, L]
            # After conv: [B, 128, L']
            
            # Get activations from the conv block
            # The conv_block applies: conv -> batch_norm -> activation -> pool
            # We need to separate this to get activations before pooling
            
            x_orig = orig_batch
            x_opt = opt_batch
            
            # Apply conv
            x_orig = conv_block.conv(x_orig)
            x_opt = conv_block.conv(x_opt)
            
            # Apply batch norm
            if hasattr(conv_block, 'batch_norm'):
                x_orig = conv_block.batch_norm(x_orig)
                x_opt = conv_block.batch_norm(x_opt)
            
            # Apply activation (ReLU)
            if hasattr(conv_block, 'activation'):
                x_orig = conv_block.activation(x_orig)
                x_opt = conv_block.activation(x_opt)
            
            # NOW we have shape [B, 128, L'] where L' is spatial dimension
            # Calculate max activation per filter (max over spatial dimension)
            max_orig = torch.max(x_orig, dim=2)[0]  # [B, 128]
            max_opt = torch.max(x_opt, dim=2)[0]    # [B, 128]
            
            all_orig_max.append(max_orig.cpu().numpy())
            all_opt_max.append(max_opt.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                n_processed = (batch_idx + 1) * batch_size
                print(f"    Processed {n_processed}/{len(df)} sequences")
    
    # Concatenate all batches
    orig_activations = np.concatenate(all_orig_max, axis=0)  # (n_seq, 128)
    opt_activations = np.concatenate(all_opt_max, axis=0)    # (n_seq, 128)
    
    # Add fold information to metadata
    df['fold'] = fold
    
    print(f"  Completed: {orig_activations.shape[0]} sequences, {orig_activations.shape[1]} filters")
    
    return orig_activations, opt_activations, df


def calculate_all_folds(folds, model, device, batch_size=4, save_per_fold=True, successful_seqs_df=None):
    """
    Calculate activations for all folds.
    
    Args:
        folds: list of fold numbers (e.g., [0, 1, 2, 3, 4, 5, 6, 7])
        save_per_fold: if True, save intermediate results per fold
        successful_seqs_df: DataFrame with only successful sequences to analyze
    
    Returns:
        all_orig: (total_sequences, 128)
        all_opt: (total_sequences, 128)
        all_metadata: DataFrame with all sequence info
    """
    all_orig_activations = []
    all_opt_activations = []
    all_metadata = []
    
    for fold in folds:
        result = calculate_max_activations_per_fold(
            fold, model, device, batch_size, successful_seqs_df
        )
        
        # Handle case where fold had no successful sequences
        if result[0] is None:
            continue
            
        orig_act, opt_act, metadata = result
        
        all_orig_activations.append(orig_act)
        all_opt_activations.append(opt_act)
        all_metadata.append(metadata)
        
        # Save per-fold results
        if save_per_fold:
            fold_results = metadata.copy()
            
            # Add activation columns for each filter
            for f_idx in range(128):
                fold_results[f'filter_{f_idx}_orig'] = orig_act[:, f_idx]
                fold_results[f'filter_{f_idx}_opt'] = opt_act[:, f_idx]
                fold_results[f'filter_{f_idx}_diff'] = opt_act[:, f_idx] - orig_act[:, f_idx]
            
            output_file = f"filter_activations_fold{fold}.tsv"
            fold_results.to_csv(output_file, sep='\t', index=False)
            print(f"  Saved per-fold results: {output_file}")
    
    # Concatenate all folds
    all_orig = np.concatenate(all_orig_activations, axis=0)
    all_opt = np.concatenate(all_opt_activations, axis=0)
    all_meta = pd.concat(all_metadata, ignore_index=True)
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {all_orig.shape[0]} sequences across {len(folds)} folds")
    print(f"{'='*70}\n")
    
    return all_orig, all_opt, all_meta


# ==========================
# Statistical Analysis
# ==========================

def compare_activations(original_activations, optimized_activations, output_file='filter_comparison.tsv'):
    """
    Compare filter activations between original and optimized sequences.
    
    Args:
        original_activations: (n_sequences, 128)
        optimized_activations: (n_sequences, 128)
    
    Returns:
        results_df: DataFrame with statistics for each filter
    """
    n_sequences, n_filters = original_activations.shape
    
    print(f"Comparing activations for {n_sequences} sequences, {n_filters} filters...")
    
    results = []
    
    for filter_idx in range(n_filters):
        orig = original_activations[:, filter_idx]
        opt = optimized_activations[:, filter_idx]
        
        # Basic statistics
        orig_mean = np.mean(orig)
        orig_std = np.std(orig)
        orig_median = np.median(orig)
        opt_mean = np.mean(opt)
        opt_std = np.std(opt)
        opt_median = np.median(opt)
        
        # Differences
        mean_diff = opt_mean - orig_mean
        median_diff = opt_median - orig_median
        
        # Fold change (avoid division by zero)
        if orig_mean > 1e-10:
            fold_change = opt_mean / orig_mean
            log2_fold_change = np.log2(fold_change)
        else:
            fold_change = np.inf if opt_mean > 1e-10 else 1.0
            log2_fold_change = np.inf if opt_mean > 1e-10 else 0.0
        
        # Paired statistical tests (since we have matched sequences)
        t_stat, t_pval = stats.ttest_rel(orig, opt)
        wilcoxon_stat, wilcoxon_pval = stats.wilcoxon(orig, opt, alternative='two-sided')
        
        # Effect size (Cohen's d for paired samples)
        diff = opt - orig
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)
        
        # Correlation between original and optimized
        if orig_std > 0 and opt_std > 0:
            correlation, corr_pval = stats.pearsonr(orig, opt)
        else:
            correlation = 0.0
            corr_pval = 1.0
        
        # Percentage of sequences where activation increased
        pct_increased = 100 * np.mean(diff > 0)
        
        results.append({
            'filter_idx': filter_idx,
            'orig_mean': orig_mean,
            'orig_std': orig_std,
            'orig_median': orig_median,
            'opt_mean': opt_mean,
            'opt_std': opt_std,
            'opt_median': opt_median,
            'mean_diff': mean_diff,
            'median_diff': median_diff,
            'abs_mean_diff': np.abs(mean_diff),
            'fold_change': fold_change,
            'log2_fold_change': log2_fold_change,
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'wilcoxon_statistic': wilcoxon_stat,
            'wilcoxon_pvalue': wilcoxon_pval,
            'cohens_d': cohens_d,
            'correlation': correlation,
            'correlation_pvalue': corr_pval,
            'pct_increased': pct_increased,
            'orig_min': np.min(orig),
            'orig_max': np.max(orig),
            'opt_min': np.min(opt),
            'opt_max': np.max(opt)
        })
        
        if (filter_idx + 1) % 32 == 0:
            print(f"  Processed {filter_idx + 1}/{n_filters} filters")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by absolute mean difference (most differential first)
    results_df = results_df.sort_values('abs_mean_diff', ascending=False)
    
    # Add significance markers
    results_df['significant_001'] = results_df['t_pvalue'] < 0.001
    results_df['significant_01'] = results_df['t_pvalue'] < 0.01
    results_df['significant_05'] = results_df['t_pvalue'] < 0.05
    
    # Save results
    results_df.to_csv(output_file, sep='\t', index=False)
    print(f"\nFilter comparison saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("FILTER COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"Total filters: {n_filters}")
    print(f"Significantly different (p<0.05): {results_df['significant_05'].sum()}")
    print(f"Significantly different (p<0.01): {results_df['significant_01'].sum()}")
    print(f"Significantly different (p<0.001): {results_df['significant_001'].sum()}")
    print(f"\nTop 10 most differential filters:")
    print(results_df[['filter_idx', 'mean_diff', 'fold_change', 't_pvalue', 'cohens_d']].head(10).to_string(index=False))
    
    return results_df


# ==========================
# Visualization
# ==========================

def create_visualizations(results_df, orig_activations, opt_activations, output_prefix='filter_analysis'):
    """
    Create comprehensive visualizations of filter activation analysis.
    """
    print("\nCreating visualizations...")
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Volcano plot
    ax1 = fig.add_subplot(gs[0, 0:2])
    
    # Calculate -log10(p-value)
    neg_log_p = -np.log10(results_df['t_pvalue'].values + 1e-300)
    
    # Color by significance and direction
    colors = []
    for _, row in results_df.iterrows():
        if row['t_pvalue'] < 0.001:
            if row['mean_diff'] > 0:
                colors.append('red')  # Increased, significant
            else:
                colors.append('blue')  # Decreased, significant
        elif row['t_pvalue'] < 0.05:
            if row['mean_diff'] > 0:
                colors.append('orange')
            else:
                colors.append('lightblue')
        else:
            colors.append('gray')
    
    ax1.scatter(results_df['log2_fold_change'], neg_log_p, c=colors, alpha=0.6, s=50)
    ax1.axhline(-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='p=0.05')
    ax1.axhline(-np.log10(0.001), color='black', linestyle='--', alpha=0.8, label='p=0.001')
    ax1.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('log2(Fold Change)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('-log10(p-value)', fontsize=12, fontweight='bold')
    ax1.set_title('Volcano Plot: Filter Activation Changes', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Annotate top filters
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        ax1.annotate(f"F{row['filter_idx']}", 
                    (row['log2_fold_change'], -np.log10(row['t_pvalue'] + 1e-300)),
                    fontsize=8, alpha=0.7)
    
    # 2. Top differential filters bar plot
    ax2 = fig.add_subplot(gs[0, 2:4])
    top_20 = results_df.head(20)
    
    colors_bar = ['red' if d > 0 else 'blue' for d in top_20['mean_diff']]
    y_pos = np.arange(len(top_20))
    ax2.barh(y_pos, top_20['mean_diff'], color=colors_bar, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"F{idx}" for idx in top_20['filter_idx']], fontsize=9)
    ax2.set_xlabel('Mean Activation Difference (Opt - Orig)', fontsize=11, fontweight='bold')
    ax2.set_title('Top 20 Differential Filters', fontsize=14, fontweight='bold')
    ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # 3. Distribution of top filter
    ax3 = fig.add_subplot(gs[1, 0])
    top_filter_idx = results_df.iloc[0]['filter_idx']
    
    orig_top = orig_activations[:, top_filter_idx]
    opt_top = opt_activations[:, top_filter_idx]
    
    bins = np.linspace(min(orig_top.min(), opt_top.min()),
                      max(orig_top.max(), opt_top.max()), 30)
    ax3.hist(orig_top, bins=bins, alpha=0.6, label='Original', 
             color='steelblue', density=True, edgecolor='black')
    ax3.hist(opt_top, bins=bins, alpha=0.6, label='Optimized', 
             color='coral', density=True, edgecolor='black')
    ax3.axvline(orig_top.mean(), color='blue', linestyle='--', linewidth=2)
    ax3.axvline(opt_top.mean(), color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Max Activation', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax3.set_title(f'Top Filter (F{top_filter_idx})', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Scatter plot of top filter
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(orig_top, opt_top, alpha=0.5, s=30)
    
    # Add diagonal line
    max_val = max(orig_top.max(), opt_top.max())
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
    
    ax4.set_xlabel('Original Activation', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Optimized Activation', fontsize=11, fontweight='bold')
    ax4.set_title(f'F{top_filter_idx} Correlation', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Effect size distribution
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(results_df['cohens_d'], bins=30, edgecolor='black', alpha=0.7, color='purple')
    ax5.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax5.axvline(0.2, color='green', linestyle='--', alpha=0.5, label='Small')
    ax5.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
    ax5.axvline(0.8, color='red', linestyle='--', alpha=0.5, label='Large')
    ax5.set_xlabel("Cohen's d", fontsize=11, fontweight='bold')
    ax5.set_ylabel('Number of Filters', fontsize=11, fontweight='bold')
    ax5.set_title('Effect Size Distribution', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Percentage increased
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.hist(results_df['pct_increased'], bins=20, edgecolor='black', alpha=0.7, color='teal')
    ax6.axvline(50, color='black', linestyle='--', alpha=0.5, label='50%')
    ax6.set_xlabel('% Sequences with Increased Activation', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Number of Filters', fontsize=11, fontweight='bold')
    ax6.set_title('Consistency of Changes', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Heatmap of top filters
    ax7 = fig.add_subplot(gs[2, :2])
    top_filters = results_df.head(20)['filter_idx'].values
    
    # Get activations for top filters
    orig_top_filt = orig_activations[:, top_filters]
    opt_top_filt = opt_activations[:, top_filters]
    
    # Calculate z-scores for better visualization (normalize each filter independently)
    orig_z = (orig_top_filt - orig_top_filt.mean(axis=0)) / (orig_top_filt.std(axis=0) + 1e-10)
    opt_z = (opt_top_filt - opt_top_filt.mean(axis=0)) / (opt_top_filt.std(axis=0) + 1e-10)
    
    # Combine for heatmap (rows=filters, columns=sequences)
    combined = np.concatenate([orig_z.T, opt_z.T], axis=1)
    
    # Subsample sequences for visualization if too many
    n_seqs_to_show = min(100, orig_activations.shape[0])
    if combined.shape[1] > n_seqs_to_show:
        # Sample evenly from original and optimized
        n_per_group = n_seqs_to_show // 2
        orig_sample_idx = np.linspace(0, orig_z.shape[0]-1, n_per_group, dtype=int)
        opt_sample_idx = np.linspace(0, opt_z.shape[0]-1, n_per_group, dtype=int)
        
        combined = np.concatenate([orig_z.T[:, orig_sample_idx], 
                                   opt_z.T[:, opt_sample_idx]], axis=1)
    
    im = ax7.imshow(combined, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax7.set_yticks(np.arange(len(top_filters)))
    ax7.set_yticklabels([f"F{idx}" for idx in top_filters], fontsize=8)
    ax7.set_xlabel('Sequence Index', fontsize=11, fontweight='bold')
    ax7.set_title('Top 20 Filters (Z-scored)', fontsize=12, fontweight='bold')
    
    # Add vertical line separating orig/opt
    n_orig_shown = combined.shape[1] // 2 if combined.shape[1] > 1 else 1
    ax7.axvline(n_orig_shown - 0.5, color='yellow', linewidth=2, linestyle='--')
    ax7.text(n_orig_shown/2, -1.5, 'Original', ha='center', fontweight='bold', fontsize=10)
    ax7.text(n_orig_shown + (combined.shape[1]-n_orig_shown)/2, -1.5, 
             'Optimized', ha='center', fontweight='bold', fontsize=10)
    
    plt.colorbar(im, ax=ax7, label='Z-score')
    
    # 8. Summary statistics text
    ax8 = fig.add_subplot(gs[2, 2:])
    ax8.axis('off')
    
    summary_text = f"""
SUMMARY STATISTICS
{'='*50}

Dataset:
  Total sequences: {orig_activations.shape[0]}
  Total filters: {orig_activations.shape[1]}

Significance:
  p < 0.001: {results_df['significant_001'].sum()} filters ({100*results_df['significant_001'].mean():.1f}%)
  p < 0.01:  {results_df['significant_01'].sum()} filters ({100*results_df['significant_01'].mean():.1f}%)
  p < 0.05:  {results_df['significant_05'].sum()} filters ({100*results_df['significant_05'].mean():.1f}%)

Effect Sizes (|Cohen's d| > 0.5):
  {(np.abs(results_df['cohens_d']) > 0.5).sum()} filters ({100*(np.abs(results_df['cohens_d']) > 0.5).mean():.1f}%)

Top Filter:
  Filter {results_df.iloc[0]['filter_idx']}
  Mean diff: {results_df.iloc[0]['mean_diff']:.4f}
  Fold change: {results_df.iloc[0]['fold_change']:.3f}x
  p-value: {results_df.iloc[0]['t_pvalue']:.4e}
  Cohen's d: {results_df.iloc[0]['cohens_d']:.3f}
  % increased: {results_df.iloc[0]['pct_increased']:.1f}%

Overall Trends:
  Filters with increased activation: {(results_df['mean_diff'] > 0).sum()}
  Filters with decreased activation: {(results_df['mean_diff'] < 0).sum()}
  Mean fold change (all filters): {results_df['fold_change'].mean():.3f}x
"""
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save figure
    output_file = f'{output_prefix}_comprehensive.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.close()


# ==========================
# Main Pipeline
# ==========================

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load successful sequences file (contains all folds)
    successful_seqs_path = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/suppressing_CTCFs/results_repeated/only_successful_seqs.tsv"
    print(f"\nLoading successful sequences from: {successful_seqs_path}")
    successful_seqs = pd.read_csv(successful_seqs_path, sep='\t')
    print(f"Total successful sequences: {len(successful_seqs)}")
    print(f"Sequences per fold:")
    for fold in sorted(successful_seqs['fold'].unique()):
        n_seqs = (successful_seqs['fold'] == fold).sum()
        print(f"  Fold {fold}: {n_seqs} sequences")
    
    # Load model
    print("\nLoading model...")
    model = SeqNN()
    model_path = (
        "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Process all folds
    folds = args.folds if args.folds else list(range(8))  # Default: folds 0-7
    print(f"\nProcessing folds: {folds}")
    
    orig_activations, opt_activations, metadata = calculate_all_folds(
        folds=folds,
        model=model,
        device=device,
        batch_size=args.batch_size,
        save_per_fold=args.save_per_fold,
        successful_seqs_df=successful_seqs
    )
    
    # Save combined activations
    print("\nSaving combined activations...")
    np.save('/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/suppressing_CTCFs/results_repeated/all_original_activations.npy', orig_activations)
    np.save('/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/suppressing_CTCFs/results_repeated/all_optimized_activations.npy', opt_activations)
    metadata.to_csv('/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/suppressing_CTCFs/results_repeated/all_sequences_metadata.tsv', sep='\t', index=False)
    print("Saved:")
    print("  - all_original_activations.npy")
    print("  - all_optimized_activations.npy")
    print("  - all_sequences_metadata.tsv")
    
    # Compare activations
    results_df = compare_activations(
        orig_activations, 
        opt_activations,
        output_file='/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/suppressing_CTCFs/results_repeated/filter_comparison_results.tsv'
    )
    
    # Create visualizations
    create_visualizations(
        results_df,
        orig_activations,
        opt_activations,
        output_prefix='filter_analysis'
    )
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print("\nOutput files:")
    print("  - filter_comparison_results.tsv (detailed statistics for all 128 filters)")
    print("  - filter_analysis_comprehensive.png (visualization)")
    print("  - all_original_activations.npy (raw activation data)")
    print("  - all_optimized_activations.npy (raw activation data)")
    print("  - all_sequences_metadata.tsv (sequence information)")
    if args.save_per_fold:
        print("  - filter_activations_fold*.tsv (per-fold results)")


# ==========================
# Entry Point
# ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze first conv layer filter activations for boundary suppression"
    )
    parser.add_argument("--folds", type=int, nargs='+', 
                        help="Fold numbers to process (default: 0-7)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for processing (default: 4)")
    parser.add_argument("--save-per-fold", action='store_true',
                        help="Save intermediate results for each fold")
    
    args = parser.parse_args()
    main(args)