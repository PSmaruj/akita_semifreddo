#!/usr/bin/env python3
"""
Save top N filters as MEME format and show their consensus sequences.

This creates a MEME file you can upload to TOMTOM web server.
Also displays the consensus sequence for each filter.
"""

import numpy as np
import pandas as pd
import torch
import sys
import os

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from akita_model.model import SeqNN


def extract_filter_pwms(model, filter_indices):
    """
    Extract PWM matrices from convolutional filters.
    
    Returns:
        dict: {filter_idx: pwm_array} where pwm_array is (kernel_size, 4)
    """
    print("Extracting filter weights from model...")
    
    # Get first conv layer weights
    first_conv = model.conv_block_1.conv
    conv_weights = first_conv.weight.data.cpu().numpy()
    
    # Shape: (out_channels=128, in_channels=4, kernel_size=15)
    # Transpose to (kernel_size, in_channels, out_channels)
    weights = conv_weights.transpose(2, 1, 0)  # (15, 4, 128)
    
    pwms = {}
    for filter_idx in filter_indices:
        # Get weight matrix for this filter: (15, 4)
        filter_weights = weights[:, :, filter_idx]
        
        # Convert to PWM using softmax (makes it interpretable as probabilities)
        exp_weights = np.exp(filter_weights - filter_weights.max(axis=1, keepdims=True))
        pwm = exp_weights / exp_weights.sum(axis=1, keepdims=True)
        
        pwms[filter_idx] = pwm
    
    print(f"Extracted {len(pwms)} filter PWMs")
    return pwms


def pwm_to_consensus(pwm, threshold=0.25):
    """
    Convert PWM to consensus sequence.
    
    Args:
        pwm: numpy array (kernel_size, 4) where columns are A, C, G, T
        threshold: minimum probability to call a base
    
    Returns:
        max_seq: sequence using only maximum probability base
        weighted_seq: sequence with IUPAC codes for ambiguous positions
    """
    bases = ['A', 'C', 'G', 'T']
    
    # IUPAC codes
    iupac = {
        'AC': 'M', 'AG': 'R', 'AT': 'W', 'CG': 'S', 'CT': 'Y', 'GT': 'K',
        'ACG': 'V', 'ACT': 'H', 'AGT': 'D', 'CGT': 'B', 'ACGT': 'N'
    }
    
    max_seq = []
    weighted_seq = []
    
    for pos in range(pwm.shape[0]):
        probs = pwm[pos, :]
        
        # Max base
        max_idx = np.argmax(probs)
        max_base = bases[max_idx]
        max_prob = probs[max_idx]
        max_seq.append(max_base)
        
        # Weighted sequence with ambiguity codes
        above_threshold = probs >= threshold
        selected = ''.join([bases[i] for i in range(4) if above_threshold[i]])
        
        if len(selected) == 1:
            weighted_seq.append(selected)
        elif len(selected) > 1:
            weighted_seq.append(iupac.get(selected, 'N'))
        else:
            # Nothing above threshold, use max but lowercase
            weighted_seq.append(max_base.lower())
    
    return ''.join(max_seq), ''.join(weighted_seq)


def analyze_filter_pwm(filter_idx, pwm):
    """
    Print detailed analysis of a filter PWM.
    """
    bases = ['A', 'C', 'G', 'T']
    
    max_seq, weighted_seq = pwm_to_consensus(pwm, threshold=0.25)
    
    print(f"\n{'='*70}")
    print(f"FILTER {filter_idx}")
    print(f"{'='*70}")
    print(f"Max sequence (highest prob base):  {max_seq}")
    print(f"Weighted sequence (IUPAC codes):   {weighted_seq}")
    
    # GC content
    gc_content = sum([pwm[i, 1] + pwm[i, 2] for i in range(pwm.shape[0])]) / pwm.shape[0]
    print(f"GC content: {gc_content:.1%}")
    
    # Information content
    epsilon = 1e-10
    ic_per_pos = []
    for i in range(pwm.shape[0]):
        entropy = -np.sum(pwm[i, :] * np.log2(pwm[i, :] + epsilon))
        ic = 2 - entropy
        ic_per_pos.append(ic)
    avg_ic = np.mean(ic_per_pos)
    print(f"Average information content: {avg_ic:.2f} bits")
    
    # Most conserved positions
    high_ic_pos = [(i, ic) for i, ic in enumerate(ic_per_pos) if ic > 1.5]
    if high_ic_pos:
        print(f"Highly conserved positions (IC > 1.5):")
        for pos, ic in high_ic_pos[:5]:
            top_base_idx = np.argmax(pwm[pos, :])
            top_base = bases[top_base_idx]
            top_prob = pwm[pos, top_base_idx]
            print(f"  Pos {pos+1:2d}: {top_base} ({top_prob:.1%}, IC={ic:.2f})")
    
    # Check for patterns
    patterns = {
        'G homopolymer (≥4)': max_seq.count('GGGG') > 0 or max_seq.count('GGGGG') > 0,
        'C homopolymer (≥4)': max_seq.count('CCCC') > 0 or max_seq.count('CCCCC') > 0,
        'GC repeat': max_seq.count('GCGC') > 0,
        'CG repeat': max_seq.count('CGCG') > 0,
        'AT-rich (>60%)': (max_seq.count('A') + max_seq.count('T')) / len(max_seq) > 0.6,
        'GC-rich (>60%)': gc_content > 0.6,
    }
    
    detected_patterns = [name for name, present in patterns.items() if present]
    if detected_patterns:
        print(f"Detected patterns: {', '.join(detected_patterns)}")


def create_meme_file(pwms, output_file='top_filters.meme', filter_results=None):
    """
    Create MEME format file from PWMs with ranking information.
    
    Args:
        pwms: dict of {filter_idx: pwm_array}
        output_file: output filename
        filter_results: optional DataFrame with filter statistics
    """
    with open(output_file, 'w') as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
        
        for filter_idx in sorted(pwms.keys()):
            pwm = pwms[filter_idx]
            kernel_size = pwm.shape[0]
            
            # Get consensus
            max_seq, _ = pwm_to_consensus(pwm)
            
            # Get ranking info if available
            rank_info = ""
            if filter_results is not None:
                row = filter_results[filter_results['filter_idx'] == filter_idx]
                if not row.empty:
                    rank = row.index[0] + 1
                    mean_diff = row.iloc[0]['mean_diff']
                    fold_change = row.iloc[0]['fold_change']
                    rank_info = f" rank{rank}_diff{mean_diff:.2f}_fc{fold_change:.2f}x"
            
            f.write(f"MOTIF Filter_{filter_idx}{rank_info} {max_seq}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {kernel_size} nsites= 20 E= 0\n")
            
            # Write PWM (A C G T)
            for pos in range(kernel_size):
                f.write(f"{pwm[pos, 0]:.6f} {pwm[pos, 1]:.6f} "
                       f"{pwm[pos, 2]:.6f} {pwm[pos, 3]:.6f}\n")
            f.write("\n")
    
    print(f"\nSaved MEME file: {output_file}")


def main():
    print("="*70)
    print("TOP FILTER ANALYSIS - MEME EXPORT")
    print("="*70)
    
    # Load filter comparison results
    results_file = '/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/suppressing_CTCFs/results_repeated/filter_comparison_results.tsv'
    
    if not os.path.exists(results_file):
        print(f"\nERROR: {results_file} not found!")
        print("Please run calculate_filter_activations.py first")
        return
    
    print(f"\nLoading filter results from: {results_file}")
    filter_results = pd.read_csv(results_file, sep='\t')
    
    # Get top 11 filters
    top_n = 11
    top_filters_df = filter_results.head(top_n)
    filter_indices = top_filters_df['filter_idx'].values.astype(int).tolist()
    
    print(f"\nTop {top_n} differential filters:")
    print(top_filters_df[['filter_idx', 'mean_diff', 'fold_change', 't_pvalue']].to_string(index=False))
    
    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    
    MODEL_PATH = (
        "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SeqNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    # Extract PWMs
    print("\n" + "="*70)
    print("EXTRACTING FILTER PWMs")
    print("="*70)
    
    pwms = extract_filter_pwms(model, filter_indices)
    
    # Analyze each filter
    print("\n" + "="*70)
    print("FILTER CONSENSUS SEQUENCES")
    print("="*70)
    
    for filter_idx in filter_indices:
        analyze_filter_pwm(filter_idx, pwms[filter_idx])
    
    # Create MEME file
    print("\n" + "="*70)
    print("CREATING MEME FILE")
    print("="*70)
    
    create_meme_file(pwms, 'top_11_filters.meme', filter_results)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nAnalyzed {len(filter_indices)} filters")
    print(f"Created: top_11_filters.meme")
    print("\nConsensus sequences (max probability base):")
    for filter_idx in filter_indices:
        max_seq, _ = pwm_to_consensus(pwms[filter_idx])
        rank = filter_results[filter_results['filter_idx'] == filter_idx].index[0] + 1
        print(f"  Filter {filter_idx:3d} (rank {rank:2d}): {max_seq}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Upload top_11_filters.meme to TOMTOM web server:")
    print("   https://meme-suite.org/meme/tools/tomtom")
    print("\n2. Select databases:")
    print("   - JASPAR CORE vertebrates")
    print("   - HOCOMOCO Mouse")
    print("\n3. Look for patterns in filters 4-11 that differ from 1-3")
    print("   (Filters 1-3 were homopolymer-related)")
    print()


if __name__ == "__main__":
    main()