#!/usr/bin/env python3
"""
Calculate GC content for each 2048bp slice within training windows.
"""

import pandas as pd
import numpy as np
from pyfaidx import Fasta
import argparse
from pathlib import Path

def calculate_gc_content(sequence):
    """
    Calculate GC content for a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        GC content as a fraction (0-1)
    """
    sequence = sequence.upper()
    g_count = sequence.count('G')
    c_count = sequence.count('C')
    
    # Count only ATGC bases (exclude N's and other ambiguous bases)
    valid_bases = sum([sequence.count(base) for base in ['A', 'T', 'G', 'C']])
    
    if valid_bases == 0:
        return np.nan
    
    return (g_count + c_count) / valid_bases


def process_bed_file(bed_file, genome_fasta, slice_size=2048, output_file=None):
    """
    Process bed file and calculate GC content for each slice.
    
    Args:
        bed_file: Path to bed file with training windows
        genome_fasta: Path to genome fasta file (e.g., mm10.fa)
        slice_size: Size of slices in bp (default: 2048)
        output_file: Path to output file (optional)
        
    Returns:
        DataFrame with GC content for each slice
    """
    # Read bed file
    print(f"Reading bed file: {bed_file}")
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, 
                         names=['chrom', 'start', 'end', 'fold'])
    
    print(f"Found {len(bed_df)} training windows")
    
    # Load genome
    print(f"Loading genome: {genome_fasta}")
    genome = Fasta(genome_fasta)
    
    # Calculate GC content for each slice
    results = []
    
    for idx, row in bed_df.iterrows():
        chrom = row['chrom']
        start = row['start']
        end = row['end']
        fold = row['fold']
        window_size = end - start
        
        # Calculate number of slices
        n_slices = window_size // slice_size
        
        if idx % 100 == 0:
            print(f"Processing window {idx + 1}/{len(bed_df)}")
        
        # Extract sequence for the entire window
        try:
            window_seq = genome[chrom][start:end].seq
        except Exception as e:
            print(f"Warning: Could not extract sequence for {chrom}:{start}-{end}: {e}")
            continue
        
        # Process each slice
        for slice_idx in range(n_slices):
            slice_start = slice_idx * slice_size
            slice_end = slice_start + slice_size
            slice_seq = window_seq[slice_start:slice_end]
            
            # Calculate GC content
            gc_content = calculate_gc_content(slice_seq)
            
            # Store results
            results.append({
                'window_idx': idx,
                'chrom': chrom,
                'window_start': start,
                'window_end': end,
                'fold': fold,
                'slice_idx': slice_idx,
                'slice_start_in_window': slice_start,
                'slice_end_in_window': slice_end,
                'genomic_start': start + slice_start,
                'genomic_end': start + slice_end,
                'gc_content': gc_content
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\nProcessed {len(results_df)} slices from {len(bed_df)} windows")
    print(f"Mean GC content: {results_df['gc_content'].mean():.4f}")
    print(f"Std GC content: {results_df['gc_content'].std():.4f}")
    
    # Save results
    if output_file:
        results_df.to_csv(output_file, sep='\t', index=False)
        print(f"Results saved to: {output_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Calculate GC content for 2048bp slices in training windows'
    )
    parser.add_argument('--bed', type=str, default="/project2/fudenber_735/tensorflow_models/akita/v2/data/mm10/sequences.bed",
                        help='Path to bed file with training windows')
    parser.add_argument('--genome', type=str, default="/project2/fudenber_735/genomes/mm10/mm10.fa",
                        help='Path to genome fasta file (e.g., mm10.fa)')
    parser.add_argument('--slice-size', type=int, default=2048,
                        help='Size of slices in bp (default: 2048)')
    parser.add_argument('--output', required=True,
                        help='Path to output file')
    
    args = parser.parse_args()
    
    # Process
    results_df = process_bed_file(
        bed_file=args.bed,
        genome_fasta=args.genome,
        slice_size=args.slice_size,
        output_file=args.output
    )
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total slices: {len(results_df)}")
    print(f"Mean GC content: {results_df['gc_content'].mean():.4f}")
    print(f"Median GC content: {results_df['gc_content'].median():.4f}")
    print(f"Std GC content: {results_df['gc_content'].std():.4f}")
    print(f"Min GC content: {results_df['gc_content'].min():.4f}")
    print(f"Max GC content: {results_df['gc_content'].max():.4f}")
    
    # Print distribution
    print("\nGC content distribution:")
    print(results_df['gc_content'].describe())


if __name__ == '__main__':
    main()