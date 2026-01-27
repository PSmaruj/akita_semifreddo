"""
CTCF Experiment: Variable Middle CTCF Configurations

Tests how different numbers and orientations of middle CTCFs affect
contacts between fixed outer CTCFs (CTCF1 and CTCF4).

Design:
- Outer CTCFs (CTCF1 >< CTCF4): Fixed at 400kb apart
- Middle CTCFs: Variable number and orientations
- Middle CTCF separation: 250bp between adjacent CTCFs
- Controls: No CTCFs, and Outer CTCFs only
"""

import pandas as pd
import numpy as np
import torch
from pyfaidx import Fasta

from Bio import SeqIO
from Bio.Seq import Seq
import argparse

import sys
import os
sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))

# from model import SeqNN
from akita_model.model import SeqNN

def parse_args():
    parser = argparse.ArgumentParser(description="Predict insulation and dot score for a CTCFs into background insertions.")
    parser.add_argument("--outer_orient", type=str, required=True, help="Orientation of the outer CTCFs.")
    return parser.parse_args()  # <-- FIXED: Added return statement


def reverse_complement(seq):
    """Get reverse complement of a DNA sequence"""
    return str(Seq(seq).reverse_complement())


def one_hot_encode(sequence, return_type='numpy', channels_first=False):
    """
    One-hot encode a DNA sequence for Akita input
    
    Parameters:
    -----------
    sequence : str
        DNA sequence string
    return_type : str
        'numpy' or 'torch' - type of array to return
    channels_first : bool
        If True, return shape (4, length) for PyTorch conv layers
        If False, return shape (length, 4)
    
    Returns:
    --------
    np.array or torch.Tensor : One-hot encoded sequence
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3,
               'a': 0, 'c': 1, 'g': 2, 't': 3,
               'N': -1, 'n': -1}
    
    seq_len = len(sequence)
    one_hot = np.zeros((seq_len, 4), dtype=np.float32)
    
    for i, base in enumerate(sequence):
        if base in mapping:
            idx = mapping[base]
            if idx >= 0:
                one_hot[i, idx] = 1.0
            else:
                one_hot[i, :] = 0.25
    
    if channels_first:
        one_hot = one_hot.T
    
    if return_type == 'torch':
        import torch
        return torch.from_numpy(one_hot)
    else:
        return one_hot


def get_ctcf_forward_seq(chrom, start, end, strand, genome_path):
    genome = Fasta(genome_path)
    flank = 15
    seq = genome[chrom][start-flank:end+flank].seq
    if strand == "-":
        # reverse complement
        complement = str.maketrans("ACGTacgt", "TGCAtgca")
        seq = seq[::-1].translate(complement)
    return seq.upper()


def insert_ctcfs_at_positions(background_seq, ctcf_seq, ctcf_orientations, ctcf_positions):
    """
    Insert CTCFs at specified positions with given orientations
    
    Parameters:
    -----------
    background_seq : str
        Background DNA sequence
    ctcf_seq : str
        CTCF motif sequence
    ctcf_orientations : list
        List of orientations for each CTCF ('forward' or 'reverse')
    ctcf_positions : list
        List of positions (in bp) where to insert each CTCF
    
    Returns:
    --------
    str : Modified sequence with CTCFs inserted
    """
    seq_list = list(background_seq)
    
    insertions = []
    for pos, orientation in zip(ctcf_positions, ctcf_orientations):
        if orientation == 'reverse':
            seq_to_insert = reverse_complement(ctcf_seq)
        else:
            seq_to_insert = ctcf_seq
        insertions.append((pos, seq_to_insert))
    
    # Sort by position (descending) to insert from right to left
    insertions.sort(key=lambda x: x[0], reverse=True)
    
    # Insert CTCFs by replacement
    for pos, seq_to_insert in insertions:
        for i, base in enumerate(seq_to_insert):
            if pos + i < len(seq_list):
                seq_list[pos + i] = base
    
    return ''.join(seq_list)

def calculate_ctcf_positions_new_design(background_length, outer_span_kb=400, 
                                        middle_config="", middle_separation_bp=250,
                                        outer_config="><"):
    """
    Calculate CTCF positions for new experimental design
    
    Parameters:
    -----------
    background_length : int
        Length of background sequence in bp
    outer_span_kb : int
        Span from CTCF1 to CTCF4 in kilobases (default: 400kb)
    middle_config : str
        Configuration of middle CTCFs, e.g., "", "<", "<<", "<<<", "<<<<", "<>", "><>", "><><"
        Empty string means no middle CTCFs
    middle_separation_bp : int
        Separation between adjacent middle CTCFs in bp (default: 250bp)
    outer_config : str
        Configuration of outer CTCFs (2 characters), e.g., "><", ">>", "<<", "<>"
        First char = CTCF1, Second char = CTCF4
        Default: "><" (divergent)
    
    Returns:
    --------
    list : Positions for all CTCFs
    list : Orientations for all CTCFs
    str : Description of configuration
    """
    if len(outer_config) != 2:
        raise ValueError("outer_config must be exactly 2 characters (e.g., '><', '>>', '<<', '<>')")
    
    middle_pos = background_length // 2
    outer_span_bp = outer_span_kb * 1000
    
    # Outer CTCFs at fixed positions
    ctcf1_pos = middle_pos - (outer_span_bp // 2)
    ctcf4_pos = middle_pos + (outer_span_bp // 2)
    
    # Parse outer orientations
    ctcf1_orientation = 'forward' if outer_config[0] == '>' else 'reverse'
    ctcf4_orientation = 'forward' if outer_config[1] == '>' else 'reverse'
    
    positions = [ctcf1_pos]
    orientations = [ctcf1_orientation]
    
    # Parse middle configuration
    if middle_config:
        num_middle = len(middle_config)
        
        # Calculate starting position for middle CTCFs (centered)
        total_middle_span = (num_middle - 1) * middle_separation_bp
        middle_start = middle_pos - (total_middle_span // 2)
        
        # Add middle CTCFs
        for i, symbol in enumerate(middle_config):
            pos = middle_start + (i * middle_separation_bp)
            positions.append(pos)
            orientations.append('forward' if symbol == '>' else 'reverse')
    
    # Add CTCF4
    positions.append(ctcf4_pos)
    orientations.append(ctcf4_orientation)
    
    # Create description
    num_ctcfs = len(positions)
    config_desc = f"Outer({outer_config})_Middle({middle_config if middle_config else 'none'})"
    
    return positions, orientations, config_desc

def create_experiment_configurations(ctcf_df, background_fasta, 
                                     outer_span_kb=400,
                                     outer_config="><",
                                     middle_configs=["", "<", "<<", "<<<", "<<<<", 
                                                    "<>", "><>", "><><"],
                                     middle_separation_bp=250,
                                     include_controls=True):
    """
    Create experiment configurations for all CTCF setups
    
    Parameters:
    -----------
    ctcf_df : pd.DataFrame
        DataFrame with CTCF information (must have 'ctcf_seq' column)
    background_fasta : str
        Path to background sequences FASTA file
    outer_span_kb : int
        Span from CTCF1 to CTCF4 in kb (default: 400)
    outer_config : str
        Configuration of outer CTCFs (2 characters), e.g., "><", ">>", "<<", "<>"
        First char = CTCF1, Second char = CTCF4
        Default: "><" (divergent - loop-forming)
        Examples:
          "><" - divergent (standard loop)
          ">>" - both forward (tandem)
          "<<" - both reverse (tandem)
          "<>" - convergent (potentially insulating)
    middle_configs : list
        List of middle CTCF configurations to test
        Examples: "", "<", "<<", "<<<", "<<<<", "<>", "><>", "><><"
    middle_separation_bp : int
        Separation between adjacent middle CTCFs in bp (default: 250)
    include_controls : bool
        If True, include control conditions (no CTCFs, outer only)
    
    Returns:
    --------
    pd.DataFrame : Experiment configuration dataframe
    dict : Background sequences {index: sequence}
    """
    # Read background sequences with simple integer indexing
    print("Loading background sequences...")
    background_seqs = {}
    for idx, record in enumerate(SeqIO.parse(background_fasta, "fasta")):
        background_seqs[idx] = str(record.seq)
        print(f"  Background {idx}: {len(record.seq):,} bp")
    
    num_backgrounds = len(background_seqs)
    print(f"Loaded {num_backgrounds} background sequences")
    
    experiments = []
    
    # Create all combinations of CTCF × Background × Configuration
    for ctcf_idx, ctcf_row in ctcf_df.iterrows():
        ctcf_id = ctcf_row.get('seq_id', f'ctcf_{ctcf_idx}')
        ctcf_seq = ctcf_row['ctcf_seq']
        
        for bg_idx in range(num_backgrounds):
            bg_seq = background_seqs[bg_idx]
            bg_length = len(bg_seq)
            
            # Control 1: No CTCFs (if requested)
            if include_controls:
                exp = {
                    'experiment_id': len(experiments),
                    'ctcf_id': ctcf_id,
                    'ctcf_index': ctcf_idx,
                    'background_idx': bg_idx,
                    'background_length': bg_length,
                    'ctcf_seq': ctcf_seq,
                    'configuration': 'CONTROL_NO_CTCF',
                    'outer_config': '',
                    'num_ctcfs': 0,
                    'num_middle_ctcfs': 0,
                    'middle_config': '',
                    'ctcf_positions': [],
                    'ctcf_orientations': [],
                    'outer_span_kb': outer_span_kb,
                    'middle_separation_bp': middle_separation_bp
                }
                experiments.append(exp)
            
            # Control 2: Outer CTCFs only (if requested)
            if include_controls:
                positions, orientations, config_desc = calculate_ctcf_positions_new_design(
                    bg_length, outer_span_kb, "", middle_separation_bp, outer_config
                )
                
                exp = {
                    'experiment_id': len(experiments),
                    'ctcf_id': ctcf_id,
                    'ctcf_index': ctcf_idx,
                    'background_idx': bg_idx,
                    'background_length': bg_length,
                    'ctcf_seq': ctcf_seq,
                    'configuration': f'CONTROL_OUTER_ONLY_{outer_config}',
                    'outer_config': outer_config,
                    'num_ctcfs': 2,
                    'num_middle_ctcfs': 0,
                    'middle_config': '',
                    'ctcf_positions': positions,
                    'ctcf_orientations': orientations,
                    'outer_span_kb': outer_span_kb,
                    'middle_separation_bp': middle_separation_bp
                }
                experiments.append(exp)
            
            # Main experiments: Different middle configurations
            for middle_config in middle_configs:
                if middle_config == "":  # Skip empty if already in controls
                    if include_controls:
                        continue
                
                positions, orientations, config_desc = calculate_ctcf_positions_new_design(
                    bg_length, outer_span_kb, middle_config, middle_separation_bp, outer_config
                )
                
                exp = {
                    'experiment_id': len(experiments),
                    'ctcf_id': ctcf_id,
                    'ctcf_index': ctcf_idx,
                    'background_idx': bg_idx,
                    'background_length': bg_length,
                    'ctcf_seq': ctcf_seq,
                    'configuration': config_desc,
                    'outer_config': outer_config,
                    'num_ctcfs': len(positions),
                    'num_middle_ctcfs': len(middle_config),
                    'middle_config': middle_config,
                    'ctcf_positions': positions,
                    'ctcf_orientations': orientations,
                    'outer_span_kb': outer_span_kb,
                    'middle_separation_bp': middle_separation_bp
                }
                
                experiments.append(exp)
    
    exp_df = pd.DataFrame(experiments)
    
    # Print summary
    print(f"\nExperiment summary:")
    print(f"  Total experiments: {len(exp_df)}")
    print(f"  CTCFs tested: {len(ctcf_df)}")
    print(f"  Backgrounds per CTCF: {num_backgrounds}")
    print(f"  Configurations per CTCF×Background: {len(middle_configs) + (2 if include_controls else 0)}")
    print(f"  Formula: {len(ctcf_df)} CTCFs × {num_backgrounds} backgrounds × {len(middle_configs) + (2 if include_controls else 0)} configs = {len(exp_df)} experiments")
    
    return exp_df, background_seqs

def run_predictions_new_design(exp_df, background_seqs, model, 
                               from_upper_triu_func):
    """
    Run Akita predictions for all experiment configurations
    
    Parameters:
    -----------
    exp_df : pd.DataFrame
        Experiment configuration dataframe
    background_seqs : dict
        Dictionary of background sequences {id: sequence}
    model : pytorch model
        Loaded Akita model
    from_upper_triu_func : function
        Function to convert model output to 512x512 matrix
    
    Returns:
    --------
    pd.DataFrame : Updated experiment dataframe with statistics
    """
    
    contact_stats = []
    
    for idx, row in exp_df.iterrows():
        print(f"\nProcessing experiment {idx+1}/{len(exp_df)}:")
        print(f"  Configuration: {row['configuration']}")
        print(f"  CTCF: {row['ctcf_id']}, Background: {row['background_idx']}")
        print(f"  Total CTCFs: {row['num_ctcfs']}, Middle: {row['num_middle_ctcfs']} ({row['middle_config']})")
        
        # Get background sequence
        background_seq = background_seqs[row['background_idx']]
        
        # Modify sequence with CTCFs
        if row['num_ctcfs'] > 0:
            modified_seq = insert_ctcfs_at_positions(
                background_seq,
                row['ctcf_seq'],
                row['ctcf_orientations'],
                row['ctcf_positions']
            )
        else:
            # Control: no CTCFs
            modified_seq = background_seq
        
        # One-hot encode
        seq_one_hot = one_hot_encode(modified_seq, return_type='torch', channels_first=True)
        seq_input = seq_one_hot.unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            prediction = model(seq_input)
        prediction_np = prediction[0].cpu().numpy()
        
        # Convert to 512x512 matrix
        matrix = from_upper_triu_func(prediction_np, matrix_len=512, num_diags=2)
        
        # Calculate statistics (no need to store matrix)
        stats = calculate_contact_statistics_new_design(
            matrix,
            row['ctcf_positions'] if row['num_ctcfs'] > 0 else [],  # <-- FIXED: Added ctcf_positions
            row['num_ctcfs'],
            row['middle_config']
        )
        
        contact_stats.append(stats)
        
        # FIXED: Added print statements for stats
        print(f"  Avg insulation: {stats.get('avg_insulation', np.nan):.6f}")
        print(f"  Avg outer dot score: {stats.get('avg_outer_dot_score', np.nan):.6f}")
    
    # Add statistics to dataframe
    stats_df = pd.DataFrame(contact_stats)
    exp_df_updated = pd.concat([exp_df.reset_index(drop=True), stats_df], axis=1)
    
    return exp_df_updated

def calculate_contact_statistics_new_design(contact_map, ctcf_positions,
                                            num_ctcfs, middle_config,
                                            matrix_size=512):
    """
    Calculate contact statistics for new experimental design
    
    Parameters:
    -----------
    contact_map : np.array
        Predicted contact map (512x512)
    ctcf_positions : list
        Positions of CTCFs in bp (can be empty for control)
    num_ctcfs : int
        Total number of CTCFs
    middle_config : str
        Middle CTCF configuration string
    matrix_size : int
        Size of contact matrix (default: 512)
    
    Returns:
    --------
    dict : Dictionary with contact statistics
    """
    stats = {
        'num_ctcfs': num_ctcfs,
        'num_middle_ctcfs': len(middle_config),
        'middle_config': middle_config
    }
    
    center = matrix_size // 2  # 256
    
    # Average insulation (upper-right quadrant)
    # matrix[:256, 256:] - contacts between left half and right half
    insulation_region = contact_map[:center, center:]
    stats['avg_insulation'] = float(np.nanmean(insulation_region))
    stats['std_insulation'] = float(np.nanstd(insulation_region))
    
    # Average outer dot score
    # Region around outer CTCF contact: [157:159, 353:356]
    # (This is approximately where outer CTCFs would contact in the matrix)
    dot_row_start = 157
    dot_row_end = 159
    dot_col_start = 353
    dot_col_end = 356
    
    # Ensure indices are within bounds
    if (dot_row_end <= matrix_size and dot_col_end <= matrix_size):
        dot_region = contact_map[dot_row_start:dot_row_end, dot_col_start:dot_col_end]
        stats['avg_outer_dot_score'] = float(np.nanmean(dot_region))
    else:
        stats['avg_outer_dot_score'] = np.nan
    
    return stats


# Helper function to set diagonal elements to a specific value
def set_diag(matrix, value, k):
    # Explicitly set the diagonal to 'value' (in this case, np.nan) for each k
    rows, cols = matrix.shape
    for i in range(rows):
        if 0 <= i + k < cols:
            matrix[i, i + k] = value

def from_upper_triu(vector_repr, matrix_len, num_diags):
    # Ensure vector_repr is a NumPy array (if it's a PyTorch tensor, convert it)
    if isinstance(vector_repr, torch.Tensor):
        vector_repr = vector_repr.detach().flatten().cpu().numpy()  # Flatten and convert to NumPy array

    # Initialize a zero matrix of shape (matrix_len, matrix_len)
    z = np.zeros((matrix_len, matrix_len))

    # Get the indices for the upper triangular matrix
    triu_tup = np.triu_indices(matrix_len, num_diags)

    # Assign the values from the vector_repr to the upper triangular part of the matrix
    z[triu_tup] = vector_repr

    # Set the diagonals specified by num_diags to np.nan
    for i in range(-num_diags + 1, num_diags):
        set_diag(z, np.nan, i)

    # Ensure the matrix is symmetric
    return z + z.T


def main():
    args = parse_args()
    OUTER_ORIENT = args.outer_orient
    
    # 100 strongest CTCFs
    ctcf_df = pd.read_csv("/scratch1/smaruj/full_akita_vs_semifreddo/top100_ctcfs.csv")
    
    genome_path = "/project2/fudenber_735/genomes/mm10/mm10.fa"
    
    # Apply to all 100 CTCFs
    ctcf_df["ctcf_seq"] = ctcf_df.apply(
        lambda row: get_ctcf_forward_seq(row["chrom"], row["start"], row["end"], row["strand"], genome_path), axis=1
    )
    
    # Configuration
    fasta_file = "/scratch1/smaruj/background_generation/background_sequences_scd30_totvar1300.fasta"
    
    # Create experiments
    print("="*80)
    print("CTCF Experiment: Variable Middle CTCF Configurations")
    print("="*80)
    
    exp_df, background_seqs = create_experiment_configurations(
        ctcf_df,
        fasta_file,
        outer_span_kb=400,
        outer_config=OUTER_ORIENT,  # Can change to ">>", "<<", "<>", etc.
        middle_configs=["<", "<<", "<<<<", "<<<<<<", "<>", "<><>", "<><><>"],
        middle_separation_bp=250,
        include_controls=True
    )
    
    print(f"\nCreated {len(exp_df)} experiment configurations")
    print(f"Outer CTCF configuration: {exp_df.iloc[1]['outer_config'] if len(exp_df) > 1 else 'N/A'}")
    
    print("\n" + "="*80)
    print("Loading model...")
    print("="*80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SeqNN()
    model.load_state_dict(torch.load("/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth", map_location=device))
    model.eval()
    model = model.to(device)
    
    print("✓ Model loaded")
    
    print("\n" + "="*80)
    print("Running predictions...")
    print("="*80)

    results = run_predictions_new_design(exp_df, background_seqs, model, from_upper_triu)
    
    # Save as TSV
    output_file = f"/scratch1/smaruj/ctcf_hopping/ctcf_{OUTER_ORIENT}_hopping_with_flanks.tsv"
    results.to_csv(output_file, sep="\t", index=False)
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)
    print(f"Results saved to: {output_file}")
    print(f"Total experiments: {len(results)}")
    print("="*80)
    
if __name__ == "__main__":
    main()