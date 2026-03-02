import numpy as np
import pandas as pd
import cooler
from cooltools.lib.numutils import observed_over_expected, adaptive_coarsegrain
from cooltools.lib.numutils import set_diag, interp_nan
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

import sys
import os
sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))

# from model import SeqNN
from model_v2_compatible import SeqNN

import torch
from pyfaidx import Fasta
from scipy.stats import pearsonr
import random


def one_hot_encode_sequence(sequence_obj):
    sequence = str(sequence_obj).upper()
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    encoded_sequence = np.array([
        base_to_int.get(base, base_to_int[random.choice("ACGT")]) for base in sequence
    ])

    one_hot_encoded = np.zeros((4, len(encoded_sequence)), dtype=np.float32)
    one_hot_encoded[encoded_sequence, np.arange(len(encoded_sequence))] = 1

    return np.expand_dims(one_hot_encoded, axis=0)


def process_hic_matrix(genome_hic_cool, mseq_str, diagonal_offset=2, padding=64, kernel_stddev=1):
    seq_hic_raw = genome_hic_cool.matrix(balance=True).fetch(mseq_str)
    
    # Check for NaN filtering percentage
    seq_hic_nan = np.isnan(seq_hic_raw)
    num_filtered_bins = np.sum(np.sum(seq_hic_nan, axis=0) == len(seq_hic_nan))
    print("num_filtered_bins:", num_filtered_bins)
    
    if num_filtered_bins > (0.5 * len(seq_hic_nan)):
        print(f"More than 50% bins filtered in {mseq_str}. Check Hi-C data quality.")
        
    # clip first diagonals and high values
    clipval = np.nanmedian(np.diag(seq_hic_raw, diagonal_offset))
    for i in range(-diagonal_offset+1, diagonal_offset):
        set_diag(seq_hic_raw, clipval, i)
    seq_hic_raw = np.clip(seq_hic_raw, 0, clipval)
    seq_hic_raw[seq_hic_nan] = np.nan
    
    # adaptively coarsegrain based on raw counts
    seq_hic_smoothed = adaptive_coarsegrain(
                            seq_hic_raw,
                            genome_hic_cool.matrix(balance=False).fetch(mseq_str),
                            cutoff=2, max_levels=8)
    seq_hic_nan = np.isnan(seq_hic_smoothed)
    
    # local obs/exp
    seq_hic_obsexp = observed_over_expected(seq_hic_smoothed, ~seq_hic_nan)[0]
    
    log_hic_obsexp = np.log(seq_hic_obsexp)
    
    # Apply padding
    if padding > 0:
        log_hic_obsexp = log_hic_obsexp[padding:-padding, padding:-padding]
    
    log_hic_obsexp = interp_nan(log_hic_obsexp)
    for i in range(-diagonal_offset+1, diagonal_offset): set_diag(log_hic_obsexp, 0,i)
    
    kernel = Gaussian2DKernel(x_stddev=kernel_stddev)
    seq_hic = convolve(log_hic_obsexp, kernel)
    
    return seq_hic    


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


def upper_triangular_to_vector_skip_diagonals(matrix, dim=512, diag=2):
    
    # Extract the upper triangular part excluding the first two diagonals
    upper_triangular_vector = matrix[np.triu_indices(dim, k=diag)]
    
    return upper_triangular_vector


def main():
    flame_df_path = "/scratch1/smaruj/stripepy_stripes/selected_stripes.tsv"
    flame_df = pd.read_csv(flame_df_path, sep="\t")
    
    FASTA_FILE = "/project/fudenber_735/genomes/mm10/mm10.fa"
    COOL_FILE = "/project/fudenber_735/GEO/Hsieh2019/4DN/mESC_mm10_4DNFILZ1CPT8.mapq_30.2048.cool"

    # --- Load Data ---
    genome = Fasta(FASTA_FILE)
    genome_hic_cool = cooler.Cooler(COOL_FILE)
    
    map_midbin = 256
    bin_size = 2048
    
    # --- Load model ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SeqNN()
    model.load_state_dict(torch.load("/home1/smaruj/pytorch_akita/model_0_v2_finetuned_correctly.pt", map_location=device))
    model.eval() 
    
    pearson_r = []
    target_flame_mean = []
    pred_flame_mean = []
    
    for i, row in enumerate(flame_df.itertuples(index=False)):
        print("index:", i)
        chrom, start, end = row.chrom1, row.window_start, row.window_end
        
        # This will round start down and end up to 2048 boundaries
        start = (start // bin_size) * bin_size
        end = ((end + bin_size - 1) // bin_size) * bin_size
        
        mseq_str = f"{chrom}:{start}-{end}"
        
        x_len, y_len = row.x_length // bin_size, row.y_length // bin_size
        
        x_end = map_midbin + x_len
        y_start = map_midbin - y_len
        
        if x_end > 512:
            x_end = 512
        if y_start < 0:
            y_start = 0
        
        try:
            # Get target
            sequence = genome[chrom][start:end]
            ohe_sequence = one_hot_encode_sequence(sequence)
            tensor_ohe_sequence = torch.from_numpy(ohe_sequence)
            matrix = process_hic_matrix(genome_hic_cool, mseq_str, diagonal_offset=2, padding=64, kernel_stddev=1)
            target_vector = upper_triangular_to_vector_skip_diagonals(matrix)
        
            flame_mean = np.nanmean(matrix[map_midbin:x_end, y_start:map_midbin])
            target_flame_mean.append(flame_mean)
            
            # Get prediction
            akita_pred = model(tensor_ohe_sequence.to(device)).cpu()
            torch_vec_np = akita_pred.squeeze().detach().cpu().numpy()
            akita_map = from_upper_triu(akita_pred, 512, 2)
            
            pred_flame_val = np.nanmean(akita_map[map_midbin:x_end, y_start:map_midbin])
            pred_flame_mean.append(pred_flame_val) 
        
            # Pearson R
            r, _ = pearsonr(target_vector, torch_vec_np)
            pearson_r.append(r)

        except ValueError as e:
            print(f"Skipping index {i} due to error: {e}")
            pearson_r.append(np.nan)
            target_flame_mean.append(np.nan)
            pred_flame_mean.append(np.nan)    
        
    flame_df["PearsonR"] = pearson_r
    flame_df["target_fl_mean"] = target_flame_mean
    flame_df["pred_fl_mean"] = pred_flame_mean
    
    flame_df.to_csv(f"/scratch1/smaruj/stripepy_stripes/selected_stripes_pearsonr.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
    