import pandas as pd
import numpy as np
import random
# import ast
from pyfaidx import Fasta
from torch.utils.data import Dataset, DataLoader
import torch

import sys
import os
sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))

# from model import SeqNN
from model_v2_compatible import SeqNN

# functions

def one_hot_encode_sequence(sequence_obj):
    # Convert pyfaidx.Sequence object to string
    sequence = str(sequence_obj).upper()
    
    # Define the mapping from bases to integers
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    valid_bases = list(base_to_int.keys())

    # Step 1: Convert sequence to integer encoding with random base for 'N'
    encoded_indices = []
    for base in sequence:
        if base in base_to_int:
            encoded_indices.append(base_to_int[base])
        else:
            random_base = random.choice(valid_bases)
            encoded_indices.append(base_to_int[random_base])

    # Step 2: One-hot encode the sequence
    encoded_sequence = np.array(encoded_indices)
    one_hot_encoded = np.zeros((4, len(encoded_sequence)), dtype=np.float32)
    one_hot_encoded[encoded_sequence, np.arange(len(encoded_sequence))] = 1

    return one_hot_encoded


class GenomicSequenceDataset(Dataset):
    def __init__(self, coord_df, genome_fasta, transform_fn=None):
        self.coords = coord_df  # DataFrame with chrom, start, end
        self.genome = genome_fasta
        self.transform_fn = transform_fn  # Optional function to modify sequence

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        TARGET_LEN = 1310720
        
        row = self.coords.iloc[idx]
        # chrom, start, end = row["chrom1"], row["window_start"], row["window_end"]
        chrom, start, end = row["chr"], row["window_start"], row["window_end"]
        seq = self.genome[chrom][start:end].seq.upper()
        
        # Fix sequence length if needed
        if len(seq) != TARGET_LEN:
            seq = seq[:TARGET_LEN].ljust(TARGET_LEN, 'N')  # pad with Ns if needed
        
        # Apply transformation, e.g. permute a window
        if self.transform_fn is not None:
            seq = self.transform_fn(seq, row)  # Pass row in case you want loc info
        
        one_hot = one_hot_encode_sequence(seq)  # shape: (4, L)
        return torch.from_numpy(one_hot.copy())
    
    
def set_diag(matrix, value, k):
    """Set diagonal `k` of a matrix to `value`."""
    rows, cols = matrix.shape
    for i in range(rows):
        if 0 <= i + k < cols:
            matrix[i, i + k] = value


def from_upper_triu_batch(batch_vectors, matrix_len=512, num_diags=2):
    """Convert a batch of upper-triangular vectors into symmetric matrices with np.nan on diagonals."""
    if isinstance(batch_vectors, torch.Tensor):
        batch_vectors = batch_vectors.detach().cpu().numpy()

    batch_size = batch_vectors.shape[0]
    matrices = np.zeros((batch_size, matrix_len, matrix_len), dtype=np.float32)

    triu_indices = np.triu_indices(matrix_len, num_diags)

    for i in range(batch_size):
        matrices[i][triu_indices] = batch_vectors[i]
        # Mirror to lower triangle
        matrices[i] = matrices[i] + matrices[i].T

        # Set diagonals to np.nan
        for k in range(-num_diags + 1, num_diags):
            set_diag(matrices[i], np.nan, k)

    return matrices  # shape: [B, 512, 512]  


def main():
    flame_df_path = "/scratch1/smaruj/stripenn_out/selected_stripes.tsv"
    flame_df = pd.read_csv(flame_df_path, sep="\t")
    
    # adding flames dimentions
    RES = 2048 # since that's Akita's pred resolution

    flame_df["y_bins"] = flame_df["length"] // RES
    flame_df["x_bins"] = flame_df["width"] // RES
    
    fasta_file = "/project/fudenber_735/genomes/mm10/mm10.fa"
    genome = Fasta(fasta_file)
    
    orig_dataset = GenomicSequenceDataset(flame_df, genome)
    orig_loader = DataLoader(orig_dataset, batch_size=16, shuffle=False)
    
    # --- Load model ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SeqNN()
    model.load_state_dict(torch.load("/home1/smaruj/pytorch_akita/model_0_v2_finetuned_correctly.pt", map_location=device))
    model.eval()
    
    results_median = []
    results_q3 = []  # 75th percentile
    results_mean = []  # mean
    
    map_size = 512
    map_midbin = 256 # midpoint of a 512x512 map
    start_idx = 0

    with torch.no_grad():
        for orig_batch in orig_loader:
            orig_preds = model(orig_batch.to(device)).cpu()
            og_maps = from_upper_triu_batch(orig_preds)
            
            this_batch_size = og_maps.shape[0]
            
            for i in range(this_batch_size):
                abs_i = start_idx + i
            
                x_end = min(max(map_midbin + flame_df["x_bins"].iat[abs_i], 0), map_size-1)

                if flame_df["triangular_half"].iat[abs_i] == "upper":
                    y_start = min(max(map_midbin - flame_df["y_bins"].iat[abs_i], 0), map_size-1)
                    if x_end > map_midbin and map_midbin > y_start:
                        region = og_maps[i, y_start:map_midbin, map_midbin:x_end]
                    else:
                        region = np.array([])
                else:
                    y_start = min(max(map_midbin + flame_df["y_bins"].iat[abs_i], 0), map_size-1)
                    if x_end > map_midbin and y_start < map_size:
                        region = og_maps[i, map_midbin:x_end, map_midbin:y_start]
                    else:
                        region = np.array([])
                
                if region.size > 0:
                    results_median.append(np.nanmedian(region))
                    results_q3.append(np.nanpercentile(region, 75))
                    results_mean.append(np.nanmean(region))
                else:
                    results_median.append(np.nan)
                    results_q3.append(np.nan)
                    results_mean.append(np.nan)
                
            start_idx += this_batch_size
            
    flame_df["flame_strength_median"] = results_median
    flame_df["flame_strength_q3"] = results_q3
    flame_df["flame_strength_mean"] = results_mean
    
    flame_df.to_csv("/scratch1/smaruj/stripenn_out/selected_stripes_strength_results_median_q3_mean.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()