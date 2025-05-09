import pandas as pd
import numpy as np
import random
import ast
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


def permute_disrupted_bins(seq, row, bin_size=2048, cropping=64):
    # Make seq mutable
    seq = list(seq)
    
    for bin_idx in row["disrupted_bin"]:
        start = (bin_idx + cropping) * bin_size
        end = start + bin_size
        if end > len(seq):  # Avoid index error
            continue
        region = seq[start:end]
        np.random.shuffle(region)
        seq[start:end] = region  # Replace with shuffled region

    return ''.join(seq)


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
        chrom, start, end = row["chrom"], row["window_start"], row["window_end"]
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
    # --- Load and process data ---
    sensitive_boundaries_path = "/scratch1/smaruj/sensitive_bins_boundaries.tsv"
    sensitive_boundaries_df = pd.read_csv(sensitive_boundaries_path, sep="\t")

    sensitive_boundaries_df["disrupted_bin"] = sensitive_boundaries_df["disrupted_bin"].apply(ast.literal_eval)

    # testing
    # sensitive_boundaries_df = sensitive_boundaries_df[:32]
    
    fasta_file = "/project/fudenber_735/genomes/hg38/hg38.fa"
    genome = Fasta(fasta_file)

    # --- Prepare datasets ---
    # Original dataset
    orig_dataset = GenomicSequenceDataset(sensitive_boundaries_df, genome)

    # Modified (permuted) dataset
    perm_dataset = GenomicSequenceDataset(sensitive_boundaries_df, genome, transform_fn=permute_disrupted_bins)

    orig_loader = DataLoader(orig_dataset, batch_size=16, shuffle=False)
    perm_loader = DataLoader(perm_dataset, batch_size=16, shuffle=False)

    # --- Load model ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SeqNN()
    model.load_state_dict(torch.load("/home1/smaruj/pytorch_akita/model_0_v2_finetuned_correctly.pt", map_location=device))
    model.eval()

    # --- Inference and metrics ---
    results = []

    with torch.no_grad():
        for orig_batch, perm_batch in zip(orig_loader, perm_loader):
            orig_preds = model(orig_batch.to(device)).cpu()
            perm_preds = model(perm_batch.to(device)).cpu()
            
            # there is no 1/2 multiplication,
            # since vectors not maps are used to calculate SCD
            scd = torch.sqrt(((perm_preds - orig_preds) ** 2).sum(dim=(1, 2)))
            
            maps = from_upper_triu_batch(orig_preds - perm_preds)
            boundary_strength = np.nanmean(maps[:, 0:256, 256:512], axis=(1, 2))  # shape: [B]
            
            # Combine into results
            for i in range(len(scd)):
                results.append({
                    "SCD": scd[i].item(),
                    "RUQ_mean": boundary_strength[i]
                })
            
    results_df = pd.DataFrame(results)
    combined_df = pd.concat([sensitive_boundaries_df.reset_index(drop=True), results_df], axis=1)

    # --- Save results (optional) ---
    combined_df.to_csv("/scratch1/smaruj/sensitive_boundary_results.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()