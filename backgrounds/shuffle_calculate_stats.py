import pandas as pd
import numpy as np
import seqpro as sp
import random

from torch.utils.data import Dataset, DataLoader
import torch
from pyfaidx import Fasta

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


def local_std_metric_batch(preds, window=32, stride=32):
    """
    Compute mean local standard deviation per map in a batch of shape [B, H, W],
    ignoring NaNs.
    Returns: array of shape [B], one std score per map.
    """
    B, H, W = preds.shape
    results = []

    for b in range(B):
        vals = []
        for i in range(0, H - window + 1, stride):
            for j in range(0, W - window + 1, stride):
                patch = preds[b, i:i+window, j:j+window]
                std = np.nanstd(patch)
                vals.append(std)
        results.append(np.nanmean(vals))

    return np.array(results)


def total_variation_batch(preds):
    """
    Compute total variation per map in a batch of shape [B, H, W],
    ignoring NaNs.
    Returns: array of shape [B]
    """
    B, H, W = preds.shape
    results = []

    for b in range(B):
        pred = preds[b]
        dx = pred[:, 1:] - pred[:, :-1]
        dy = pred[1:, :] - pred[:-1, :]

        # Flatten and remove NaNs
        dx = dx[~np.isnan(dx)]
        dy = dy[~np.isnan(dy)]

        total_var = np.sum(np.abs(dx)) + np.sum(np.abs(dy))
        results.append(total_var)

    return np.array(results)


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

    batch_size = len(batch_vectors)
    matrices = np.zeros((batch_size, matrix_len, matrix_len), dtype=np.float32)

    triu_indices = np.triu_indices(matrix_len, num_diags)

    for i in range(batch_size):
        matrices[i][triu_indices] = batch_vectors[i][0,:]
        # Mirror to lower triangle
        matrices[i] = matrices[i] + matrices[i].T

        # Set diagonals to np.nan
        for k in range(-num_diags + 1, num_diags):
            set_diag(matrices[i], np.nan, k)

    return matrices  # shape: [B, 512, 512]


def random_kmer_shuffle(seq, k, seed=None):
    """
    Split the sequence into non-overlapping k-mers and shuffle them randomly.
    """
    if seed is not None:
        random.seed(seed)

    # Truncate to full k-mer blocks
    seq = seq[:len(seq) - len(seq) % k]
    kmers = [seq[i:i+k] for i in range(0, len(seq), k)]
    random.shuffle(kmers)
    return ''.join(kmers)


class GenomicSequenceDataset(Dataset):
    def __init__(self, coord_df, genome_fasta):
        self.coords = coord_df
        self.genome = genome_fasta

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        TARGET_LEN = 1310720
        row = self.coords.iloc[idx]
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        k = int(row["shuffle_parameter"])

        seq = self.genome[chrom][start:end].seq.upper()
        if len(seq) != TARGET_LEN:
            seq = seq[:TARGET_LEN].ljust(TARGET_LEN, 'N')

        if k > 0:
            shuffled_bytes = sp.k_shuffle(seq, k=k)
            seq = b"".join(shuffled_bytes).decode()
            
            # to compare with random shuffling
            # seq = random_kmer_shuffle(seq, k)

        one_hot = one_hot_encode_sequence(seq)
        return torch.from_numpy(one_hot.copy())
    

def main():
    df = pd.read_csv("/home1/smaruj/akitaV2-analyses/experiments/background_generation/sequence_shuffling/input_data/shuffled_600seqs.tsv", sep="\t")
    
    # testing
    df = df[:32]
    
    fasta_file = "/project/fudenber_735/genomes/hg38/hg38.fa"
    genome = Fasta(fasta_file)
    
    orig_dataset = GenomicSequenceDataset(df, genome)
    orig_loader = DataLoader(orig_dataset, batch_size=16, shuffle=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SeqNN()
    model.load_state_dict(torch.load("/home1/smaruj/pytorch_akita/model_0_v2_finetuned_correctly.pt", map_location=device))
    model.eval()

    scd_values = []
    local_sd_values = []
    total_var_values = []

    model.eval()
    with torch.no_grad():
        for batch in orig_loader:
            batch = batch.to(device)  # shape [B, 4, L]
            
            # Predict
            preds = model(batch).cpu()  # shape [B, T]
            
            # Compute SCD
            scd_batch = torch.sqrt((preds ** 2).sum(dim=(1, 2)))  # [B]
            scd_values.extend(scd_batch.numpy())
            
            # Decode full 2D maps
            maps = from_upper_triu_batch(preds)  # shape [B, H, W]
            # maps_np = maps.numpy()

            # Compute per-sample metrics
            local_std_batch = local_std_metric_batch(maps)      # shape [B]
            total_var_batch = total_variation_batch(maps)       # shape [B]

            local_sd_values.extend(local_std_batch)
            total_var_values.extend(total_var_batch)
    
    df["scd"] = scd_values
    df["local_std"] = local_sd_values
    df["total_var"] = total_var_values
    
    # saving
    df.to_csv("/scratch1/smaruj/background_generation/shuffled_results_test.tsv", sep="\t", index=False)
    
    
if __name__ == "__main__":
    main()