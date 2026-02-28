import pandas as pd
import numpy as np
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


class GenomicSequenceDataset(Dataset):
    def __init__(self, coord_df, genome_fasta):
        self.coords = coord_df  # DataFrame with chrom, start, end
        self.genome = genome_fasta

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
    boundaries_df = pd.read_csv("/scratch1/smaruj/natural_boundaries_URQmean/all_natural_boundaries.tsv", sep="\t")
    
    fasta_file = "/project/fudenber_735/genomes/mm10/mm10.fa"
    genome = Fasta(fasta_file)    
    
    orig_dataset = GenomicSequenceDataset(boundaries_df, genome)
    orig_loader = DataLoader(orig_dataset, batch_size=16, shuffle=False)
    
    # --- Load model ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SeqNN()
    model.load_state_dict(torch.load("/home1/smaruj/pytorch_akita/model_0_v2_finetuned_correctly.pt", map_location=device))
    model.eval()
    
    URQ_mean_all = []

    with torch.no_grad():
        for orig_batch in orig_loader:
            orig_preds = model(orig_batch.to(device)).cpu()
            
            maps = from_upper_triu_batch(orig_preds)
            URQ_mean = np.nanmean(maps[:, 0:250, 260:512], axis=(1, 2))  # shape: [B]
            
            URQ_mean_all.extend(URQ_mean)
            
    boundaries_df["URQ_mean"] = URQ_mean_all
    boundaries_df.to_csv("/scratch1/smaruj/natural_boundaries_URQmean/all_boundaries_URQ_mean.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()