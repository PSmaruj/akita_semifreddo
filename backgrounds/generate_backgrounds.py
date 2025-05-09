import pandas as pd
import torch
from torch.utils.data import DataLoader
import seqpro as sp
import random
import numpy as np
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


def total_variation(pred):
    """
    Total variation across the map. Lower → flatter.
    """
    dx = pred[:, 1:] - pred[:, :-1]
    dy = pred[1:, :] - pred[:-1, :]
    return np.sum(np.abs(dx)) + np.sum(np.abs(dy))


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
    # === CONFIG ===
    TSV_PATH = "/home1/smaruj/akitaV2-analyses/experiments/background_generation/background_generation/input_data/50seqs_GCuniform_maxSCD35.tsv"
    OUTPUT_FASTA = "/scratch1/smaruj/background_generation/background_sequences_test.fasta"
    K = 8
    MAX_TRIES_PER_SEQ = 20
    TARGET_SCD_THRESHOLD = 30
    TARGET_VAR_THRESHOLD = 1300
    TARGET_COUNT = 10

    # === LOAD ===
    df = pd.read_csv(TSV_PATH, sep='\t')
    seen_indices = set()
    saved_seqs = []
    scd_scores = []
    totvar_scores = []

    fasta_file = "/project/fudenber_735/genomes/hg38/hg38.fa"
    genome = Fasta(fasta_file)

    # === Model (assumes already loaded and on CUDA) ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SeqNN()
    model.load_state_dict(torch.load("/home1/smaruj/pytorch_akita/model_0_v2_finetuned_correctly.pt", map_location=device))
    model.eval()

    # === Shuffling loop ===
    with torch.no_grad():
        while len(saved_seqs) < TARGET_COUNT and len(seen_indices) < len(df):
            idx = random.choice(df.index.difference(seen_indices))
            seen_indices.add(idx)

            row = df.loc[idx]
            chrom, start, end = row["chrom"], row["start"], row["end"]

            # Load original sequence
            seq = genome[chrom][start:end].seq.upper()
            seq = seq[:1310720].ljust(1310720, "N")

            for attempt in range(MAX_TRIES_PER_SEQ):
                shuffled_arr = sp.k_shuffle(seq, k=K)
                shuffled_seq = b"".join(shuffled_arr).decode()

                one_hot = one_hot_encode_sequence(shuffled_seq)
                batch = torch.from_numpy(one_hot).unsqueeze(0).to(device)  # (1, 4, L)

                preds = model(batch).cpu()
                scd = torch.sqrt((preds ** 2).sum(dim=(1, 2))).item()

                map = from_upper_triu(preds.squeeze(0), matrix_len=512, num_diags=2)
                total_var = total_variation(map)
                
                if scd < TARGET_SCD_THRESHOLD and total_var < TARGET_VAR_THRESHOLD:
                    saved_seqs.append((chrom, start, end, shuffled_seq))
                    scd_scores.append(scd)
                    totvar_scores.append(total_var)
                    print(f"Saved #{len(saved_seqs)}: idx={idx}, SCD={scd:.2f}, tot_var={total_var:.2f}")
                    break  # go to next unique sequence

    # === Save result ===
    with open(OUTPUT_FASTA, "w") as f:
        for i, (chrom, start, end, seq) in enumerate(saved_seqs):
            f.write(f">shuffled_{i}_chr{chrom}_{start}_{end}_scd{scd_scores[i]:.2f}_totvar={totvar_scores[i]:.2f}\n")
            f.write(f"{seq}\n")

    print(f"\nDone. {len(saved_seqs)} sequences saved to {OUTPUT_FASTA}")

if __name__ == "__main__":
    main()