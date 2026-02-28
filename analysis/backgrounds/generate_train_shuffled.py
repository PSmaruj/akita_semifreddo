import pandas as pd
import torch
import seqpro as sp
import random
import numpy as np
from pyfaidx import Fasta


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
    dx = pred[:, 1:] - pred[:, :-1]
    dy = pred[1:, :] - pred[:-1, :]
    
    dx = dx[~np.isnan(dx)]
    dy = dy[~np.isnan(dy)]

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
    TSV_PATH = "/home1/smaruj/akitaV2-analyses/experiments/background_generation/sequence_shuffling/input_data/shuffled_600seqs.tsv"
    K_choices = [2, 4, 8, 16]
    TARGET_COUNT = 590

    # === LOAD ===
    df = pd.read_csv(TSV_PATH, sep='\t')
    df = df[df["shuffle_parameter"] == 1]
    seen_indices = set()
    saved_seqs = []

    fasta_file = "/project/fudenber_735/genomes/mm10/mm10.fa"
    genome = Fasta(fasta_file)

    # === Shuffling loop ===
    chunk_size = 50
    chunk_idx = 0
    
    with torch.no_grad():
        while len(saved_seqs) < TARGET_COUNT and len(seen_indices) < len(df):
            idx = random.choice(df.index.difference(seen_indices))
            seen_indices.add(idx)

            row = df.loc[idx]
            chrom, start, end = row["chrom"], row["start"], row["end"]

            # Load original sequence
            seq = genome[chrom][start:end].seq.upper()
            seq = seq[:1310720].ljust(1310720, "N")

            # choosing randomly k
            k_value = random.choice(K_choices)
            shuffled_arr = sp.k_shuffle(seq, k=k_value)
            shuffled_seq = b"".join(shuffled_arr).decode()

            saved_seqs.append((chrom, start, end, shuffled_seq))

            # === Flush to disk every 50 sequences ===
            if len(saved_seqs) == chunk_size or (len(saved_seqs) + chunk_idx * chunk_size) >= TARGET_COUNT:
                output_path = f"/scratch1/smaruj/background_generation/train_shuffled/shuffled_sequences_chunk_{chunk_idx}.fasta"
                with open(output_path, "w") as f:
                    for i, (chrom, start, end, seq) in enumerate(saved_seqs):
                        f.write(f">shuffled_{chunk_idx*chunk_size + i}_chr{chrom}_{start}_{end}\n")
                        f.write(f"{seq}\n")

                print(f"\nWrote {len(saved_seqs)} sequences to {output_path}")
                chunk_idx += 1
                saved_seqs.clear()

if __name__ == "__main__":
    main()