import torch
import numpy as np
from Bio import SeqIO
import os
import random


# === Config ===
fasta_dir = "/scratch1/smaruj/background_generation/train_backgrounds"  # path to .fasta files
output_dir = "/scratch1/smaruj/background_generation/training_pt_files"  # where to save .pt files
matrix_len = 512
diag_offset = 2
chunk_size = 100  # how many (seq, label) pairs to save per .pt file
fold = 0

os.makedirs(output_dir, exist_ok=True)

def one_hot_encode_sequence(sequence_obj):
    sequence = str(sequence_obj).upper()
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    encoded_sequence = np.array([
        base_to_int.get(base, base_to_int[random.choice("ACGT")]) for base in sequence
    ])

    one_hot_encoded = np.zeros((4, len(encoded_sequence)), dtype=np.float32)
    one_hot_encoded[encoded_sequence, np.arange(len(encoded_sequence))] = 1

    return np.expand_dims(one_hot_encoded, axis=0)


# === Number of upper triangle values (excluding diagonal_offset diagonals) ===
triu_mask = np.triu(np.ones((matrix_len, matrix_len)), k=diag_offset)
num_upper_vals = int(np.sum(triu_mask))

# === Process all FASTA chunks ===
data_list = []
file_count = 0
global_idx = 0

for chunk_id in range(5):  # You have chunks 0 to 4
    fasta_path = os.path.join(fasta_dir, f"background_sequences_chunk_{chunk_id}.fasta")
    print(f"Processing: {fasta_path}")

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        ohe = one_hot_encode_sequence(seq)
        ohe_tensor = torch.tensor(ohe, dtype=torch.float32)

        # Create flat Hi-C vector filled with zeros
        hic_tensor = torch.zeros((1, num_upper_vals), dtype=torch.float32)

        data_list.append((ohe_tensor, hic_tensor))

        # Save in chunks
        if len(data_list) >= chunk_size:
            output_path = os.path.join(output_dir, f"background_fold{fold}_{file_count}.pt")
            torch.save(data_list, output_path)
            print(f"Saved {len(data_list)} samples to {output_path}")
            data_list = []
            file_count += 1

        global_idx += 1

# Save remaining data
if data_list:
    output_path = os.path.join(output_dir, f"background_fold{fold}_{file_count}.pt")
    torch.save(data_list, output_path)
    print(f"Saved final {len(data_list)} samples to {output_path}")