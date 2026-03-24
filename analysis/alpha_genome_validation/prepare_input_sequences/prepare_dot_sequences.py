"""
prepare_dot_sequences.py
------------------------
Generate designed FASTA files for dot sequences, all folds.
Original FASTAs are written separately by prepare_original_sequences.py.

The dot design stores both edited bins as a single .pt tensor of shape
[1, 4, 4096]. It is split in half before splicing: the first 2048 bp
replace bin 295 (anchor 0) and the second 2048 bp replace bin 345 (anchor 1).
"""

import torch
import pandas as pd
from helper import load_and_splice, trim_and_decode, save_to_fasta

BASE = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

FEATURE      = "dot"
FOLDS        = range(8)
DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BIN_SIZE     = 2048
BIN_ANCHOR_0 = 295
BIN_ANCHOR_1 = 345

for fold in FOLDS:
    print(f"\n=== Fold {fold} ===")

    flat_regions_path = (
        f"{BASE}/analysis/flat_regions/mouse_flat_regions_chrom_states_tsv/"
        f"fold{fold}_selected_genomic_windows_centered_chrom_states.tsv"
    )
    ohe_dir     = f"{BASE}/analysis/flat_regions/mouse_sequences/fold{fold}"
    results_dir = f"{BASE}/optimizations/dots/results/dot_d50/fold{fold}"
    out_dir     = f"{BASE}/analysis/alpha_genome_validation/{FEATURE}_design/fold{fold}"

    df = pd.read_csv(flat_regions_path, sep="\t")

    for i, row in enumerate(df.itertuples(index=False)):
        chrom, start, end = row.chrom, row.centered_start, row.centered_end
        locus = f"{chrom}_{start}_{end}"
        print(f"  [{i:>4}] {locus}")

        gen_seq = torch.load(f"{results_dir}/{locus}_gen_seq.pt", map_location=DEVICE)
        edits = [
            (BIN_ANCHOR_0, gen_seq[:, :, :BIN_SIZE]),
            (BIN_ANCHOR_1, gen_seq[:, :, BIN_SIZE:]),
        ]

        tensor       = load_and_splice(f"{ohe_dir}/{locus}_X.pt", edits=edits, device=str(DEVICE))
        designed_seq = trim_and_decode(tensor)

        save_to_fasta(designed_seq, f"{out_dir}/{locus}.fasta", header=locus)