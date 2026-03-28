"""
prepare_human_original_sequences.py
-------------------------------------
Generate original (unedited) FASTA files for all human folds.
Run once — output is shared by cross-cell-type boundary validation.
"""

import os
import torch
import pandas as pd
from helper import (
    load_and_splice, trim_and_decode, save_to_fasta
)

BASE = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

FOLDS  = range(8)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for fold in FOLDS:
    print(f"\n=== Fold {fold} ===")

    flat_regions_path = (
        f"{BASE}/analysis/flat_regions/human_flat_regions_tsv/"
        f"fold{fold}_selected_genomic_windows_centered.tsv"
    )
    ohe_dir = f"{BASE}/analysis/flat_regions/human_sequences/fold{fold}"
    out_dir = f"{BASE}/analysis/alpha_genome_validation/human_original/fold{fold}"

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(flat_regions_path, sep="\t")

    for i, row in enumerate(df.itertuples(index=False)):
        chrom, start, end = row.chrom, row.centered_start, row.centered_end
        locus = f"{chrom}_{start}_{end}"
        print(f"  [{i:>4}] {locus}")

        tensor   = load_and_splice(f"{ohe_dir}/{locus}_X.pt", edits=[], device=str(DEVICE))
        orig_seq = trim_and_decode(tensor)

        save_to_fasta(orig_seq, f"{out_dir}/{locus}.fasta", header=locus)