"""
prepare_cross_celltype_boundary_sequences.py
---------------------------------------------
Generate designed FASTA files for cross-cell-type boundary sequences, all folds.
Original FASTAs are written separately by prepare_human_original_sequences.py.

The designed central bin (bin 320 in 640-bin tower space) is spliced into the
full human sequence for each optimised window.
"""

import os
import torch
import pandas as pd
from helper import (
    load_and_splice, trim_and_decode, save_to_fasta
)

BASE = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

STRONG_CELL_TYPE = "H1hESC"
WEAK_CELL_TYPE   = "HFF"
STRONG_TAG       = "neg0p5"
WEAK_TAG         = "neg0p2"
TARGET_TAG       = f"{STRONG_CELL_TYPE}_strong_{STRONG_TAG}_{WEAK_CELL_TYPE}_weak_{WEAK_TAG}"
RUN_NAME         = f"results/{TARGET_TAG}"

FOLDS  = range(8)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Central bin index in 640-bin tower space (256 map + 64 cropping = 320)
EDIT_SPEC = [(320, "{locus}_gen_seq.pt")]

for fold in FOLDS:
    print(f"\n=== Fold {fold} ===")

    flat_regions_path = (
        f"{BASE}/analysis/flat_regions/human_flat_regions_tsv/"
        f"fold{fold}_selected_genomic_windows_centered.tsv"
    )
    ohe_dir     = f"{BASE}/analysis/flat_regions/human_sequences/fold{fold}"
    results_dir = (
        f"{BASE}/optimizations/cross_celltype_boundaries/"
        f"{RUN_NAME}/fold{fold}"
    )
    out_dir = (
        f"{BASE}/analysis/alpha_genome_validation/"
        f"cross_celltype_boundary_design/{TARGET_TAG}/fold{fold}"
    )

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(flat_regions_path, sep="\t")

    for i, row in enumerate(df.itertuples(index=False)):
        chrom, start, end = row.chrom, row.centered_start, row.centered_end
        locus = f"{chrom}_{start}_{end}"
        print(f"  [{i:>4}] {locus}")

        gen_seq_path = f"{results_dir}/{locus}_gen_seq.pt"
        if not os.path.exists(gen_seq_path):
            print(f"    skipping — no generated sequence found")
            continue

        edits = [
            (bin_idx, f"{results_dir}/{tmpl.format(locus=locus)}")
            for bin_idx, tmpl in EDIT_SPEC
        ]

        tensor       = load_and_splice(f"{ohe_dir}/{locus}_X.pt", edits=edits, device=str(DEVICE))
        designed_seq = trim_and_decode(tensor)

        save_to_fasta(designed_seq, f"{out_dir}/{locus}.fasta", header=locus)