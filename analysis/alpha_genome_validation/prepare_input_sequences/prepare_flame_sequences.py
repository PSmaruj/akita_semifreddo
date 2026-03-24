"""
prepare_flame_sequences.py
--------------------------
Generate designed FASTA files for flame sequences, all folds.
Original FASTAs are written separately by prepare_original_sequences.py.
"""

import torch
import pandas as pd
from helper import load_and_splice, trim_and_decode, save_to_fasta

BASE = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

FEATURE = "flame"
FOLDS   = range(8)
DEVICE  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EDIT_SPEC = [(320, "{locus}_gen_seq.pt")]

for fold in FOLDS:
    print(f"\n=== Fold {fold} ===")

    flat_regions_path = (
        f"{BASE}/analysis/flat_regions/mouse_flat_regions_chrom_states_tsv/"
        f"fold{fold}_selected_genomic_windows_centered_chrom_states.tsv"
    )
    ohe_dir     = f"{BASE}/analysis/flat_regions/mouse_sequences/fold{fold}"
    results_dir = f"{BASE}/optimizations/flames/results/flame_pos1p0/fold{fold}"
    out_dir     = f"{BASE}/analysis/alpha_genome_validation/{FEATURE}_design/fold{fold}"

    df = pd.read_csv(flat_regions_path, sep="\t")

    for i, row in enumerate(df.itertuples(index=False)):
        chrom, start, end = row.chrom, row.centered_start, row.centered_end
        locus = f"{chrom}_{start}_{end}"
        print(f"  [{i:>4}] {locus}")

        edits = [
            (bin_idx, f"{results_dir}/{tmpl.format(locus=locus)}")
            for bin_idx, tmpl in EDIT_SPEC
        ]

        tensor       = load_and_splice(f"{ohe_dir}/{locus}_X.pt", edits=edits, device=str(DEVICE))
        designed_seq = trim_and_decode(tensor)

        save_to_fasta(designed_seq, f"{out_dir}/{locus}.fasta", header=locus)