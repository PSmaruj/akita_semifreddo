"""
prepare_boundary_suppression_sequences.py
------------------------------------------
Generate designed FASTA files for boundary suppression sequences, all folds.
Original FASTAs are written separately by prepare_original_sequences.py.
"""

import torch
import pandas as pd
from helper import load_and_splice, trim_and_decode, save_to_fasta

BASE = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

FEATURE   = "boundary_suppression"
FOLDS     = range(8)
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EDIT_SPEC = [(320, "{locus}_gen_seq.pt")]

DF_PATH = f"{BASE}/optimizations/boundaries/successful_optimizations_-0.5.tsv"
df = pd.read_csv(DF_PATH, sep="\t")

for fold in FOLDS:
    print(f"\n=== Fold {fold} ===")

    df_fold = df[df["fold"] == fold]
    print(f"  {len(df_fold)} loci in this fold")

    ohe_dir     = f"{BASE}/analysis/flat_regions/mouse_sequences/fold{fold}"
    results_dir = f"{BASE}/optimizations/boundary_suppression/results/fold{fold}"
    out_dir     = f"{BASE}/analysis/alpha_genome_validation/{FEATURE}_design/fold{fold}"

    for i, row in enumerate(df_fold.itertuples(index=False)):
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