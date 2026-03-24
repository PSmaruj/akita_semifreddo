"""
run_boundary_alphagenome.py
---------------------------
Run Alpha Genome predictions on original and boundary-designed sequences
for all 8 folds and save per-fold TSVs with URQ mean insulation scores.
"""

import numpy as np
import pandas as pd
from alphagenome.models import dna_client

from helper import boundary_score, score_fasta_dir

# ── Constants ──────────────────────────────────────────────────────────────────

BASE    = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"
FEATURE = "boundary"
FOLDS   = range(8)

API_KEY      = "AIzaSyBh9ICxEr8WOH63OELhl13TtqI1xvNo6LY"
ORGANISM     = dna_client.Organism.MUS_MUSCULUS
ONTOLOGY     = ["EFO:0004038"]  # mESC

# URQ region: upper-right quadrant of the 512x512 contact map
URQ_ROW_SLICE = (0,   250)
URQ_COL_SLICE = (260, 512)


# ── Main ───────────────────────────────────────────────────────────────────────

dna_model = dna_client.create(API_KEY)

for fold in FOLDS:
    print(f"\n{'='*60}\nFold {fold}\n{'='*60}")

    flat_regions_path = (
        f"{BASE}/analysis/flat_regions/mouse_flat_regions_chrom_states_tsv/"
        f"fold{fold}_selected_genomic_windows_centered_chrom_states.tsv"
    )
    og_fasta_dir  = f"{BASE}/analysis/alpha_genome_validation/original/fold{fold}"
    mod_fasta_dir = f"{BASE}/analysis/alpha_genome_validation/{FEATURE}_design/fold{fold}"
    out_path      = f"{BASE}/analysis/alpha_genome_validation/{FEATURE}_design/fold{fold}_alphagenome_results.tsv"

    df = pd.read_csv(flat_regions_path, sep="\t")

    print(f"\nOriginal sequences ({len(df)} loci):")
    df["alpha_og_urq"] = score_fasta_dir(dna_model, df, og_fasta_dir,  boundary_score, label="original")

    print(f"\nDesigned sequences ({len(df)} loci):")
    df["alpha_ed_urq"] = score_fasta_dir(dna_model, df, mod_fasta_dir, boundary_score, label="designed")

    df["alpha_urq_diff"] = df["alpha_ed_urq"] - df["alpha_og_urq"]

    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved → {out_path}")