"""
run_dot_rmsd_alphagenome.py
---------------------------
Compute Alpha Genome RMSD (Root Mean Squared Difference) for dot designs
across all 8 folds and append the result to the existing per-fold TSVs.
"""

import pandas as pd
from alphagenome.models import dna_client
from helper import rmsd_fasta_dirs

# ── Config ─────────────────────────────────────────────────────────────────────

BASE    = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"
FEATURE = "dot"
FOLDS   = range(8)

API_KEY = "AIzaSyBh9ICxEr8WOH63OELhl13TtqI1xvNo6LY"

# ── Main ───────────────────────────────────────────────────────────────────────

dna_model = dna_client.create(API_KEY)

for fold in FOLDS:
    print(f"\n{'='*60}\nFold {fold}\n{'='*60}")

    og_fasta_dir  = f"{BASE}/analysis/alpha_genome_validation/original/fold{fold}"
    mod_fasta_dir = f"{BASE}/analysis/alpha_genome_validation/{FEATURE}_design/fold{fold}"
    tsv_path      = f"{BASE}/analysis/alpha_genome_validation/{FEATURE}_design/fold{fold}_alphagenome_results.tsv"

    df = pd.read_csv(tsv_path, sep="\t")

    print(f"\nComputing SCD ({len(df)} loci):")
    df["alpha_scd"] = rmsd_fasta_dirs(dna_model, df, og_fasta_dir, mod_fasta_dir, label=f"fold{fold}")

    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"\nSaved → {tsv_path}")