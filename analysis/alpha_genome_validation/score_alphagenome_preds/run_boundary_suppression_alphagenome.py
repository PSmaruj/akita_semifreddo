"""
run_boundary_suppression_alphagenome.py
----------------------------------------
Run Alpha Genome predictions on original and boundary-suppression-designed
sequences for all 8 folds and save per-fold TSVs with boundary and URQ scores.
"""

import pandas as pd
from alphagenome.models import dna_client
from helper import boundary_score, score_fasta_dir

# ── Config ─────────────────────────────────────────────────────────────────────

BASE    = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"
FEATURE = "boundary_suppression"
FOLDS   = range(8)

API_KEY = "AIzaSyBh9ICxEr8WOH63OELhl13TtqI1xvNo6LY"

DF_PATH = f"{BASE}/optimizations/boundaries/successful_optimizations_-0.5.tsv"

# ── Main ───────────────────────────────────────────────────────────────────────

dna_model = dna_client.create(API_KEY)
df_all    = pd.read_csv(DF_PATH, sep="\t")

for fold in FOLDS:
    print(f"\n{'='*60}\nFold {fold}\n{'='*60}")

    df = df_all[df_all["fold"] == fold].copy()
    print(f"  {len(df)} loci in this fold")

    og_fasta_dir  = f"{BASE}/analysis/alpha_genome_validation/{FEATURE}_original/fold{fold}"
    mod_fasta_dir = f"{BASE}/analysis/alpha_genome_validation/{FEATURE}_design/fold{fold}"
    out_path      = f"{BASE}/analysis/alpha_genome_validation/{FEATURE}_design/fold{fold}_alphagenome_results.tsv"

    print(f"\nOriginal sequences ({len(df)} loci):")
    df["alpha_og_boundary_score"] = score_fasta_dir(dna_model, df, og_fasta_dir, boundary_score, label="original boundary")

    print(f"\nDesigned sequences ({len(df)} loci):")
    df["alpha_ed_boundary_score"] = score_fasta_dir(dna_model, df, mod_fasta_dir, boundary_score, label="designed boundary")

    df["alpha_boundary_score_diff"] = df["alpha_ed_boundary_score"] - df["alpha_og_boundary_score"]

    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved → {out_path}")