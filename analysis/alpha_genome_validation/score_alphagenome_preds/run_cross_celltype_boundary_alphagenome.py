"""
run_cross_celltype_boundary_alphagenome.py
------------------------------------------
Run Alpha Genome predictions on original and cross-cell-type boundary-designed
human sequences for all 8 folds. Saves per-fold TSVs with URQ mean insulation
scores for both H1hESC and HFF cell types.

Scores are computed separately for each cell type's ontology term, for both
original and designed sequences.
"""

import numpy as np
import pandas as pd
from alphagenome.models import dna_client

from helper import boundary_score, score_fasta_dir
import helper as alpha_helper

# ── Constants ─────────────────────────────────────────────────────────────────

BASE  = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"
FOLDS = range(8)

# API_KEY  = # set your API key here
ORGANISM = dna_client.Organism.HOMO_SAPIENS

CELL_TYPES = {
    "H1hESC": ["EFO:0003042"],
    "HFF":    ["EFO:0009318"],
}

STRONG_CELL_TYPE = "H1hESC"
WEAK_CELL_TYPE   = "HFF"
STRONG_TAG       = "neg0p5"
WEAK_TAG         = "neg0p2"
RUN_NAME         = "lambda/lambda_125"
TARGET_TAG       = f"{STRONG_CELL_TYPE}_strong_{STRONG_TAG}_{WEAK_CELL_TYPE}_weak_{WEAK_TAG}"

URQ_ROW_SLICE = (0,   250)
URQ_COL_SLICE = (260, 512)


# ── Main ──────────────────────────────────────────────────────────────────────

dna_model = dna_client.create(API_KEY)

for fold in FOLDS:
    print(f"\n{'='*60}\nFold {fold}\n{'='*60}")

    flat_regions_path = (
        f"{BASE}/analysis/flat_regions/human_flat_regions_tsv/"
        f"fold{fold}_selected_genomic_windows_centered.tsv"
    )
    og_fasta_dir  = f"{BASE}/analysis/alpha_genome_validation/human_original/fold{fold}"
    mod_fasta_dir = (
        f"{BASE}/analysis/alpha_genome_validation/"
        f"cross_celltype_boundary_design/{TARGET_TAG}/fold{fold}"
    )
    out_path = (
        f"{BASE}/analysis/alpha_genome_validation/"
        f"cross_celltype_boundary_design/{TARGET_TAG}/"
        f"fold{fold}_alphagenome_results.tsv"
    )

    df = pd.read_csv(flat_regions_path, sep="\t")

    for ct, ontology in CELL_TYPES.items():
        print(f"\n  Cell type: {ct} (ontology: {ontology})")

        # Override module-level organism and ontology before each cell type
        alpha_helper.ORGANISM = ORGANISM
        alpha_helper.ONTOLOGY = ontology

        print(f"    Original sequences ({len(df)} loci):")
        df[f"alpha_og_urq_{ct}"] = score_fasta_dir(
            dna_model, df, og_fasta_dir, boundary_score, label=f"original {ct}"
        )

        print(f"    Designed sequences ({len(df)} loci):")
        df[f"alpha_ed_urq_{ct}"] = score_fasta_dir(
            dna_model, df, mod_fasta_dir, boundary_score, label=f"designed {ct}"
        )

        df[f"alpha_urq_diff_{ct}"] = df[f"alpha_ed_urq_{ct}"] - df[f"alpha_og_urq_{ct}"]

    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Saved → {out_path}")