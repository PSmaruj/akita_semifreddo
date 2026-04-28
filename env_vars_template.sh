#!/bin/bash
#==============================================================================
# AkitaSF Environment Variables
#==============================================================================
#
# SETUP INSTRUCTIONS:
#   1. Copy this file:
#        cp env_vars_template.sh env_vars.sh
#   2. Fill in the paths below to match your local environment.
#   3. Source the file before running any AkitaSF scripts:
#        source env_vars.sh
#      Or add it to your ~/.bashrc to make it permanent:
#        echo "source /path/to/akita_semifreddo/env_vars.sh" >> ~/.bashrc
#
# WARNING: env_vars.sh contains your AlphaGenome API key — never commit it.
# It is listed in .gitignore to prevent accidental exposure. Only
# env_vars_template.sh (this file, with placeholder values) is version-controlled.
#==============================================================================

# ── Repository root ───────────────────────────────────────────────────────────
export AKITA_SF_DIR=/path/to/akita_semifreddo

# ── Model weights (extracted from Zenodo: https://doi.org/10.5281/zenodo.19599537) ──
# Path to the finetuned model checkpoint (.pth) for each organism.
export MOUSE_MODEL_CKPT=/path/to/models/finetuned/mouse/<dataset>/checkpoints/<model>.pth
export HUMAN_MODEL_CKPT=/path/to/models/finetuned/human/<dataset>/checkpoints/<model>.pth

# ── Reference genomes ─────────────────────────────────────────────────────────
export MOUSE_GENOME_FASTA=/path/to/mm10.fa
export HUMAN_GENOME_FASTA=/path/to/hg38.fa

# ── Chromosome sizes ──────────────────────────────────────────────────────────
export MOUSE_CHROM_SIZES_FILE=/path/to/mm10.chrom.sizes.reduced

# ── Flat regions TSV directories ──────────────────────────────────────────────
# Directories containing per-fold flat region TSV files.
# Files are expected to follow the naming convention:
#   fold{N}_selected_genomic_windows_centered.tsv
export MOUSE_FLAT_REGIONS_TSV=/path/to/analysis/flat_regions/mouse_flat_regions
export HUMAN_FLAT_REGIONS_TSV=/path/to/analysis/flat_regions/human_flat_regions

# ── AlphaGenome API key ───────────────────────────────────────────────────────
# Obtain your key from: https://deepmind.google/blog/alphagenome-ai-for-better-understanding-the-genome/
export ALPHAGENOME_API_KEY=your_api_key_here

echo "AkitaSF environment variables set."#!/bin/bash