"""
generate_boundaries_optimization_files.py

Pre-generates all files needed for Ledidi + Semifreddo optimisation across
every fold of the flat-regions dataset:

  <output_dir>/
    masks/
      boundary_c{strength}_indices.pt    ← upper-tri indices of the boundary mask
      boundary_c{strength}_vector.pt     ← full upper-tri vector with mask applied
    sequences/fold{N}/
      {chrom}_{start}_{end}_X.pt         ← one-hot encoded input sequence (4, L)
    tower_outputs/fold{N}/
      {chrom}_{start}_{end}_tower_out.pt ← cached conv tower activations (1, 128, 640)
    targets/boundary_c{strength}/fold{N}/
      {chrom}_{start}_{end}_target.pt    ← y_bar with boundary mask applied

Usage
-----
python generate_boundaries_optimization_files.py \
    --boundary_strength -5.0 \
    --mask_output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/feature_masks \
    --seq_output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/flat_regions \
    --target_output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/boundaries

To generate targets at multiple boundary strengths, run the script again
with a different --boundary_strength; sequences and tower outputs are
shared and will not be regenerated if they already exist.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))

from akita.model import SeqNN
from utils.data_utils import one_hot_encode_sequence
from utils.optimization_utils import strength_tag, store_tower_output, make_target
from utils.data_utils import upper_triangular_to_vector, fragment_indices_in_upper_triangular

# =============================================================================
# Constants
# =============================================================================

MAP_SIZE   = 512
HALF       = MAP_SIZE // 2
NUM_DIAGS  = 2   # diagonals excluded from the upper-tri representation

FASTA_PATH = "/project2/fudenber_735/genomes/mm10/mm10.fa"

TSV_PATTERN = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/"
    "analysis/flat_regions/mouse_flat_regions_chrom_states_tsv/"
    "fold{fold}_selected_genomic_windows_centered_chrom_states.tsv"
)
MODEL_PATH_PATTERN = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/"
    "Akita_v2_mouse_Hsieh2019_mESC_model{model_idx}_finetuned.pth"
)

# =============================================================================
# Boundary mask utilities
# =============================================================================


def make_boundary_mask(
    strength: float,
    map_size: int = 512,
    num_diags: int = 2,
):
    """Build the boundary mask that suppresses inter-compartment contacts.

    The top-right and bottom-left quadrants of the contact map are set to
    `strength` (typically negative); all other entries are 0.

    Returns
    -------
    indices : (N,) LongTensor   — positions in the flattened upper-tri vector
    vector  : (M,) FloatTensor  — full upper-tri vector with mask values filled in
    """
    half = map_size // 2

    matrix = np.zeros((map_size, map_size))
    matrix[:half, half:] = strength
    matrix[half:, :half] = strength

    fragment_bool = np.zeros((map_size, map_size), dtype=bool)
    fragment_bool[:half, half:] = True
    fragment_bool[half:, :half] = True

    vector  = upper_triangular_to_vector(matrix, num_diags)
    indices = fragment_indices_in_upper_triangular(
        matrix_size=map_size, fragment_mask=fragment_bool
    )

    return torch.tensor(indices), torch.tensor(vector).float()


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one-hot sequences, cached tower activations, and "
            "boundary targets for Ledidi + Semifreddo optimisation."
        )
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(range(8)),
        help="Fold indices to process (default: 0–7)",
    )
    parser.add_argument(
        "--boundary_strength", type=float, default=-0.5,
        help="Value applied to the off-diagonal quadrants of the boundary mask "
             "(default: -0.5). Negative values suppress contacts.",
    )
    parser.add_argument(
        "--mask_output_dir", type=str, required=True,
        help="Root directory for the generated feature mask.",
    )
    parser.add_argument(
        "--seq_output_dir", type=str, required=True,
        help="Root directory for all generated sequences and cached tower outputs.",
    )
    parser.add_argument(
        "--target_output_dir", type=str, required=True,
        help="Root directory for all generated target files.",
    )
    parser.add_argument(
        "--model_idx", type=int, default=0,
        help="Akita ensemble model index to use for predictions (default: 0).",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tag = strength_tag(args.boundary_strength)

    # ── Output directories ────────────────────────────────────────────────────
    os.makedirs(args.mask_output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = MODEL_PATH_PATTERN.format(model_idx=args.model_idx)
    print(f"Loading model: {model_path}")
    model = SeqNN()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    # ── Build boundary mask (shared across all folds and strengths) ───────────
    print(f"Building boundary mask strength={args.boundary_strength}")
    b_indices, b_vector = make_boundary_mask(args.boundary_strength)

    torch.save(b_indices, os.path.join(args.mask_output_dir, f"boundary_mask.pt"))
    print(f"  → saved to {args.mask_output_dir}/boundary_mask.pt")

    # ── Load genome ───────────────────────────────────────────────────────────
    genome = Fasta(FASTA_PATH)

    # ── Process each fold ─────────────────────────────────────────────────────
    for fold in args.folds:
        tsv_path = TSV_PATTERN.format(fold=fold)
        if not os.path.exists(tsv_path):
            print(f"\n[fold {fold}] TSV not found, skipping:\n  {tsv_path}")
            continue

        df = pd.read_csv(tsv_path, sep="\t")
        print(f"\n[fold {fold}]  {len(df)} sequences")

        seq_dir    = os.path.join(args.seq_output_dir, "mouse_sequences",     f"fold{fold}")
        tower_dir  = os.path.join(args.seq_output_dir, "mouse_tower_outputs", f"fold{fold}")
        target_dir = os.path.join(args.target_output_dir, "targets",
                                  f"boundary_{tag}", f"fold{fold}")
        for d in (seq_dir, tower_dir, target_dir):
            os.makedirs(d, exist_ok=True)

        for i, row in enumerate(df.itertuples(index=False)):
            chrom = row.chrom
            start = row.centered_start
            end   = row.centered_end
            stem  = f"{chrom}_{start}_{end}"

            print(f"  [{i+1:>4}/{len(df)}] {stem}", end="\r", flush=True)

            seq_path   = os.path.join(seq_dir,   f"{stem}_X.pt")
            tower_path = os.path.join(tower_dir, f"{stem}_tower_out.pt")
            target_path = os.path.join(target_dir, f"{stem}_target.pt")

            # One-hot encode and save (skip if already exists)
            if not os.path.exists(seq_path):
                sequence = genome[chrom][start:end]
                X        = one_hot_encode_sequence(sequence)   # (4, L)
                X_tensor = torch.tensor(X).to(device)  # (1, 4, L)
                torch.save(X_tensor.cpu(), seq_path)
            else:
                X_tensor = torch.load(seq_path, weights_only=True).to(device)

            # Cached tower activations (skip if already exists)
            if not os.path.exists(tower_path):
                store_tower_output(X_tensor, model, tower_path)

            # Boundary target (always regenerate — strength may differ)
            target = make_target(model, X_tensor, b_indices, b_vector, device)
            torch.save(target, target_path)

        print(f"\n[fold {fold}] done")

    print("\nAll folds complete.")


if __name__ == "__main__":
    main()