"""
generate_dots_optimization_files.py

Pre-generates all files needed for Ledidi + Semifreddo optimisation across
every fold of the flat-regions dataset, targeting a data-driven dot feature
at a specified inter-anchor distance.

  <output_dir>/
    masks/
      dot_d{distance}_mask_indices.pt    ← upper-tri indices of the dot mask
      dot_d{distance}_mask_vector.pt     ← full upper-tri vector with mask applied
    targets/dot_d{distance}/fold{N}/
      {chrom}_{start}_{end}_target.pt    ← y_bar with dot mask applied

Sequences and tower outputs are shared with the boundary pipeline and are
read from --seq_output_dir (not regenerated if they already exist).

Usage
-----
python generate_dots_optimization_files.py \
    --distances 30 50 70 \
    --dot_mask_path /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/natural_features/dots/data_driven_dot_mask.npy \
    --mask_output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/feature_masks \
    --seq_output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/flat_regions \
    --target_output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/dots

To test additional distances, run the script again with different --distances;
sequences and tower outputs will not be regenerated.
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
from utils.model_utils import store_tower_output, make_target
from utils.data_utils import one_hot_encode_sequence

# =============================================================================
# Constants
# =============================================================================

MAP_SIZE  = 512
HALF      = MAP_SIZE // 2
NUM_DIAGS = 2   # diagonals excluded from the upper-tri representation

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
# Dot mask utilities
# =============================================================================

def place_pileup_at_center(pileup, shape=(512, 512), center=(256, 281)):
    """
    Embed a 15x15 pileup matrix into a larger mask, centered at a specific
    location. Ensures symmetry along the main diagonal.

    Parameters
    ----------
    pileup : np.ndarray, shape (15, 15)
    shape  : tuple, shape of the output array
    center : (row, col) where the pileup centre should be placed

    Returns
    -------
    mask : np.ndarray, shape == shape
    """
    assert pileup.shape == (15, 15), "Pileup must be 15x15"

    H, W = shape
    r0, c0 = center
    half_size = 7  # 15x15 → half = 7

    mask = np.zeros((H, W), dtype=float)

    r_start, r_end = r0 - half_size, r0 + half_size + 1
    c_start, c_end = c0 - half_size, c0 + half_size + 1

    if not (0 <= r_start < r_end <= H) or not (0 <= c_start < c_end <= W):
        raise ValueError(
            f"Pileup does not fit inside mask shape {shape} at center {center}."
        )

    mask[r_start:r_end, c_start:c_end] = pileup

    # Mirror across the main diagonal
    for r in range(H):
        for c in range(W):
            if r < W and c < H and mask[r, c] != 0:
                mask[c, r] = mask[r, c]

    return mask


def dot_center_from_distance(distance: int, half: int = HALF):
    """
    Convert an inter-anchor distance (in bins) to a 2-D map centre.

    For distance d the two anchors sit at HALF ± d//2, placing the dot at
    (HALF - d//2, HALF + d//2) in the upper triangle.
    """
    offset = distance // 2
    return (half - offset, half + offset)


def make_dot_mask(pileup: np.ndarray, distance: int):
    """
    Build the full-map dot mask and return the flat upper-tri indices and the
    full upper-tri value vector (same convention as make_boundary_mask /
    make_flame_mask_indices).

    Returns
    -------
    indices : LongTensor of shape (K,)         — flat positions within the
              upper-tri vector where the dot mask is nonzero.
    vector  : FloatTensor of shape (N_triu,)   — full upper-tri vector with
              dot values at masked positions and 0 elsewhere.
    """
    center = dot_center_from_distance(distance)
    full_mask = place_pileup_at_center(pileup, shape=(MAP_SIZE, MAP_SIZE), center=center)

    triu_rows, triu_cols = np.triu_indices(MAP_SIZE, k=NUM_DIAGS)
    upper_tri_values = full_mask[triu_rows, triu_cols]                    # (N_triu,)

    indices = torch.tensor(np.nonzero(upper_tri_values)[0], dtype=torch.long)
    vector  = torch.tensor(upper_tri_values, dtype=torch.float32)         # full length
    return indices, vector


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dot-feature targets for Ledidi + Semifreddo optimisation."
        )
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(range(8)),
        help="Fold indices to process (default: 0–7).",
    )
    parser.add_argument(
        "--distances", type=int, nargs="+", default=[30, 50, 70],
        help="Inter-anchor distances in bins to generate targets for (default: 30 50 70).",
    )
    parser.add_argument(
        "--dot_mask_path", type=str,
        default=(
            "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/"
            "analysis/natural_features/dots/data_driven_dot_mask.npy"
        ),
        help="Path to the 15x15 data-driven dot pileup (.npy).",
    )
    parser.add_argument(
        "--mask_output_dir", type=str, required=True,
        help="Root directory for saving the generated dot masks.",
    )
    parser.add_argument(
        "--seq_output_dir", type=str, required=True,
        help="Root directory containing pre-generated sequences and tower outputs "
             "(shared with the boundary pipeline).",
    )
    parser.add_argument(
        "--target_output_dir", type=str, required=True,
        help="Root directory for all generated dot target files.",
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

    # ── Load data-driven pileup ───────────────────────────────────────────────
    pileup = np.load(args.dot_mask_path)
    assert pileup.shape == (15, 15), (
        f"Expected a 15x15 pileup at {args.dot_mask_path}, got {pileup.shape}"
    )
    print(f"Loaded dot pileup from {args.dot_mask_path}")

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = MODEL_PATH_PATTERN.format(model_idx=args.model_idx)
    print(f"Loading model: {model_path}")
    model = SeqNN()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    os.makedirs(args.mask_output_dir, exist_ok=True)

    # ── Load genome ───────────────────────────────────────────────────────────
    genome = Fasta(FASTA_PATH)
    
    # ── Loop over distances ───────────────────────────────────────────────────
    for distance in args.distances:
        print(f"\n{'='*60}")
        print(f"Distance: {distance} bins  →  center {dot_center_from_distance(distance)}")

        # Build and save dot mask for this distance
        d_indices, d_vector = make_dot_mask(pileup, distance)
        
        torch.save(d_indices, os.path.join(args.mask_output_dir, f"dot_d{distance}_mask.pt"))
        print(f"  → mask saved to {args.mask_output_dir}/dot_d{distance}_mask.pt")

        # ── Loop over folds ───────────────────────────────────────────────────
        for fold in args.folds:
            tsv_path = TSV_PATTERN.format(fold=fold)
            if not os.path.exists(tsv_path):
                print(f"\n  [fold {fold}] TSV not found, skipping:\n    {tsv_path}")
                continue

            df = pd.read_csv(tsv_path, sep="\t")
            print(f"\n  [fold {fold}]  {len(df)} sequences")

            seq_dir    = os.path.join(args.seq_output_dir, "mouse_sequences",     f"fold{fold}")
            tower_dir  = os.path.join(args.seq_output_dir, "mouse_tower_outputs", f"fold{fold}")
            target_dir = os.path.join(args.target_output_dir, "targets",
                                      f"dot_d{distance}", f"fold{fold}")
            os.makedirs(target_dir, exist_ok=True)

            for i, row in enumerate(df.itertuples(index=False)):
                chrom = row.chrom
                start = row.centered_start
                end   = row.centered_end
                stem  = f"{chrom}_{start}_{end}"

                print(f"    [{i+1:>4}/{len(df)}] {stem}", end="\r", flush=True)

                seq_path   = os.path.join(seq_dir,   f"{stem}_X.pt")
                tower_path  = os.path.join(tower_dir,  f"{stem}_tower_out.pt")
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

                # Reuse make_target with dot mask (same interface as boundary)
                target = make_target(model, X_tensor, d_indices, d_vector, device)
                torch.save(target, target_path)

            print(f"\n  [fold {fold}] done")

    print("\nAll distances and folds complete.")


if __name__ == "__main__":
    main()