"""
generate_flames_optimization_files.py

Pre-generates all files needed for Ledidi + Semifreddo optimisation across
every fold of the flat-regions dataset, targeting flame (stripe) features:

  <output_dir>/
    masks/
      flame_c{strength}_mask.pt          ← upper-tri indices of the flame mask
    sequences/fold{N}/
      {chrom}_{start}_{end}_X.pt         ← one-hot encoded input sequence (4, L)  [shared]
    tower_outputs/fold{N}/
      {chrom}_{start}_{end}_tower_out.pt ← cached conv tower activations (1, 128, 640)  [shared]
    targets/flame_c{strength}/fold{N}/
      {chrom}_{start}_{end}_target.pt    ← y_bar with flame mask applied

Sequences and tower outputs are shared with the boundary optimisation script
and will not be regenerated if they already exist.

Usage
-----
python generate_flames_optimization_files.py \
    --flame_strength 1.0 \
    --mask_output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/feature_masks \
    --seq_output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/flat_regions \
    --target_output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/flames

To generate targets at multiple flame strengths, run the script again with a
different --flame_strength; sequences and tower outputs will be reused.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta

sys.path.append(os.path.abspath("/home1/smaruj/akita_pytorch/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))

from akita.model import SeqNN
from semifreddo.optimization_loop import strength_tag
from utils.data_utils import one_hot_encode_sequence
from utils.model_utils import store_tower_output, make_target

# =============================================================================
# Constants
# =============================================================================

MAP_SIZE    = 512
HALF        = MAP_SIZE // 2
NUM_DIAGS   = 2   # diagonals excluded from the upper-tri representation
FLAME_WIDTH = 3

FASTA_PATH = "/project2/fudenber_735/genomes/mm10/mm10.fa"

TSV_PATTERN = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/"
    "analysis/flat_regions/mouse_flat_regions_chrom_states_tsv/"
    "fold{fold}_selected_genomic_windows_centered_chrom_states.tsv"
)
MODEL_PATH_PATTERN = (
    "/home1/smaruj/akita_pytorch/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/"
    "Akita_v2_mouse_Hsieh2019_mESC_model{model_idx}_finetuned.pth"
)


# =============================================================================
# Flame mask
# =============================================================================

def create_flame_mask(
    shape: tuple[int, int] = (MAP_SIZE, MAP_SIZE),
    center: tuple[int, int] | None = None,
    flame_width: int = FLAME_WIDTH,
    value: float = 0.5,
) -> np.ndarray:
    """
    Creates a flame/stripe-shaped mask: vertical stripe (top half of map) and
    horizontal stripe (left half of map) forming an 'L' shape.

    Parameters
    ----------
    shape       : (rows, cols) of the mask.
    center      : (row, col) origin of the flame; defaults to map centre.
    flame_width : Total width of each stripe in bins.
    value       : Fill value within the stripe.

    Returns
    -------
    mask : 2-D float array of the requested shape.
    """
    H, W = shape
    half_r, half_c = center if center is not None else (H // 2, W // 2)
    half_w = flame_width // 2

    mask = np.zeros((H, W), dtype=float)
    mask[:half_r, half_c - half_w : half_c + half_w] = value   # vertical stripe
    mask[half_r - half_w : half_r + half_w, :half_c] = value   # horizontal stripe
    return mask


def make_flame_mask_indices(value: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the flat upper-tri index tensor and full upper-tri value vector for
    the flame mask, matching the format expected by make_target().

    Returns
    -------
    f_indices : LongTensor of shape (K,) — flat positions within the upper-tri
                vector where the flame mask is nonzero.
    f_vector  : FloatTensor of shape (N_upper_tri,) — full upper-tri vector with
                flame values at masked positions and 0 elsewhere.
    """
    full_mask = create_flame_mask(value=value)

    rows, cols = np.triu_indices(MAP_SIZE, k=NUM_DIAGS)
    upper_tri_values = full_mask[rows, cols]                          # (N_upper_tri,)

    f_indices = torch.tensor(np.nonzero(upper_tri_values)[0], dtype=torch.long)
    f_vector  = torch.tensor(upper_tri_values, dtype=torch.float32)  # full length
    return f_indices, f_vector


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one-hot sequences, cached tower activations, and "
            "flame targets for Ledidi + Semifreddo optimisation."
        )
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(range(8)),
        help="Fold indices to process (default: 0–7)",
    )
    parser.add_argument(
        "--flame_strength", type=float, default=0.5,
        help="Positive value applied to the flame stripe mask (default: 0.5).",
    )
    parser.add_argument(
        "--mask_output_dir", type=str, required=True,
        help="Root directory for the generated feature mask.",
    )
    parser.add_argument(
        "--seq_output_dir", type=str, required=True,
        help="Root directory for all generated sequences and cached tower outputs "
             "(shared with boundary script).",
    )
    parser.add_argument(
        "--target_output_dir", type=str, required=True,
        help="Root directory for all generated flame target files.",
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

    if args.flame_strength <= 0:
        raise ValueError(
            f"--flame_strength must be positive, got {args.flame_strength}"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tag = strength_tag(args.flame_strength)

    # ── Output directories ────────────────────────────────────────────────────
    os.makedirs(args.mask_output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = MODEL_PATH_PATTERN.format(model_idx=args.model_idx)
    print(f"Loading model: {model_path}")
    model = SeqNN()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    # ── Build flame mask ──────────────────────────────────────────────────────
    print(f"Building flame mask  strength={args.flame_strength}  width={FLAME_WIDTH}")
    f_indices, f_vector = make_flame_mask_indices(args.flame_strength)

    mask_path = os.path.join(args.mask_output_dir, f"flame_mask.pt")
    torch.save(f_indices, mask_path)
    print(f"  → saved to {mask_path}")

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
                                  f"flame_{tag}", f"fold{fold}")
        for d in (seq_dir, tower_dir, target_dir):
            os.makedirs(d, exist_ok=True)

        for i, row in enumerate(df.itertuples(index=False)):
            chrom = row.chrom
            start = row.centered_start
            end   = row.centered_end
            stem  = f"{chrom}_{start}_{end}"

            print(f"  [{i+1:>4}/{len(df)}] {stem}", end="\r", flush=True)

            seq_path    = os.path.join(seq_dir,    f"{stem}_X.pt")
            tower_path  = os.path.join(tower_dir,  f"{stem}_tower_out.pt")
            target_path = os.path.join(target_dir, f"{stem}_target.pt")

            # One-hot encode and save (skip if already exists)
            if not os.path.exists(seq_path):
                sequence = genome[chrom][start:end]
                X        = one_hot_encode_sequence(sequence)   # (4, L)
                X_tensor = torch.tensor(X).to(device)
                torch.save(X_tensor.cpu(), seq_path)
            else:
                X_tensor = torch.load(seq_path, weights_only=True).to(device)

            # Cached tower activations — shared; skip if already exists
            if not os.path.exists(tower_path):
                store_tower_output(X_tensor, model, tower_path)

            # Flame target (always regenerate — strength may differ)
            target = make_target(model, X_tensor, f_indices, f_vector, device)
            torch.save(target, target_path)

        print(f"\n[fold {fold}] done")

    print("\nAll folds complete.")


if __name__ == "__main__":
    main()