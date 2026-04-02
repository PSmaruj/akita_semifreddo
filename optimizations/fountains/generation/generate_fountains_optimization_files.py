"""
generate_fountains_optimization_files.py

Pre-generates all files needed for Ledidi + Semifreddo optimisation across
every fold of the flat-regions dataset, for fountain (antidiagonal cone) targets.

  <output_dir>/
    masks/
      fountain_c{strength}_indices.pt    ← upper-tri indices of the fountain mask
      fountain_c{strength}_vector.pt     ← full upper-tri vector with mask applied
    sequences/fold{N}/
      {chrom}_{start}_{end}_X.pt         ← one-hot encoded input sequence (4, L)
    tower_outputs/fold{N}/model{M}/
      {chrom}_{start}_{end}_tower_out.pt ← cached conv tower activations (1, 128, 640)
    targets/fountain_c{strength}/fold{N}/
      {chrom}_{start}_{end}_target.pt    ← y_bar (1, 4, M) with fountain mask applied

Usage
-----
python generate_fountains_optimization_files.py \
    --fountain_strength 0.5 \
    --max_width 120 \
    --mask_output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/feature_masks \
    --seq_output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/flat_regions \
    --target_output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/fountains
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
from utils.data_utils import one_hot_encode_sequence
from utils.optimization_utils import strength_tag, store_tower_output
from utils.data_utils import upper_triangular_to_vector, fragment_indices_in_upper_triangular

# =============================================================================
# Constants
# =============================================================================

MAP_SIZE  = 512
NUM_DIAGS = 2

FASTA_PATH = "/project2/fudenber_735/genomes/mm10/mm10.fa"

TSV_PATTERN = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/"
    "analysis/flat_regions/mouse_flat_regions_chrom_states_tsv/"
    "fold{fold}_selected_genomic_windows_centered_chrom_states.tsv"
)
MODEL_PATH_PATTERN = (
    "/home1/smaruj/akita_pytorch/models/trained_from_scratch/Vian2018_Bcells/checkpoints/"
    "Akita_v2_mouse_Vian2018_Bcells_model{model_idx}_from_scratch.pth"
)

NUM_MODELS = 4

# =============================================================================
# Fountain mask utilities
# =============================================================================


def create_symmetric_antidiagonal_cone_mask(shape=(512, 512), max_dist=120):
    """Boolean mask: True inside the symmetric antidiagonal cone."""
    H, W = shape
    i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    p1 = np.array([0, W - 1])        # top-right corner
    p2 = np.array([H - 1, 0])        # bottom-left corner

    line_vec  = p2 - p1
    line_len  = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len

    pix_vecs = np.stack([i - p1[0], j - p1[1]], axis=-1)
    t        = np.sum(pix_vecs * line_unit, axis=-1)

    center_t  = line_len / 2
    t_clipped = np.clip(t, 0, center_t)
    closest   = p1 + (t_clipped[..., None] * line_unit)

    dist_to_line    = np.linalg.norm(np.stack([i, j], axis=-1) - closest, axis=-1)
    max_allowed_dist = max_dist * (1 - (t_clipped / center_t))

    mask_upper = (t >= 0) & (t <= center_t) & (dist_to_line <= max_allowed_dist)
    mask_lower = np.flipud(np.fliplr(mask_upper))

    return mask_upper | mask_lower


def make_fountain_mask(
    strength: float,
    max_width: int = 120,
    map_size: int  = MAP_SIZE,
    num_diags: int = NUM_DIAGS,
):
    """Build the fountain mask for the antidiagonal cone.

    Only the upper-triangular portion of the cone is retained (lower-tri
    entries are dropped by the upper-tri vector conversion).

    Returns
    -------
    indices : (N,) LongTensor   — positions in the flattened upper-tri vector
    vector  : (M,) FloatTensor  — full upper-tri vector with mask values filled in
    """
    bool_mask = create_symmetric_antidiagonal_cone_mask(
        shape=(map_size, map_size), max_dist=max_width
    )

    matrix = np.zeros((map_size, map_size), dtype=np.float32)
    matrix[bool_mask] = strength

    indices = fragment_indices_in_upper_triangular(
        matrix_size=map_size, fragment_mask=bool_mask
    )
    vector = upper_triangular_to_vector(matrix, num_diags)

    return torch.tensor(indices), torch.tensor(vector).float()


# =============================================================================
# Multi-model target construction
# =============================================================================


def make_fountain_target_multi_model(
    models: list,
    X_tensor: torch.Tensor,
    feature_mask: torch.Tensor,
    feature_vector: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run all models on X, stack outputs, then overwrite fountain positions.

    Each model returns (1, 1, M); we concatenate along dim=1 to get (1, 4, M),
    matching the DesignWrapper output shape used during optimisation.

    Parameters
    ----------
    models        : list of 4 SeqNN instances (eval mode, on device)
    X_tensor      : (1, 4, L) one-hot sequence
    feature_mask  : (N,) LongTensor — upper-tri indices to overwrite
    feature_vector: (M,) FloatTensor — full upper-tri vector (same length as M)
    device        : torch.device

    Returns
    -------
    y_bar : (1, 4, M) FloatTensor on CPU
    """
    with torch.no_grad():
        outs = [m(X_tensor.to(device)) for m in models]   # each (1, 1, M)
    y = torch.cat(outs, dim=1)                             # (1, 4, M)

    y_bar = y.clone()
    y_bar[0, :, feature_mask] = feature_vector[feature_mask].to(device)
    return y_bar.cpu()


# =============================================================================
# Argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one-hot sequences, cached tower activations, and "
            "fountain targets for Ledidi + Semifreddo optimisation."
        )
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(range(8)),
        help="Fold indices to process (default: 0–7)",
    )
    parser.add_argument(
        "--fountain_strength", type=float, default=0.5,
        help="Value applied inside the antidiagonal cone (default: 0.5). "
             "Positive values enhance contacts.",
    )
    parser.add_argument(
        "--max_width", type=int, default=120,
        help="Maximum width (in bins) of the antidiagonal cone at the corners "
             "(default: 120).",
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
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tag = strength_tag(args.fountain_strength)

    # ── Output directories ────────────────────────────────────────────────────
    os.makedirs(args.mask_output_dir, exist_ok=True)

    # ── Load all 4 models ─────────────────────────────────────────────────────
    models = []
    for model_idx in range(NUM_MODELS):
        model_path = MODEL_PATH_PATTERN.format(model_idx=model_idx)
        print(f"Loading model {model_idx}: {model_path}")
        m = SeqNN()
        m.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        m.to(device).eval()
        models.append(m)

    # ── Build fountain mask ───────────────────────────────────────────────────
    print(f"Building fountain mask: strength={args.fountain_strength}, max_width={args.max_width}")
    f_indices, f_vector = make_fountain_mask(args.fountain_strength, args.max_width)

    mask_stem = f"fountain_{tag}"
    torch.save(f_indices, os.path.join(args.mask_output_dir, f"{mask_stem}_mask.pt"))
    print(f"  → saved mask to {args.mask_output_dir}/{mask_stem}_mask.pt")

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
        target_dir = os.path.join(args.target_output_dir, "targets",
                                  f"fountain_c{tag}", f"fold{fold}")

        # One tower output dir per model
        tower_dirs = [
            os.path.join(args.seq_output_dir, "mouse_tower_outputs",
                         f"model{model_idx}", f"fold{fold}")
            for model_idx in range(NUM_MODELS)
        ]

        for d in [seq_dir, target_dir] + tower_dirs:
            os.makedirs(d, exist_ok=True)

        for i, row in enumerate(df.itertuples(index=False)):
            chrom = row.chrom
            start = row.centered_start
            end   = row.centered_end
            stem  = f"{chrom}_{start}_{end}"

            print(f"  [{i+1:>4}/{len(df)}] {stem}", end="\r", flush=True)

            seq_path    = os.path.join(seq_dir,    f"{stem}_X.pt")
            target_path = os.path.join(target_dir, f"{stem}_target.pt")
            tower_paths = [
                os.path.join(tower_dirs[m], f"{stem}_tower_out.pt")
                for m in range(NUM_MODELS)
            ]

            # ── One-hot encode (skip if exists) ───────────────────────────────
            if not os.path.exists(seq_path):
                sequence = genome[chrom][start:end]
                X        = one_hot_encode_sequence(sequence)   # (4, L)
                X_tensor = torch.tensor(X).unsqueeze(0).to(device)  # (1, 4, L)
                torch.save(X_tensor.cpu(), seq_path)
            else:
                X_tensor = torch.load(seq_path, weights_only=True).to(device)

            # ── Cached tower activations, one file per model (skip if exists) ─
            for model_idx, (m, tower_path) in enumerate(zip(models, tower_paths)):
                if not os.path.exists(tower_path):
                    store_tower_output(X_tensor, m, tower_path)

            # ── Fountain target (always regenerate — strength may differ) ──────
            target = make_fountain_target_multi_model(
                models, X_tensor, f_indices, f_vector, device
            )
            torch.save(target, target_path)

        print(f"\n[fold {fold}] done")

    print("\nAll folds complete.")


if __name__ == "__main__":
    main()