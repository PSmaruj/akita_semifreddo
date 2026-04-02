"""
generate_cross_celltype_boundary_files.py

Pre-generates all files needed for cross-cell-type boundary optimisation:
designing sequences that fold into strong boundaries in one human cell type
and weak boundaries in another simultaneously.

Uses two H1hESC models (indices 0,1) and two HFF models (indices 0,1).
The combined target has shape (1, 4, N_triu):
  slots 0–1 : H1hESC baseline prediction + boundary mask at strong/weak strength
  slots 2–3 : HFF    baseline prediction + boundary mask at weak/strong strength

Output layout
-------------
  <output_dir>/
    human_sequences/fold{N}/
      {chrom}_{start}_{end}_X.pt
    human_tower_outputs/H1hESC/model{M}/fold{N}/
      {chrom}_{start}_{end}_tower_out.pt
    human_tower_outputs/HFF/model{M}/fold{N}/
      {chrom}_{start}_{end}_tower_out.pt
    targets/H1hESC_{direction}_{tag1}_HFF_{direction}_{tag2}/fold{N}/
      {chrom}_{start}_{end}_target.pt

Usage
-----
python generate_cross_celltype_boundary_files.py \
    --strong_cell_type HFF \
    --strong_strength -0.5 \
    --weak_strength -0.2 \
    --seq_output_dir  /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/analysis/flat_regions \
    --target_output_dir /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/cross_celltype_boundaries
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
from semifreddo.optimization_loop import strength_tag
from utils.data_utils import one_hot_encode_sequence
from utils.model_utils import store_tower_output
from utils.data_utils import upper_triangular_to_vector, fragment_indices_in_upper_triangular

# =============================================================================
# Constants
# =============================================================================

MAP_SIZE   = 512
NUM_DIAGS  = 2
NUM_MODELS = 2       # model indices 0 and 1 per cell type

FASTA_PATH = "/project2/fudenber_735/genomes/hg38/hg38.fa"

TSV_PATTERN = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/"
    "analysis/flat_regions/human_flat_regions_tsv/"
    "fold{fold}_selected_genomic_windows_centered.tsv"
)
MODEL_PATH_PATTERN = (
    "/home1/smaruj/pytorch_akita/models/finetuned/human/"
    "Krietenstein2019_{cell_type}/checkpoints/"
    "Akita_v2_human_Krietenstein2019_{cell_type}_model{model_idx}_finetuned.pth"
)

CELL_TYPES = ["H1hESC", "HFF"]

# =============================================================================
# Boundary mask
# =============================================================================


def make_boundary_mask(strength: float, map_size: int = MAP_SIZE,
                       num_diags: int = NUM_DIAGS):
    """Off-diagonal quadrant boundary mask (identical logic to mouse script)."""
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
# Cross-cell-type target construction
# =============================================================================


def make_cross_celltype_target(
    h1hesc_models: list,
    hff_models: list,
    X_tensor: torch.Tensor,
    h1hesc_indices: torch.Tensor,
    h1hesc_vector: torch.Tensor,
    hff_indices: torch.Tensor,
    hff_vector: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build a (1, 4, N_triu) target with per-cell-type boundary strengths.

    Slots 0–1 : H1hESC model predictions + H1hESC boundary mask
    Slots 2–3 : HFF    model predictions + HFF    boundary mask

    Parameters
    ----------
    h1hesc_models  : list of 2 SeqNN instances for H1hESC
    hff_models     : list of 2 SeqNN instances for HFF
    X_tensor       : (1, 4, L) one-hot sequence on device
    h1hesc_indices : (K,) upper-tri indices for H1hESC boundary mask
    h1hesc_vector  : (M,) full upper-tri vector for H1hESC mask
    hff_indices    : (K,) upper-tri indices for HFF boundary mask
    hff_vector     : (M,) full upper-tri vector for HFF mask
    device         : torch.device

    Returns
    -------
    target : (1, 4, N_triu) FloatTensor on CPU
    """
    with torch.no_grad():
        h1hesc_outs = [m(X_tensor) for m in h1hesc_models]   # each (1, 1, N_triu)
        hff_outs    = [m(X_tensor) for m in hff_models]

    y = torch.cat(h1hesc_outs + hff_outs, dim=1)   # (1, 4, N_triu)
    y_bar = y.clone()

    # Overwrite H1hESC slots with H1hESC-strength mask
    y_bar[0, 0:2, h1hesc_indices] = h1hesc_vector[h1hesc_indices].to(device)
    # Overwrite HFF slots with HFF-strength mask
    y_bar[0, 2:4, hff_indices]    = hff_vector[hff_indices].to(device)

    return y_bar.cpu()


# =============================================================================
# Argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate files for cross-cell-type boundary optimisation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--folds", type=int, nargs="+", default=list(range(8)),
                   help="Fold indices to process (default: 0–7)")
    p.add_argument("--strong_cell_type", choices=CELL_TYPES, required=True,
                   help="Cell type that receives the strong boundary target.")
    p.add_argument("--strong_strength", type=float, default=-0.5,
                   help="Boundary mask value for the strong cell type (default: -0.5).")
    p.add_argument("--weak_strength", type=float, default=-0.2,
                   help="Boundary mask value for the weak cell type (default: -0.2).")
    p.add_argument("--seq_output_dir", type=str, required=True,
                   help="Root dir for human sequences and tower outputs.")
    p.add_argument("--target_output_dir", type=str, required=True,
                   help="Root dir for target files.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Resolve which cell type gets which strength ───────────────────────────
    weak_cell_type = [ct for ct in CELL_TYPES if ct != args.strong_cell_type][0]

    cell_type_strengths = {
        args.strong_cell_type: args.strong_strength,
        weak_cell_type:        args.weak_strength,
    }
    print(f"Strong boundary ({args.strong_strength}) → {args.strong_cell_type}")
    print(f"Weak   boundary ({args.weak_strength})  → {weak_cell_type}")

    # ── Build target directory tag ────────────────────────────────────────────
    strong_tag = strength_tag(args.strong_strength)
    weak_tag   = strength_tag(args.weak_strength)
    target_tag = (
        f"{args.strong_cell_type}_strong_{strong_tag}_"
        f"{weak_cell_type}_weak_{weak_tag}"
    )
    print(f"Target tag: {target_tag}")

    # ── Load models ───────────────────────────────────────────────────────────
    loaded_models = {}
    for cell_type in CELL_TYPES:
        loaded_models[cell_type] = []
        for model_idx in range(NUM_MODELS):
            path = MODEL_PATH_PATTERN.format(
                cell_type=cell_type, model_idx=model_idx
            )
            print(f"Loading {cell_type} model {model_idx}: {path}")
            m = SeqNN()
            m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            m.to(device).eval()
            loaded_models[cell_type].append(m)

    h1hesc_models = loaded_models["H1hESC"]
    hff_models    = loaded_models["HFF"]

    # ── Build boundary masks ──────────────────────────────────────────────────
    masks = {}
    for cell_type in CELL_TYPES:
        strength = cell_type_strengths[cell_type]
        print(f"Building boundary mask for {cell_type} (strength={strength})")
        indices, vector = make_boundary_mask(strength)
        masks[cell_type] = {"indices": indices, "vector": vector}

    h1hesc_indices = masks["H1hESC"]["indices"]
    h1hesc_vector  = masks["H1hESC"]["vector"]
    hff_indices    = masks["HFF"]["indices"]
    hff_vector     = masks["HFF"]["vector"]

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

        seq_dir    = os.path.join(args.seq_output_dir,    "human_sequences",    f"fold{fold}")
        target_dir = os.path.join(args.target_output_dir, "targets", target_tag, f"fold{fold}")

        tower_dirs = {
            cell_type: [
                os.path.join(args.seq_output_dir, "human_tower_outputs",
                             cell_type, f"model{m}", f"fold{fold}")
                for m in range(NUM_MODELS)
            ]
            for cell_type in CELL_TYPES
        }

        for d in [seq_dir, target_dir] + [
            td for dirs in tower_dirs.values() for td in dirs
        ]:
            os.makedirs(d, exist_ok=True)

        for i, row in enumerate(df.itertuples(index=False)):
            chrom = row.chrom
            start = row.centered_start
            end   = row.centered_end
            stem  = f"{chrom}_{start}_{end}"

            print(f"  [{i+1:>4}/{len(df)}] {stem}", end="\r", flush=True)

            seq_path    = os.path.join(seq_dir,    f"{stem}_X.pt")
            target_path = os.path.join(target_dir, f"{stem}_target.pt")

            tower_paths = {
                cell_type: [
                    os.path.join(tower_dirs[cell_type][m], f"{stem}_tower_out.pt")
                    for m in range(NUM_MODELS)
                ]
                for cell_type in CELL_TYPES
            }

            # ── One-hot encode (skip if exists) ───────────────────────────────
            if not os.path.exists(seq_path):
                sequence = genome[chrom][start:end]
                X        = one_hot_encode_sequence(sequence)
                X_tensor = torch.tensor(X).to(device)
                torch.save(X_tensor.cpu(), seq_path)
            else:
                X_tensor = torch.load(seq_path, weights_only=True).to(device)
                # if X_tensor.ndim == 4:
                #     X_tensor = X_tensor.squeeze(0)
                #     torch.save(X_tensor.cpu(), seq_path)

            # ── Tower outputs per cell type and model (skip if exists) ────────
            for cell_type, models in loaded_models.items():
                for m_idx, (m, tower_path) in enumerate(
                    zip(models, tower_paths[cell_type])
                ):
                    if not os.path.exists(tower_path):
                        store_tower_output(X_tensor, m, tower_path)

            # ── Cross-cell-type target (always regenerate) ────────────────────
            target = make_cross_celltype_target(
                h1hesc_models = h1hesc_models,
                hff_models    = hff_models,
                X_tensor      = X_tensor,
                h1hesc_indices = h1hesc_indices,
                h1hesc_vector  = h1hesc_vector,
                hff_indices    = hff_indices,
                hff_vector     = hff_vector,
                device         = device,
            )
            torch.save(target, target_path)

        print(f"\n[fold {fold}] done")

    print("\nAll folds complete.")


if __name__ == "__main__":
    main()