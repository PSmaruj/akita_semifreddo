"""
generate_boundaries_suppression_files.py

Pre-generates all files needed for Ledidi + Semifreddo optimisation of
boundary suppression, starting from sequences that were successfully
optimised towards strong (-0.5) boundaries.

Starting sequences are built by inserting the optimised boundary slice
(from the -0.5 boundary optimisation run) into the original genomic sequence.
Targets are model predictions of the *unmodified* original sequence.

  <ROOT_OUTPUT_DIR>/
    ohe_X/fold{N}/
      {chrom}_{start}_{end}_X.pt          ← modified starting sequence (1, 4, L)
    tower_outputs/fold{N}/
      {chrom}_{start}_{end}_tower_out.pt  ← cached conv tower activations (1, 128, 640)
    targets/fold{N}/
      {chrom}_{start}_{end}_target.pt     ← model prediction of unmodified sequence

Usage
-----
python generate_boundaries_suppression_files.py

# Process specific folds only:
python generate_boundaries_suppression_files.py --folds 0 1 2 3 4 5 6 7

Sequences and tower outputs are skipped if they already exist.
Targets are also skipped if already present; remove the existence
check in the script to force regeneration.
"""

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))

from akita.model import SeqNN
from utils.model_utils import store_tower_output

# =============================================================================
# Constants
# =============================================================================

BIN_SIZE            = 2048
CROPPING_BINS       = 64
CENTER_BIN_MAP      = 256   # editable bin index in the 512-bin contact map
CENTER_BIN_640      = CENTER_BIN_MAP + CROPPING_BINS   # = 320, index in the 640-bin tower output

MOD_SLICE_START = CENTER_BIN_640 * BIN_SIZE
MOD_SLICE_END   = (CENTER_BIN_640 + 1) * BIN_SIZE

SUCCESSFUL_OPT_TSV = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/"
    "optimizations/boundaries/successful_optimizations_-0.5.tsv"
)
SEQ_PATH_PATTERN = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/"
    "analysis/flat_regions/mouse_sequences/"
    "fold{fold}/{chrom}_{start}_{end}_X.pt"
)
GEN_SEQ_PATH_PATTERN = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/"
    "optimizations/boundaries/results/boundary_neg0p5/"
    "fold{fold}/{chrom}_{start}_{end}_gen_seq.pt"
)
MODEL_PATH_PATTERN = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/"
    "Akita_v2_mouse_Hsieh2019_mESC_model{model_idx}_finetuned.pth"
)

ROOT_OUTPUT_DIR = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/"
    "optimizations/boundary_suppression"
)

# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate starting sequences, cached tower activations, and targets "
            "for boundary suppression optimisation. "
            "Starting sequences are built by inserting the optimised boundary slice "
            "into the original genomic sequence. "
            "Targets are model predictions of the unmodified original sequence."
        )
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=None,
        help=(
            "Fold indices to process. Defaults to all folds present in the TSV."
        ),
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

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = MODEL_PATH_PATTERN.format(model_idx=args.model_idx)
    print(f"Loading model: {model_path}")
    model = SeqNN()
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()

    # ── Load successful optimizations TSV ─────────────────────────────────────
    df = pd.read_csv(SUCCESSFUL_OPT_TSV, sep="\t")
    print(f"Loaded {len(df)} rows from successful_optimizations_-0.5.tsv")

    folds = args.folds if args.folds is not None else sorted(df["fold"].unique())
    print(f"Processing folds: {folds}")

    # ── Process each fold ─────────────────────────────────────────────────────
    for fold in folds:
        fold_df = df[df["fold"] == fold].reset_index(drop=True)
        if fold_df.empty:
            print(f"\n[fold {fold}] No rows found, skipping.")
            continue

        print(f"\n[fold {fold}]  {len(fold_df)} sequences")

        seq_dir    = os.path.join(ROOT_OUTPUT_DIR, "initial_sequences",         f"fold{fold}")
        tower_dir  = os.path.join(ROOT_OUTPUT_DIR, "initial_tower_outputs", f"fold{fold}")
        target_dir = os.path.join(ROOT_OUTPUT_DIR, "targets",       f"fold{fold}")
        for d in (seq_dir, tower_dir, target_dir):
            os.makedirs(d, exist_ok=True)

        for i, row in enumerate(fold_df.itertuples(index=False)):
            chrom = row.chrom
            start = row.centered_start
            end   = row.centered_end
            stem  = f"{chrom}_{start}_{end}"

            print(f"  [{i+1:>4}/{len(fold_df)}] {stem}", end="\r", flush=True)

            seq_path    = os.path.join(seq_dir,    f"{stem}_X.pt")
            tower_path  = os.path.join(tower_dir,  f"{stem}_tower_out.pt")
            target_path = os.path.join(target_dir, f"{stem}_target.pt")

            # ── Load original one-hot sequence ────────────────────────────────
            src_seq_path = SEQ_PATH_PATTERN.format(
                fold=fold, chrom=chrom, start=start, end=end
            )
            X_tensor = torch.load(
                src_seq_path, map_location=device, weights_only=True
            )   # (1, 4, L) or (4, L)

            # ── Target: model prediction of the unmodified sequence ───────────
            if not os.path.exists(target_path):
                with torch.no_grad():
                    y = model(X_tensor)
                torch.save(y.cpu(), target_path)

            # ── Build X_mod: replace centre bin with optimised slice ──────────
            gen_seq_path = GEN_SEQ_PATH_PATTERN.format(
                fold=fold, chrom=chrom, start=start, end=end
            )
            mod_slice = torch.load(
                gen_seq_path, map_location=device, weights_only=True
            )   # (1, 4, BIN_SIZE) or (4, BIN_SIZE)

            X_mod = X_tensor.clone()
            X_mod[..., MOD_SLICE_START:MOD_SLICE_END] = mod_slice

            # ── Save starting sequence and tower output ───────────────────────
            if not os.path.exists(seq_path):
                torch.save(X_mod.cpu(), seq_path)

            if not os.path.exists(tower_path):
                store_tower_output(X_mod, model, tower_path)

        print(f"\n[fold {fold}] done")

    print("\nAll folds complete.")


if __name__ == "__main__":
    main()