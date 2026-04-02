"""
run_boundary_suppression_design.py

Batch boundary suppression optimisation over all successfully designed boundary
sequences. Starting from sequences that were optimised towards a strong TAD
boundary (insulation score target = -0.5), this script drives them back towards
the original unmodified genomic prediction (no boundary) using Ledidi +
Semifreddo.

CTCF motifs ± CTCF_FLANK bp are frozen in Ledidi's input_mask to preserve
the binding sites introduced during boundary design. Positions and orientations
are read from the successful_optimizations_-0.5.tsv file produced by the
boundary design run.

Output layout under SUPPRESSION_DIR/<run_name>/fold{N}/:
    {chrom}_{start}_{end}_gen_seq.pt    ← optimised central 2048 bp bin

An enriched TSV with n_edits and last_accepted_step is saved alongside each
fold's results directory.

Usage
-----
python run_boundary_suppression_design.py \\
    --folds 0 1 2 3 \\
    --run_name ctcf_flank15/lambda_0.01 \\
    --L 0.01
"""

import ast
import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath("/home1/smaruj/akita_pytorch/"))
sys.path.insert(0, "/home1/smaruj/ledidi/ledidi/")
sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))

from semifreddo.semifreddo import SemifreddoLedidiWrapper
from semifreddo.losses import LocalL1Loss
from semifreddo.optimization_loop import build_stem, run_one_design
from utils.model_utils import load_model
from helper import make_ctcf_exclusion_mask

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

DEFAULT_MODEL_PATH = (
    "/home1/smaruj/akita_pytorch/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/"
    "Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
DEFAULT_MASK_PATH        = f"{_PROJ}/optimizations/feature_masks/boundary_mask.pt"
DEFAULT_SUPPRESSION_DIR  = f"{_PROJ}/optimizations/boundary_suppression"
DEFAULT_SUCCESSFUL_TSV   = f"{_PROJ}/optimizations/boundaries/successful_optimizations_-0.5.tsv"

# ── Semifreddo / architecture constants ───────────────────────────────────────
CENTER_BIN_MAP   = 256
CONTEXT_BINS     = 5
BIN_SIZE         = 2048
CROPPING_APPLIED = 64
N_TRIU           = 130305

# ── CTCF exclusion flank ──────────────────────────────────────────────────────
CTCF_FLANK = 15

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch boundary suppression optimisation over one or more folds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--folds",    type=int, nargs="+", required=True,
                   help="One or more fold indices, e.g. --folds 0 1 2 3")
    p.add_argument("--run_name", type=str, required=True,
                   help="Results subdirectory name (e.g. 'ctcf_flank15/lambda_0.01')")
    p.add_argument("--L",   type=float, default=0.01,  help="Input-loss regularisation weight")
    p.add_argument("--tau", type=float, default=1.0,   help="Ledidi tau parameter")
    p.add_argument("--eps", type=float, default=1e-4,  help="Ledidi eps parameter")
    p.add_argument("--no_ctcf_mask", action="store_true",
                   help="If set, allow edits everywhere (no CTCF exclusion mask). "
                        "Use for control runs.")
    p.add_argument("--model_path",       default=DEFAULT_MODEL_PATH)
    p.add_argument("--mask_path",        default=DEFAULT_MASK_PATH)
    p.add_argument("--suppression_dir",  default=DEFAULT_SUPPRESSION_DIR)
    p.add_argument("--successful_tsv",   default=DEFAULT_SUCCESSFUL_TSV)
    return p.parse_args()


# ── Fold runner ───────────────────────────────────────────────────────────────

def run_fold_suppression(
    fold: int,
    args,
    model,
    output_loss: torch.nn.Module,
    device: torch.device,
) -> None:
    """Iterate over all successfully designed boundary sequences in a fold.

    Loads pre-generated starting sequences, tower outputs, and targets from
    suppression_dir, builds a per-window CTCF exclusion mask, and calls
    run_one_design for each window.

    Parameters
    ----------
    fold        : fold index
    args        : parsed CLI arguments
    model       : loaded Akita SeqNN in eval mode
    output_loss : LocalL1Loss restricted to the boundary mask indices
    device      : torch device
    """
    log.info(f"{'=' * 60}")
    log.info(f"Fold {fold}")
    log.info(f"{'=' * 60}")

    out_dir = os.path.join(args.suppression_dir, args.run_name, f"fold{fold}")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.successful_tsv, sep="\t")
    fold_df = df[df["fold"] == fold].reset_index(drop=True)
    if fold_df.empty:
        log.info(f"No rows for fold {fold}, skipping.")
        return
    log.info(f"Loaded {len(fold_df)} windows for fold {fold}")

    # Drop columns added by the boundary design analysis step to avoid
    # duplicate column names (e.g. n_edits.1) when suppression metadata
    # is concatenated below.
    cols_to_drop = [c for c in fold_df.columns if c in (
        "insul_score_diff", "optimization_success", "n_edits", "last_accepted_step"
    )]
    fold_df = fold_df.drop(columns=cols_to_drop)

    seq_dir    = os.path.join(args.suppression_dir, "initial_sequences",    f"fold{fold}")
    tower_dir  = os.path.join(args.suppression_dir, "initial_tower_outputs", f"fold{fold}")
    target_dir = os.path.join(args.suppression_dir, "targets",              f"fold{fold}")

    results = []
    for i, row in fold_df.iterrows():
        chrom = row["chrom"]
        start = int(row["centered_start"])
        end   = int(row["centered_end"])
        stem  = build_stem(chrom, start, end)
        log.info(f"  [{i + 1}/{len(fold_df)}] {stem}")

        try:
            X      = torch.load(f"{seq_dir}/{stem}_X.pt",             weights_only=True).to(device)
            tower  = torch.load(f"{tower_dir}/{stem}_tower_out.pt",   weights_only=True).to(device)
            target = torch.load(f"{target_dir}/{stem}_target.pt",     weights_only=True).to(device)

            sf_wrapper = SemifreddoLedidiWrapper(
                model                   = model,
                precomputed_full_output = tower,
                full_X                  = X,
                edited_bin              = CENTER_BIN_MAP,
                context_bins            = CONTEXT_BINS,
                cropping_applied        = CROPPING_APPLIED,
            )

            # Build per-window CTCF exclusion mask (skipped for control runs)
            ctcf_positions = ast.literal_eval(row["positions"])
            if args.no_ctcf_mask:
                input_mask = None
                log.info(f"    No CTCF mask (control run)")
            else:
                input_mask = make_ctcf_exclusion_mask(
                    ctcf_positions, flank=CTCF_FLANK, seq_len=BIN_SIZE,
                )
                log.info(f"    CTCF positions: {ctcf_positions}  |  "
                         f"Frozen bp: {input_mask.sum().item()}")

            meta = run_one_design(
                row, fold, args, sf_wrapper, output_loss, X, target, device,
                out_dir, input_mask=input_mask,
            )
        except Exception as e:
            log.error(f"  FAILED: {e}")
            meta = {"n_edits": np.nan, "last_accepted_step": np.nan}

        results.append(meta)

    df_out  = pd.concat([fold_df, pd.DataFrame(results, index=fold_df.index)], axis=1)
    out_tsv = os.path.join(
        args.suppression_dir, args.run_name,
        f"fold{fold}_suppression_opt.tsv",
    )
    df_out.to_csv(out_tsv, sep="\t", index=False)
    log.info(f"Saved enriched table → {out_tsv}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log.info(f"Device: {device}  |  Folds: {args.folds}  |  Run: {args.run_name}")
    log.info(f"Ledidi params — L={args.L}  tau={args.tau}  eps={args.eps}")
    log.info(f"CTCF flank: {CTCF_FLANK} bp  |  CTCF mask: {'disabled (control)' if args.no_ctcf_mask else 'enabled'}")

    # ── Shared resources (loaded once across all folds) ───────────────────────
    model       = load_model(args.model_path, device)
    mask        = torch.load(args.mask_path, weights_only=True).to(device)
    output_loss = LocalL1Loss(mask, n_triu=N_TRIU, reduction="sum").to(device)

    for fold in args.folds:
        run_fold_suppression(fold, args, model, output_loss, device)

    log.info("All folds complete.")


if __name__ == "__main__":
    main()