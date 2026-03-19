"""
run_boundary_design.py

Batch boundary optimisation over all genomic windows in a fold's flat-regions table,
or over a specific subset of windows supplied via --retry_tsv.

Usage (full run):
    python run_boundary_design.py \\
        --folds 0 1 2 3 \\
        --run_name lambda/lambda_0.01 \\
        --boundary_strength -0.5 \\
        --L 0.01

Usage (retry failed windows):
    python run_boundary_design.py \\
        --folds 0 1 2 3 \\
        --run_name lambda/lambda_0.01_retry \\
        --boundary_strength -0.5 \\
        --L 0.01 \\
        --retry_tsv unsuccessful_all_folds_-0.5.tsv
"""

import os
import sys
import argparse
import logging

import pandas as pd
import torch

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, "/home1/smaruj/ledidi/ledidi/")
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))

from semifreddo.semifreddo import SemifreddoLedidiWrapper
from utils.losses_utils import LocalL1Loss
from utils.optimization_utils import strength_tag, build_stem
from utils.model_utils import load_model
from utils.optimization_loop_utils import run_one_design, run_fold

# ── Default paths ─────────────────────────────────────────────────────────────
_PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

DEFAULT_MODEL_PATH = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/"
    "Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
DEFAULT_SEQ_BASE_DIR      = f"{_PROJ}/analysis/flat_regions"
DEFAULT_TARGET_BASE_DIR   = f"{_PROJ}/optimizations/boundaries/targets"
DEFAULT_MASK_PATH         = f"{_PROJ}/optimizations/feature_masks/boundary_mask.pt"
DEFAULT_RESULTS_BASE_DIR  = f"{_PROJ}/optimizations/boundaries"
DEFAULT_FLAT_REGIONS_BASE = f"{_PROJ}/analysis/flat_regions/mouse_flat_regions_chrom_states_tsv"

# ── Semifreddo / architecture constants ───────────────────────────────────────
CENTER_BIN_MAP   = 256
CONTEXT_BINS     = 5
BIN_SIZE         = 2048
CROPPING_APPLIED = 64
N_TRIU           = 130305

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
        description="Batch boundary optimisation over one or more folds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--folds",    type=int, nargs="+", required=True,
                   help="One or more fold indices, e.g. --folds 0 1 2 3")
    p.add_argument("--run_name", type=str, required=True,
                   help="Results subdirectory name (e.g. 'lambda/lambda_0.01')")
    p.add_argument("--boundary_strength", type=float, required=True,
                   help="Value applied to the off-diagonal quadrants of the boundary mask "
                        "(e.g. -0.5). Negative values suppress contacts.")
    p.add_argument("--L",   type=float, default=0.01,  help="Input-loss regularisation weight")
    p.add_argument("--tau", type=float, default=1.0,   help="Ledidi tau parameter")
    p.add_argument("--eps", type=float, default=1e-4,  help="Ledidi eps parameter")
    p.add_argument("--retry_tsv", type=str, default=None,
                   help="Path to a TSV of unsuccessful windows to retry. Must contain "
                        "chrom, centered_start, centered_end, and fold columns. "
                        "When provided, only these windows are processed.")
    p.add_argument("--model_path",        default=DEFAULT_MODEL_PATH)
    p.add_argument("--seq_base_dir",      default=DEFAULT_SEQ_BASE_DIR)
    p.add_argument("--target_base_dir",   default=DEFAULT_TARGET_BASE_DIR)
    p.add_argument("--mask_path",         default=DEFAULT_MASK_PATH)
    p.add_argument("--results_base_dir",  default=DEFAULT_RESULTS_BASE_DIR)
    p.add_argument("--flat_regions_base", default=DEFAULT_FLAT_REGIONS_BASE)
    return p.parse_args()


# ── Retry helpers ─────────────────────────────────────────────────────────────

def load_retry_windows(tsv_path: str) -> pd.DataFrame:
    """Load and validate the retry TSV."""
    df = pd.read_csv(tsv_path, sep="\t")
    required = {"chrom", "centered_start", "centered_end", "fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"retry_tsv is missing columns: {missing}")
    log.info(f"Retry mode: loaded {len(df)} windows from {tsv_path} "
             f"across folds {sorted(df['fold'].unique().tolist())}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    tag    = strength_tag(args.boundary_strength)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log.info(f"Device: {device}  |  Folds: {args.folds}  |  Run: {args.run_name}")
    log.info(f"Boundary strength: {args.boundary_strength} (tag: {tag})")
    log.info(f"Ledidi params — L={args.L}  tau={args.tau}  eps={args.eps}")

    # ── Shared resources (loaded once across all folds) ───────────────────────
    model       = load_model(args.model_path, device)
    mask        = torch.load(args.mask_path, weights_only=True).to(device)
    output_loss = LocalL1Loss(mask, n_triu=N_TRIU, reduction="sum").to(device)

    # ── Retry TSV (optional) ──────────────────────────────────────────────────
    retry_df = load_retry_windows(args.retry_tsv) if args.retry_tsv else None

    # ── Boundary-specific run_one closure ─────────────────────────────────────
    def run_one_fn(row, fold, args, out_dir):
        stem = build_stem(row["chrom"], int(row["centered_start"]), int(row["centered_end"]))
        log.info(f"  Window: {stem}")
        X      = torch.load(f"{args.seq_base_dir}/mouse_sequences/fold{fold}/{stem}_X.pt", weights_only=True).to(device)
        tower  = torch.load(f"{args.seq_base_dir}/mouse_tower_outputs/fold{fold}/{stem}_tower_out.pt", weights_only=True).to(device)
        target = torch.load(f"{args.target_base_dir}/boundary_{tag}/fold{fold}/{stem}_target.pt", weights_only=True).to(device)
        sf_wrapper = SemifreddoLedidiWrapper(
            model=model, precomputed_full_output=tower, full_X=X,
            edited_bin=CENTER_BIN_MAP, context_bins=CONTEXT_BINS,
            cropping_applied=CROPPING_APPLIED,
        )
        return run_one_design(row, fold, args, sf_wrapper, output_loss, X, target, device, out_dir)

    # ── Run all folds ─────────────────────────────────────────────────────────
    for fold in args.folds:
        if retry_df is not None:
            fold_windows = retry_df[retry_df["fold"] == fold]
            if fold_windows.empty:
                log.info(f"Fold {fold}: no retry windows, skipping.")
                continue
            log.info(f"Fold {fold}: retrying {len(fold_windows)} windows.")
            out_dir = os.path.join(args.results_base_dir, args.run_name, f"fold{fold}")
            os.makedirs(out_dir, exist_ok=True)
            for _, row in fold_windows.iterrows():
                run_one_fn(row, fold, args, out_dir)
        else:
            run_fold(fold, args, run_one_fn, args.flat_regions_base, args.results_base_dir)

    log.info("All folds complete.")


if __name__ == "__main__":
    main()