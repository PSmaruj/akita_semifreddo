"""
run_boundary_design.py

Batch boundary optimisation over all genomic windows in a fold's flat-regions table.
Saves per-window generated sequences and an enriched TSV with optimisation metadata.

Usage:
    python run_boundary_design.py \
        --folds 0 1 2 3 \
        --seeds 0 \
        --run_name indep_runs_lambda_125.0/seed0 \
        --boundary_strength -0.5 \
        --L 125.0 \
"""

import os
import sys
import argparse
import logging

import torch

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, "/home1/smaruj/ledidi/ledidi/")
sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))

from semifreddo.semifreddo import SemifreddoLedidiWrapper
from semifreddo.losses import LocalL1Loss
from semifreddo.optimization_loop import strength_tag, build_stem, run_one_design, run_fold
from utils.model_utils import load_model

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
    p.add_argument("--seeds",    type=int, nargs="+", default=[0],
                   help="One or more random seeds, e.g. --seeds 0 1 2 3 4 5 6 7 8 9")
    p.add_argument("--run_name", type=str, required=True,
                   help="Results subdirectory name (e.g. 'lambda/lambda_0.01'). "
                        "When multiple seeds are used, each seed's outputs are written "
                        "to '<run_name>/seed<seed>/'.")
    p.add_argument("--boundary_strength", type=float, required=True,
                   help="Value applied to the off-diagonal quadrants of the boundary mask "
                        "(e.g. -0.5). Negative values suppress contacts.")
    p.add_argument("--L",   type=float, default=0.01,  help="Input-loss regularisation weight")
    p.add_argument("--tau", type=float, default=1.0,   help="Ledidi tau parameter")
    p.add_argument("--eps", type=float, default=1e-4,  help="Ledidi eps parameter")
    p.add_argument("--model_path",        default=DEFAULT_MODEL_PATH)
    p.add_argument("--seq_base_dir",      default=DEFAULT_SEQ_BASE_DIR)
    p.add_argument("--target_base_dir",   default=DEFAULT_TARGET_BASE_DIR)
    p.add_argument("--mask_path",         default=DEFAULT_MASK_PATH)
    p.add_argument("--results_base_dir",  default=DEFAULT_RESULTS_BASE_DIR)
    p.add_argument("--flat_regions_base", default=DEFAULT_FLAT_REGIONS_BASE)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    tag    = strength_tag(args.boundary_strength)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log.info(f"Device: {device}  |  Folds: {args.folds}  |  Seeds: {args.seeds}  |  Run: {args.run_name}")
    log.info(f"Boundary strength: {args.boundary_strength} (tag: {tag})")
    log.info(f"Ledidi params — L={args.L}  tau={args.tau}  eps={args.eps}")

    # ── Shared resources (loaded once across all folds / seeds) ───────────────
    model       = load_model(args.model_path, device)
    mask        = torch.load(args.mask_path, weights_only=True).to(device)
    output_loss = LocalL1Loss(mask, n_triu=N_TRIU, reduction="sum").to(device)

    # ── Outer loop: seeds ─────────────────────────────────────────────────────
    for seed in args.seeds:
        torch.manual_seed(seed)
        log.info(f"=== Seed {seed} ===")

        # Nest seed outputs under <results_base_dir>/<run_name>/
        # seed_results_dir = os.path.join(args.results_base_dir, args.run_name)

        # ── Boundary-specific run_one closure (captures current seed) ─────────
        def run_one_fn(row, fold, args, out_dir, _seed=seed):
            stem = build_stem(row["chrom"], int(row["centered_start"]), int(row["centered_end"]))
            log.info(f"  Window: {stem}  seed={_seed}")
            X      = torch.load(f"{args.seq_base_dir}/mouse_sequences/fold{fold}/{stem}_X.pt", weights_only=True).to(device)
            tower  = torch.load(f"{args.seq_base_dir}/mouse_tower_outputs/fold{fold}/{stem}_tower_out.pt", weights_only=True).to(device)
            target = torch.load(f"{args.target_base_dir}/boundary_{tag}/fold{fold}/{stem}_target.pt", weights_only=True).to(device)
            sf_wrapper = SemifreddoLedidiWrapper(
                model=model, precomputed_full_output=tower, full_X=X,
                edited_bin=CENTER_BIN_MAP, context_bins=CONTEXT_BINS,
                cropping_applied=CROPPING_APPLIED,
            )
            return run_one_design(row, fold, args, sf_wrapper, output_loss, X, target, device, out_dir)

        # ── Run all folds for this seed ───────────────────────────────────────
        for fold in args.folds:
            run_fold(fold, args, run_one_fn, args.flat_regions_base, args.results_base_dir)

    log.info("All seeds and folds complete.")


if __name__ == "__main__":
    main()