"""
run_boundary_no_ctcf_design.py

Batch boundary optimisation over all genomic windows in a fold's flat-regions table.
Saves per-window generated sequences and an enriched TSV with optimisation metadata.

Usage:
    python run_boundary_no_ctcf_design.py \\
        --folds 0 1 2 3 \\
        --run_name results/gamma_3000 \\
        --boundary_strength -0.2 \\
        --L 0.01 \\
        --gamma 3000
"""

import os
import sys
import argparse
import logging

import torch

sys.path.append(os.path.abspath("/home1/smaruj/akita_pytorch/"))
sys.path.insert(0, "/home1/smaruj/ledidi/ledidi/")
sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))

from semifreddo.semifreddo import SemifreddoLedidiWrapper, CTCFAwareSemifreddoWrapper
from semifreddo.losses import LocalL1Loss, LocalL1LossWithCTCFPenalty
from semifreddo.optimization_loop import strength_tag, build_stem, run_one_design, run_fold
from utils.model_utils import load_model
from utils.fimo_utils import read_meme_pwm

# ── Default paths ─────────────────────────────────────────────────────────────
_PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

DEFAULT_MODEL_PATH = (
    "/home1/smaruj/akita_pytorch/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/"
    "Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
DEFAULT_SEQ_BASE_DIR      = f"{_PROJ}/analysis/flat_regions"
DEFAULT_TARGET_BASE_DIR   = f"{_PROJ}/optimizations/boundaries/targets"
DEFAULT_MASK_PATH         = f"{_PROJ}/optimizations/feature_masks/boundary_mask.pt"
DEFAULT_RESULTS_BASE_DIR  = f"{_PROJ}/optimizations/boundaries_no_ctcf"
DEFAULT_FLAT_REGIONS_BASE = f"{_PROJ}/analysis/flat_regions/mouse_flat_regions_chrom_states_tsv"

CTCF_PWM_PATH = "/home1/smaruj/akita_semifreddo/data/pwm/MA0139.1.meme"

# ── Semifreddo / architecture constants ───────────────────────────────────────
CENTER_BIN_MAP   = 256
CONTEXT_BINS     = 5
BIN_SIZE         = 2048
CROPPING_APPLIED = 64
N_TRIU           = 130305

# Edited slice is relative to X_center (shape (1, 4, 2048) — just the central bin),
# so it always spans the full extent of what Ledidi is optimising.
EDITED_SLICE = slice(0, BIN_SIZE)

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
                   help="Results subdirectory name (e.g. 'results/gamma_3000')")
    p.add_argument("--boundary_strength", type=float, required=True,
                   help="Value applied to the off-diagonal quadrants of the boundary mask "
                        "(e.g. -0.2). Negative values suppress contacts.")
    p.add_argument("--gamma", type=float, default=3000, help="CTCF penalty weight (gamma)")
    p.add_argument("--L",     type=float, default=0.01,  help="Input-loss regularisation weight")
    p.add_argument("--tau",   type=float, default=1.0,   help="Ledidi tau parameter")
    p.add_argument("--eps",   type=float, default=1e-4,  help="Ledidi eps parameter")
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

    log.info(f"Device: {device}  |  Folds: {args.folds}  |  Run: {args.run_name}")
    log.info(f"Boundary strength: {args.boundary_strength} (tag: {tag})")
    log.info(f"Ledidi params — L={args.L}  tau={args.tau}  eps={args.eps}")
    log.info(f"CTCF penalty — gamma={args.gamma}")

    # ── Shared resources (loaded once across all folds) ───────────────────────
    model = load_model(args.model_path, device)
    mask  = torch.load(args.mask_path, weights_only=True).to(device)

    # CTCF PWM — read_meme_pwm returns (4, 19) directly, no .T needed
    pwm_CTCF    = read_meme_pwm(CTCF_PWM_PATH)
    motifs_dict = {"CTCF": pwm_CTCF}

    # Base structural loss — shared across all windows
    output_loss = LocalL1Loss(mask, n_triu=N_TRIU, reduction="sum").to(device)

    # ── Boundary-specific run_one closure ─────────────────────────────────────
    def run_one_fn(row, fold, args, out_dir):
        stem = build_stem(row["chrom"], int(row["centered_start"]), int(row["centered_end"]))
        log.info(f"  Window: {stem}")
        X      = torch.load(f"{args.seq_base_dir}/mouse_sequences/fold{fold}/{stem}_X.pt",
                            weights_only=True).to(device)
        tower  = torch.load(f"{args.seq_base_dir}/mouse_tower_outputs/fold{fold}/{stem}_tower_out.pt",
                            weights_only=True).to(device)
        target = torch.load(f"{args.target_base_dir}/boundary_{tag}/fold{fold}/{stem}_target.pt",
                            weights_only=True).to(device)
        
        sf_wrapper = SemifreddoLedidiWrapper(
            model=model, precomputed_full_output=tower, full_X=X,
            edited_bin=CENTER_BIN_MAP, context_bins=CONTEXT_BINS,
            cropping_applied=CROPPING_APPLIED,
        )
        sf_ctcf_wrapper = CTCFAwareSemifreddoWrapper(sf_wrapper)

        ctcf_penalty_output_loss = LocalL1LossWithCTCFPenalty(
            local_loss_fn = output_loss,
            motifs_dict   = motifs_dict,
            edited_slice  = slice(0, BIN_SIZE),
            gamma         = args.gamma,
            seq_wrapper   = sf_ctcf_wrapper,
        ).to(device)

        print(type(sf_ctcf_wrapper))
        print(type(ctcf_penalty_output_loss.seq_wrapper))
        
        return run_one_design(row, fold, args, sf_ctcf_wrapper, ctcf_penalty_output_loss,
                              X, target, device, out_dir)

    # ── Run all folds ─────────────────────────────────────────────────────────
    for fold in args.folds:
        run_fold(fold, args, run_one_fn, args.flat_regions_base, args.results_base_dir)

    log.info("All folds complete.")


if __name__ == "__main__":
    main()