"""
run_cross_celltype_boundary_design.py

Batch cross-cell-type boundary optimisation over all genomic windows in a
fold's flat-regions table. Uses two H1hESC and two HFF Akita v2 models
simultaneously to design sequences folding into strong boundaries in one
cell type and weak boundaries in the other.

Saves per-window generated sequences and an enriched TSV with optimisation
metadata.

Usage:
    python run_cross_celltype_boundary_design.py \
        --folds 0 1 2 3 \
        --run_name lambda/lambda_125 \
        --strong_cell_type H1hESC \
        --strong_strength -0.5 \
        --weak_strength -0.2 \
"""

import os
import sys
import argparse
import logging

import pandas as pd
import torch
import torch.nn as nn

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, "/home1/smaruj/ledidi/ledidi/")
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))

from akita.model import SeqNN
from ledidi import Ledidi
from semifreddo.semifreddo import SemifreddoLedidiWrapper, StackingDesignWrapper
from semifreddo.optimization_loop import strength_tag, build_stem, count_edits, last_accepted_step, run_fold
from semifreddo.losses import LocalL1Loss

# ── Default paths ─────────────────────────────────────────────────────────────
_PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

MODEL_PATH_PATTERN = (
    "/home1/smaruj/pytorch_akita/models/finetuned/human/"
    "Krietenstein2019_{cell_type}/checkpoints/"
    "Akita_v2_human_Krietenstein2019_{cell_type}_model{model_idx}_finetuned.pth"
)
DEFAULT_SEQ_BASE_DIR      = f"{_PROJ}/analysis/flat_regions"
DEFAULT_TARGET_BASE_DIR   = f"{_PROJ}/optimizations/cross_celltype_boundaries/targets"
DEFAULT_MASK_PATH         = f"{_PROJ}/optimizations/feature_masks/boundary_mask.pt"
DEFAULT_RESULTS_BASE_DIR  = f"{_PROJ}/optimizations/cross_celltype_boundaries"
DEFAULT_FLAT_REGIONS_BASE = f"{_PROJ}/analysis/flat_regions/human_flat_regions_tsv"

# ── Architecture constants ────────────────────────────────────────────────────
CENTER_BIN_MAP   = 256
CONTEXT_BINS     = 5
CROPPING_APPLIED = 64
N_TRIU           = 130305
NUM_MODELS       = 2        # per cell type
CELL_TYPES       = ["H1hESC", "HFF"]

# ── Ledidi constants ──────────────────────────────────────────────────────────
MAX_ITER       = 2000
EARLY_STOPPING = 2000

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Single-window optimisation ────────────────────────────────────────────────

def run_one_cross_celltype(
    row: pd.Series,
    fold: int,
    args,
    models: list,
    output_loss,
    device: torch.device,
    out_dir: str,
) -> dict:
    """Run Ledidi cross-cell-type boundary optimisation for a single window.

    Parameters
    ----------
    row         : pd.Series from the flat-regions TSV
    fold        : fold index
    args        : argparse.Namespace — must carry .L, .tau, .eps,
                  .strong_cell_type, .strong_strength, .weak_strength,
                  .seq_base_dir
    models      : list of 4 SeqNN instances in slot order
                  [strong_ct model 0, strong_ct model 1,
                   weak_ct  model 0, weak_ct  model 1]
    output_loss : callable — summed LocalL1Loss across all 4 heads
    device      : torch.device
    out_dir     : directory where <stem>_gen_seq.pt is saved

    Returns
    -------
    dict with keys 'n_edits' and 'last_accepted_step'
    """
    chrom = row["chrom"]
    start = int(row["centered_start"])
    end   = int(row["centered_end"])
    stem  = build_stem(chrom, start, end)
    log.info(f"  Window: {stem}")

    weak_cell_type = [ct for ct in CELL_TYPES if ct != args.strong_cell_type][0]
    cell_types_ordered = [args.strong_cell_type, weak_cell_type]

    # ── Load sequence, towers, target ─────────────────────────────────────────
    X = torch.load(
        f"{args.seq_base_dir}/human_sequences/fold{fold}/{stem}_X.pt",
        weights_only=True,
    ).to(device)

    towers = [
        torch.load(
            f"{args.seq_base_dir}/human_tower_outputs/{ct}/model{m}/fold{fold}/{stem}_tower_out.pt",
            weights_only=True,
        ).to(device)
        for ct in cell_types_ordered for m in range(NUM_MODELS)
    ]

    strong_tag = strength_tag(args.strong_strength)
    weak_tag   = strength_tag(args.weak_strength)
    target_tag = f"{args.strong_cell_type}_strong_{strong_tag}_{weak_cell_type}_weak_{weak_tag}"

    target = torch.load(
        f"{args.target_base_dir}/{target_tag}/fold{fold}/{stem}_target.pt",
        weights_only=True,
    ).to(device)

    # ── Build Semifreddo wrappers + combined model ─────────────────────────────
    sf_wrappers = [
        SemifreddoLedidiWrapper(
            model                   = m,
            precomputed_full_output = tower,
            full_X                  = X,
            edited_bin              = CENTER_BIN_MAP,
            context_bins            = CONTEXT_BINS,
            cropping_applied        = CROPPING_APPLIED,
        )
        for m, tower in zip(models, towers)
    ]
    combined_model = StackingDesignWrapper(sf_wrappers).to(device)

    sf0      = sf_wrappers[0]
    X_center = X[:, :, sf0.center_bp_start:sf0.center_bp_end]   # (1, 4, 2048)

    # ── Run Ledidi ─────────────────────────────────────────────────────────────
    ledidi_optimizer = Ledidi(
        combined_model,
        shape               = X_center.shape[1:],
        input_loss          = torch.nn.L1Loss(reduction="sum"),
        output_loss         = output_loss,
        batch_size          = 1,
        l                   = args.L,
        tau                 = args.tau,
        eps                 = args.eps,
        max_iter            = MAX_ITER,
        early_stopping_iter = EARLY_STOPPING,
        return_history      = True,
        verbose             = False,
    ).cuda()

    generated_seq, history = ledidi_optimizer.fit_transform(X_center, target)

    # ── Count edits on full sequence ───────────────────────────────────────────
    full_generated_seq = X.clone()
    full_generated_seq[:, :, sf0.center_bp_start:sf0.center_bp_end] = generated_seq

    n_edits   = count_edits(X, full_generated_seq)
    last_step = last_accepted_step(history)
    log.info(f"    Edits: {n_edits}  |  Last accepted step: {last_step}")

    torch.save(generated_seq.cpu(), os.path.join(out_dir, f"{stem}_gen_seq.pt"))
    log.info(f"    Saved → {os.path.join(out_dir, f'{stem}_gen_seq.pt')}")

    return {"n_edits": n_edits, "last_accepted_step": last_step}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch cross-cell-type boundary optimisation over one or more folds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--folds",    type=int, nargs="+", required=True)
    p.add_argument("--run_name", type=str, required=True,
                   help="Results subdirectory (e.g. 'lambda/lambda_125')")
    p.add_argument("--strong_cell_type", choices=CELL_TYPES, required=True,
                   help="Cell type receiving the strong boundary target.")
    p.add_argument("--strong_strength", type=float, default=-0.5)
    p.add_argument("--weak_strength",   type=float, default=-0.2)
    p.add_argument("--L",   type=float, default=0.01)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--seq_base_dir",      default=DEFAULT_SEQ_BASE_DIR)
    p.add_argument("--target_base_dir",   default=DEFAULT_TARGET_BASE_DIR)
    p.add_argument("--mask_path",         default=DEFAULT_MASK_PATH)
    p.add_argument("--results_base_dir",  default=DEFAULT_RESULTS_BASE_DIR)
    p.add_argument("--flat_regions_base", default=DEFAULT_FLAT_REGIONS_BASE)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weak_cell_type     = [ct for ct in CELL_TYPES if ct != args.strong_cell_type][0]
    cell_types_ordered = [args.strong_cell_type, weak_cell_type]

    strong_tag = strength_tag(args.strong_strength)
    weak_tag   = strength_tag(args.weak_strength)
    target_tag = f"{args.strong_cell_type}_strong_{strong_tag}_{weak_cell_type}_weak_{weak_tag}"

    log.info(f"Device: {device}  |  Folds: {args.folds}  |  Run: {args.run_name}")
    log.info(f"Strong boundary ({args.strong_strength}) → {args.strong_cell_type}")
    log.info(f"Weak   boundary ({args.weak_strength})  → {weak_cell_type}")
    log.info(f"Target tag: {target_tag}")
    log.info(f"Ledidi params — L={args.L}  tau={args.tau}  eps={args.eps}")

    # ── Load all 4 models once ────────────────────────────────────────────────
    models = []
    for ct in cell_types_ordered:
        for model_idx in range(NUM_MODELS):
            m = SeqNN()
            m.load_state_dict(torch.load(
                MODEL_PATH_PATTERN.format(cell_type=ct, model_idx=model_idx),
                map_location=device, weights_only=True,
            ))
            m.to(device).eval()
            models.append(m)
            log.info(f"{ct} model {model_idx} loaded")

    # ── Build shared loss ─────────────────────────────────────────────────────
    boundary_mask = torch.load(args.mask_path, weights_only=True).to(device)
    single_loss   = LocalL1Loss(boundary_mask, n_triu=N_TRIU, reduction="sum").to(device)

    output_loss = lambda y_hat, y_bar: sum(
        single_loss(y_hat[:, i:i+1, :], y_bar[:, i:i+1, :])
        for i in range(4)
    )

    # ── Fold closure ──────────────────────────────────────────────────────────
    def run_one_fn(row, fold, args, out_dir):
        return run_one_cross_celltype(
            row, fold, args, models, output_loss, device, out_dir
        )

    # ── Run all folds ─────────────────────────────────────────────────────────
    for fold in args.folds:
        run_fold(
            fold, args, run_one_fn,
            args.flat_regions_base, args.results_base_dir,
            tsv_suffix="fold{fold}_selected_genomic_windows_centered.tsv",
        )

    log.info("All folds complete.")


if __name__ == "__main__":
    main()