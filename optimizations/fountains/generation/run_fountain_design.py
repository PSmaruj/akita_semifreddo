"""
run_fountain_design.py

Batch fountain optimisation over all genomic windows in a fold's flat-regions
table, using all 4 Akita ensemble models simultaneously via StackingDesignWrapper
and MultiBinSemifreddoLedidiWrapper.

Saves per-window generated sequences and an enriched TSV with optimisation metadata.

Usage:
    python run_fountain_design.py \
        --folds 0 1 2 3 \
        --run_name results \
        --fountain_strength 0.5
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
from semifreddo.semifreddo import MultiBinSemifreddoLedidiWrapper, StackingDesignWrapper
from semifreddo.losses import MultiHeadLocalL1Loss
from semifreddo.optimization_loop import strength_tag, build_stem, count_edits, last_accepted_step, run_fold

# ── Default paths ─────────────────────────────────────────────────────────────
_PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

MODEL_PATH_PATTERN    = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC/checkpoints/"
    "Akita_v2_mouse_Hsieh2019_mESC_model{model_idx}_finetuned.pth"
)
MASK_DIR = (
    "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/"
    "optimizations/feature_masks"
)
DEFAULT_SEQ_BASE_DIR = f"{_PROJ}/analysis/flat_regions"
DEFAULT_FOUNTAIN_DIR      = f"{_PROJ}/optimizations/fountains"
DEFAULT_RESULTS_BASE_DIR  = f"{_PROJ}/optimizations/fountains"
DEFAULT_FLAT_REGIONS_BASE = f"{_PROJ}/analysis/flat_regions/mouse_flat_regions_chrom_states_tsv"

# ── Semifreddo / architecture constants ───────────────────────────────────────
CENTER_BIN_MAP   = 256
N_EDIT_BINS      = 50
CONTEXT_BINS     = 5
SPLICE_BUFFER    = 2
CROPPING_APPLIED = 64
N_TRIU           = 130305
NUM_MODELS       = 4

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

def run_one_fountain(
    row: pd.Series,
    fold: int,
    args,
    models: list,
    output_loss: nn.Module,
    device: torch.device,
    out_dir: str,
) -> dict:
    """Run Ledidi fountain optimisation for a single genomic window.

    Parameters
    ----------
    row         : pd.Series from the flat-regions TSV
    fold        : fold index (for logging and path construction)
    args        : argparse.Namespace — must carry .L, .tau, .eps,
                  .fountain_strength, .seq_base_dir, .fountain_dir
    models      : list of 4 SeqNN instances (eval mode, on device)
    output_loss : pre-built MultiHeadLocalL1Loss (callable)
    device      : torch.device
    out_dir     : directory where <stem>_gen_seq.pt is saved

    Returns
    -------
    dict with keys 'n_edits' and 'last_accepted_step'
    """
    tag   = strength_tag(args.fountain_strength)
    chrom = row["chrom"]
    start = int(row["centered_start"])
    end   = int(row["centered_end"])
    stem  = build_stem(chrom, start, end)

    # ── Load sequence, towers, target ─────────────────────────────────────────
    X = torch.load(
        f"{args.seq_base_dir}/mouse_sequences/fold{fold}/{stem}_X.pt",
        weights_only=True,
    ).to(device)

    towers = [
        torch.load(
            f"{args.seq_base_dir}/mouse_tower_outputs/model{m}/fold{fold}/{stem}_tower_out.pt",
            weights_only=True,
        ).to(device)
        for m in range(NUM_MODELS)
    ]

    target = torch.load(
        f"{args.fountain_dir}/targets/fountain_{tag}/fold{fold}/{stem}_target.pt",
        weights_only=True,
    ).to(device)

    # ── Build per-model Semifreddo wrappers + combined model ──────────────────
    sf_wrappers = [
        MultiBinSemifreddoLedidiWrapper(
            model                   = m,
            precomputed_full_output = tower,
            full_X                  = X,
            center_bin              = CENTER_BIN_MAP,
            n_edit_bins             = N_EDIT_BINS,
            context_bins            = CONTEXT_BINS,
            splice_buffer           = SPLICE_BUFFER,
            cropping_applied        = CROPPING_APPLIED,
        )
        for m, tower in zip(models, towers)
    ]
    combined_model = StackingDesignWrapper(sf_wrappers).to(device)

    # Use edit coordinates from wrapper 0 (identical across all wrappers)
    sf0    = sf_wrappers[0]
    X_edit = X[:, :, sf0.edit_bp_start:sf0.edit_bp_end]   # (1, 4, N_EDIT_BINS*BIN_SIZE)

    # ── Run Ledidi ─────────────────────────────────────────────────────────────
    ledidi_optimizer = Ledidi(
        combined_model,
        shape               = X_edit.shape[1:],
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

    generated_seq, history = ledidi_optimizer.fit_transform(X_edit, target)

    # ── Count edits on the full sequence ──────────────────────────────────────
    full_generated_seq = X.clone()
    full_generated_seq[:, :, sf0.edit_bp_start:sf0.edit_bp_end] = generated_seq

    n_edits   = count_edits(X, full_generated_seq)
    last_step = last_accepted_step(history)
    log.info(f"    {stem}  |  Edits: {n_edits}  |  Last accepted step: {last_step}")

    torch.save(generated_seq.cpu(), os.path.join(out_dir, f"{stem}_gen_seq.pt"))
    log.info(f"    Saved → {os.path.join(out_dir, f'{stem}_gen_seq.pt')}")

    return {"n_edits": n_edits, "last_accepted_step": last_step}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch fountain optimisation over one or more folds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--folds",    type=int, nargs="+", required=True,
                   help="One or more fold indices, e.g. --folds 0 1 2 3")
    p.add_argument("--run_name", type=str, required=True,
                   help="Results subdirectory name (e.g. 'lambda/lambda_125')")
    p.add_argument("--fountain_strength", type=float, required=True,
                   help="Value applied inside the antidiagonal cone (e.g. 0.5).")
    p.add_argument("--L",   type=float, default=0.01, help="Input-loss regularisation weight")
    p.add_argument("--tau", type=float, default=1.0,   help="Ledidi tau parameter")
    p.add_argument("--eps", type=float, default=1e-4,  help="Ledidi eps parameter")
    p.add_argument("--seq_base_dir",      default=DEFAULT_SEQ_BASE_DIR)
    p.add_argument("--fountain_dir",      default=DEFAULT_FOUNTAIN_DIR)
    p.add_argument("--results_base_dir",  default=DEFAULT_RESULTS_BASE_DIR)
    p.add_argument("--flat_regions_base", default=DEFAULT_FLAT_REGIONS_BASE)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    tag    = strength_tag(args.fountain_strength)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log.info(f"Device: {device}  |  Folds: {args.folds}  |  Run: {args.run_name}")
    log.info(f"Fountain strength: {args.fountain_strength} (tag: {tag})")
    log.info(f"Ledidi params — L={args.L}  tau={args.tau}  eps={args.eps}")

    # ── Load all 4 models once (shared across all folds and windows) ──────────
    models = []
    for model_idx in range(NUM_MODELS):
        m = SeqNN()
        m.load_state_dict(torch.load(
            MODEL_PATH_PATTERN.format(model_idx=model_idx),
            map_location=device, weights_only=True,
        ))
        m.to(device).eval()
        models.append(m)
        log.info(f"Model {model_idx} loaded")

    # ── Build output loss (shared across all windows) ─────────────────────────
    mask_path    = f"{MASK_DIR}/fountain_pos0p5_mask.pt"
    fountain_mask = torch.load(mask_path, weights_only=True).to(device)
    
    multi_loss = MultiHeadLocalL1Loss(
        mask      = fountain_mask,
        n_triu    = N_TRIU,
        n_models  = NUM_MODELS,
        reduction = 'sum',
    ).to(device)

    # ── Fountain-specific run_one closure ─────────────────────────────────────
    def run_one_fn(row, fold, args, out_dir):
        return run_one_fountain(row, fold, args, models, multi_loss, device, out_dir)

    # ── Run all folds ─────────────────────────────────────────────────────────
    for fold in args.folds:
        run_fold(fold, args, run_one_fn, args.flat_regions_base, args.results_base_dir)

    log.info("All folds complete.")


if __name__ == "__main__":
    main()