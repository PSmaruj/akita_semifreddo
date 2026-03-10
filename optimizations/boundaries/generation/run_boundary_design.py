"""
run_boundary_design.py

Batch boundary optimisation over all genomic windows in a fold's flat-regions table.
Saves per-window generated sequences and an enriched TSV with optimisation metadata.

Usage:
    python run_boundary_design.py \\
        --fold 0 \\
        --run_name lambda/lambda_0.01 \\
        --boundary_strength -0.5 \\
        --L 0.01 \\
"""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, "/home1/smaruj/ledidi/ledidi/")
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))

from akita.model import SeqNN
from ledidi import Ledidi
from semifreddo.semifreddo import SemifreddoLedidiWrapper
from utils.losses_utils import LocalL1Loss
from utils.optimization_utils import strength_tag


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

# ── Fixed optimisation settings ───────────────────────────────────────────────
MAX_ITER       = 2000
EARLY_STOPPING = 2000

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_stem(chrom: str, start: int, end: int) -> str:
    return f"{chrom}_{start}_{end}"


def last_accepted_step(history: dict) -> int:
    """Return the index of the last iteration that introduced a new edited position.

    Each entry in history["edits"] is a tuple of three tensors:
        (batch_indices, nucleotide_indices, position_indices)
    We track the cumulative set of positions seen across steps 0..i-1 and
    return the last i where history["edits"][i][2] contains a position not
    previously seen.
    """
    edits = history["edits"]
    seen  = set()
    last  = 0
    for i, edit in enumerate(edits):
        positions = set(edit[2].cpu().tolist())
        if positions - seen:
            last = i
            seen |= positions
    return last


def count_edits(original_X: torch.Tensor, generated_full: torch.Tensor) -> int:
    """Number of nucleotide positions that differ between original and generated."""
    return int(
        (torch.argmax(generated_full, dim=1) != torch.argmax(original_X, dim=1))
        .sum()
        .item()
    )


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    model = SeqNN()
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    log.info("Model loaded")
    return model


def run_one(
    row: pd.Series,
    fold: int,
    tag: str,
    args: argparse.Namespace,
    model: torch.nn.Module,
    boundary_mask: torch.Tensor,
    device: torch.device,
    out_dir: str,
) -> dict:
    """Run boundary optimisation for a single genomic window.

    Returns a dict of metadata columns to be added to the row.
    """
    chrom = row["chrom"]
    start = int(row["centered_start"])
    end   = int(row["centered_end"])
    stem  = build_stem(chrom, start, end)

    log.info(f"  Window: {stem}")

    # ── Load tensors ──────────────────────────────────────────────────────────
    X = torch.load(
        f"{args.seq_base_dir}/mouse_sequences/fold{fold}/{stem}_X.pt",
        weights_only=True,
    ).to(device)
    tower = torch.load(
        f"{args.seq_base_dir}/mouse_tower_outputs/fold{fold}/{stem}_tower_out.pt",
        weights_only=True,
    ).to(device)
    target = torch.load(
        f"{args.target_base_dir}/boundary_{tag}/fold{fold}/{stem}_target.pt",
        weights_only=True,
    ).to(device)

    # ── Semifreddo wrapper ────────────────────────────────────────────────────
    sf_wrapper = SemifreddoLedidiWrapper(
        model                   = model,
        precomputed_full_output = tower,
        full_X                  = X,
        edited_bin              = CENTER_BIN_MAP,
        context_bins            = CONTEXT_BINS,
        cropping_applied        = CROPPING_APPLIED,
    )

    X_center = X[:, :, sf_wrapper.center_bp_start:sf_wrapper.center_bp_end]

    # ── Loss ──────────────────────────────────────────────────────────────────
    local_output_loss = LocalL1Loss(
        boundary_mask, n_triu=N_TRIU, reduction="sum"
    ).to(device)

    # ── Optimise ──────────────────────────────────────────────────────────────
    ledidi_optimizer = Ledidi(
        sf_wrapper,
        shape               = X_center.shape[1:],
        input_loss          = torch.nn.L1Loss(reduction="sum"),
        output_loss         = local_output_loss,
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

    # ── Reconstruct full sequence ─────────────────────────────────────────────
    full_generated_seq = X.clone()
    full_generated_seq[
        :, :, sf_wrapper.center_bp_start:sf_wrapper.center_bp_end
    ] = generated_seq

    # ── Metrics ───────────────────────────────────────────────────────────────
    n_edits   = count_edits(X, full_generated_seq)
    last_step = last_accepted_step(history)
    log.info(f"    Edits: {n_edits}  |  Last accepted step: {last_step}")

    # ── Save generated sequence ───────────────────────────────────────────────
    out_path = os.path.join(out_dir, f"{stem}_gen_seq.pt")
    torch.save(generated_seq.cpu(), out_path)
    log.info(f"    Saved → {out_path}")

    return {"n_edits": n_edits, "last_accepted_step": last_step}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch boundary optimisation by fold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--fold",     type=int,   required=True,
                   help="Fold index (e.g. 0)")
    p.add_argument("--run_name", type=str,   required=True,
                   help="Results subdirectory name (e.g. 'results/boundary_neg0p5', 'lambda/lambda_0.01')")
    p.add_argument("--boundary_strength", type=float, required=True,
                   help="Value applied to the off-diagonal quadrants of the boundary mask "
                        "(e.g. -0.5). Negative values suppress contacts.")

    # Ledidi hyperparameters
    p.add_argument("--L",   type=float, default=0.01,  help="Input-loss regularisation weight")
    p.add_argument("--tau", type=float, default=1.0,   help="Ledidi tau parameter")
    p.add_argument("--eps", type=float, default=1e-4,  help="Ledidi eps parameter")

    # Paths (all have defaults)
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
    fold   = args.fold
    tag    = strength_tag(args.boundary_strength)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log.info(f"Device: {device}  |  Fold: {fold}  |  Run: {args.run_name}")
    log.info(f"Boundary strength: {args.boundary_strength} (tag: {tag})")
    log.info(f"Ledidi params — L={args.L}  tau={args.tau}  eps={args.eps}")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = os.path.join(args.results_base_dir, args.run_name, f"fold{fold}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Shared resources ──────────────────────────────────────────────────────
    model         = load_model(args.model_path, device)
    boundary_mask = torch.load(args.mask_path, weights_only=True).to(device)

    # ── Flat-regions table ────────────────────────────────────────────────────
    tsv_path = (
        f"{args.flat_regions_base}/"
        f"fold{fold}_selected_genomic_windows_centered_chrom_states.tsv"
    )
    df = pd.read_csv(tsv_path, sep="\t")
    log.info(f"Loaded {len(df)} windows from {tsv_path}")

    # ── Run optimisations ─────────────────────────────────────────────────────
    results = []
    for i, row in df.iterrows():
        log.info(f"[{i + 1}/{len(df)}]")
        try:
            meta = run_one(row, fold, tag, args, model, boundary_mask, device, out_dir)
        except Exception as e:
            log.error(f"  FAILED: {e}")
            meta = {"n_edits": np.nan, "last_accepted_step": np.nan}
        results.append(meta)

    # ── Enrich and save table ─────────────────────────────────────────────────
    df_out  = pd.concat([df, pd.DataFrame(results, index=df.index)], axis=1)
    out_tsv = os.path.join(
        os.path.dirname(out_dir),
        f"fold{fold}_selected_genomic_windows_centered_chrom_states_opt.tsv",
    )
    df_out.to_csv(out_tsv, sep="\t", index=False)
    log.info(f"Saved enriched table → {out_tsv}")


if __name__ == "__main__":
    main()