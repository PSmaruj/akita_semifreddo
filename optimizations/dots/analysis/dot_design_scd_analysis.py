"""
dot_design_scd_analysis.py
------------------------
Compute Akita SCD (upper-triangle RMSD between original and designed contact
maps) for dot designs and append as a column to the existing results TSV.

Usage:
    python dot_design_scd_analysis.py --fold 0 --run_name results/dot_d50 --inter_anchor_dist 50
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))

from utils.dataset_utils import DoubleInsertionDataset, SequenceDataset
from utils.model_utils import load_model

# ── Fixed paths ────────────────────────────────────────────────────────────────

_PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

MODEL_CKPT        = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
SEQ_BASE_DIR      = f"{_PROJ}/analysis/flat_regions"
RESULTS_BASE_DIR  = f"{_PROJ}/optimizations/dots"

# ── Architecture constants ─────────────────────────────────────────────────────

CENTER_BIN_MAP = 256
CROPPING       = 64
BIN_SIZE       = 2048


# ── Helpers ────────────────────────────────────────────────────────────────────

def anchor_bp_coords(bin_lo: int, bin_hi: int) -> tuple[int, int, int, int]:
    """Convert anchor bin indices to bp coordinates in the full sequence."""
    bp_lo_start = (bin_lo + CROPPING) * BIN_SIZE
    bp_lo_end   = bp_lo_start + BIN_SIZE
    bp_hi_start = (bin_hi + CROPPING) * BIN_SIZE
    bp_hi_end   = bp_hi_start + BIN_SIZE
    return bp_lo_start, bp_lo_end, bp_hi_start, bp_hi_end


def batch_scd(preds_orig: torch.Tensor, preds_edited: torch.Tensor) -> list[float]:
    """
    Compute RMSD between original and edited Akita predictions directly on
    the upper-triangular output vectors (no conversion to full map needed).

    Parameters
    ----------
    preds_orig : torch.Tensor
        Batch of original upper-triu predictions, shape [B, T].
    preds_edited : torch.Tensor
        Batch of edited upper-triu predictions, shape [B, T].

    Returns
    -------
    list of float
        RMSD per locus in the batch.
    """
    diff = (preds_edited - preds_orig).numpy()
    return [float(np.sqrt(np.nanmean(row ** 2))) for row in diff]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute Akita SCD for dot designs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fold",              type=int, required=True)
    p.add_argument("--run_name",          type=str, required=True)
    p.add_argument("--inter_anchor_dist", type=int, required=True)
    p.add_argument("--batch_size",        type=int, default=4)
    p.add_argument("--results_base_dir",  default=RESULTS_BASE_DIR)
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    fold   = args.fold
    d      = args.inter_anchor_dist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    half_dist = d // 2
    bin_lo    = CENTER_BIN_MAP - half_dist
    bin_hi    = CENTER_BIN_MAP + half_dist
    bp_lo_start, bp_lo_end, bp_hi_start, bp_hi_end = anchor_bp_coords(bin_lo, bin_hi)

    run_dir  = os.path.join(args.results_base_dir, args.run_name)
    fold_dir = os.path.join(run_dir, f"fold{fold}")
    tsv_path     = os.path.join(
        run_dir,
        f"fold{fold}_selected_genomic_windows_centered_chrom_states_results.tsv",
    )
    out_tsv_path = os.path.join(
        run_dir,
        f"fold{fold}_dot_scd_akita_results.tsv",
    )

    print(f"Device            : {device}")
    print(f"Run dir           : {run_dir}")
    print(f"Inter-anchor dist : {d}  →  bins {bin_lo} / {bin_hi}")

    df = pd.read_csv(tsv_path, sep="\t")
    print(f"Loaded {len(df)} windows from {tsv_path}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = load_model(MODEL_CKPT, device)

    # ── Datasets ───────────────────────────────────────────────────────────────
    seq_path = f"{SEQ_BASE_DIR}/mouse_sequences/fold{fold}/"

    orig_dataset   = SequenceDataset(df, seq_path, "chrom", "centered_start", "centered_end", "X")
    edited_dataset = DoubleInsertionDataset(
        df, seq_path, fold_dir + "/",
        bp_lo_start, bp_lo_end, bp_hi_start, bp_hi_end,
    )

    orig_loader   = DataLoader(orig_dataset,   batch_size=args.batch_size, shuffle=False)
    edited_loader = DataLoader(edited_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Prediction loop ────────────────────────────────────────────────────────
    all_scds = []
    with torch.no_grad():
        for i, (orig_b, edited_b) in enumerate(zip(orig_loader, edited_loader)):
            print(f"  Batch {i}", flush=True)
            preds_orig   = model(orig_b.to(device)).cpu()
            preds_edited = model(edited_b.to(device)).cpu()
            all_scds.extend(batch_scd(preds_orig, preds_edited))

    # ── Append and save ────────────────────────────────────────────────────────
    df["akita_scd"] = all_scds
    df.to_csv(out_tsv_path, sep="\t", index=False)
    print(f"Saved → {out_tsv_path}")


if __name__ == "__main__":
    main()