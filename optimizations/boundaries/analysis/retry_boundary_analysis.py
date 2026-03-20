"""
retry_boundary_analysis.py

Analyse boundary optimisation results for a subset of windows supplied via
--retry_tsv (e.g. windows that failed in a previous run).  Computes predicted
insulation scores, GC content, and CTCF motif hits and appends them to the TSV.

Usage:
    python retry_boundary_analysis.py \
        --retry_tsv /project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita/optimizations/boundaries/unsuccessful_all_folds_-0.5.tsv \
        --run_name rerun_unsuccessful \
        --boundary_strength -0.5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from memelite import fimo

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/ledidi_akita/"))

from utils.dataset_utils import CentralInsertionDataset, SequenceDataset, TriuMatrixDataset
from utils.data_utils import from_upper_triu_batch, gc_content
from utils.fimo_utils import read_meme_pwm, ctcf_hits_per_seq
from utils.optimization_utils import strength_tag
from utils.model_utils import load_model
from utils.scores_utils import insulation_score

# ── Fixed paths ───────────────────────────────────────────────────────────────
_PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

MODEL_CKPT        = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
SEQ_BASE_DIR      = f"{_PROJ}/analysis/flat_regions"
TARGET_BASE_DIR   = f"{_PROJ}/optimizations/boundaries/targets"
RESULTS_BASE_DIR  = f"{_PROJ}/optimizations/boundaries"
PWM_PATH          = "/home1/smaruj/ledidi_akita/data/pwm/MA0139.1.meme"

# ── Architecture constants ────────────────────────────────────────────────────
CENTER_BIN_MAP = 256
CROPPING       = 64
BIN_SIZE       = 2048
EDIT_START     = (CENTER_BIN_MAP + CROPPING) * BIN_SIZE
EDIT_END       = EDIT_START + BIN_SIZE
EXTRA_FLANK    = 60

# ── URQ slice ─────────────────────────────────────────────────────────────────
URQ_ROW_SLICE = slice(0, 250)
URQ_COL_SLICE = slice(260, 512)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Boundary analysis for a retry subset of windows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--retry_tsv",         type=str,   required=True,
                   help="TSV of unsuccessful windows; must have chrom, centered_start, "
                        "centered_end, and fold columns.")
    p.add_argument("--run_name",          type=str,   required=True,
                   help="Results subdirectory containing the retry optimisation outputs.")
    p.add_argument("--boundary_strength", type=float, required=True)
    p.add_argument("--batch_size",        type=int,   default=4)
    p.add_argument("--results_base_dir",  default=RESULTS_BASE_DIR)
    return p.parse_args()


# ── Per-fold analysis ─────────────────────────────────────────────────────────

def analyse_fold(fold, fold_df, args, tag, model, pwm, device):
    """
    Run insulation score and CTCF analysis for one fold's subset of windows.

    Parameters
    ----------
    fold : int
    fold_df : pd.DataFrame  rows from the retry TSV belonging to this fold.
    args : argparse.Namespace
    tag : str  boundary strength tag (e.g. 'neg0.5').
    model : loaded Akita model.
    pwm : CTCF PWM.
    device : torch.device.

    Returns
    -------
    pd.DataFrame  fold_df with analysis columns appended.
    """
    fold_df = fold_df.reset_index(drop=True)
    fold_dir = os.path.join(args.results_base_dir, args.run_name, f"fold{fold}")
    print(f"\nFold {fold}: {len(fold_df)} windows  →  {fold_dir}")

    seq_path    = f"{SEQ_BASE_DIR}/mouse_sequences/fold{fold}/"
    target_path = f"{TARGET_BASE_DIR}/boundary_{tag}/fold{fold}/"

    orig_dataset   = SequenceDataset(fold_df, seq_path, "chrom", "centered_start", "centered_end", "X")
    edited_dataset = CentralInsertionDataset(fold_df, seq_path, fold_dir + "/", EDIT_START, EDIT_END)
    target_dataset = TriuMatrixDataset(fold_df, target_path, "chrom", "centered_start", "centered_end", "target")

    orig_loader   = DataLoader(orig_dataset,   batch_size=args.batch_size, shuffle=False)
    edited_loader = DataLoader(edited_dataset, batch_size=args.batch_size, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)

    urq_orig, urq_edited, urq_target = [], [], []
    gc_seq, gc_slice_orig, gc_slice_edited = [], [], []

    with torch.no_grad():
        for orig_b, edited_b, target_b in zip(orig_loader, edited_loader, target_loader):
            orig_b   = orig_b.to(device)
            edited_b = edited_b.to(device)
            target_b = target_b.to(device).squeeze(1)

            gc_seq.extend(gc_content(orig_b).tolist())
            gc_slice_orig.extend(gc_content(orig_b, EDIT_START, EDIT_END).tolist())
            gc_slice_edited.extend(gc_content(edited_b, EDIT_START, EDIT_END).tolist())

            maps_orig   = from_upper_triu_batch(model(orig_b).cpu())
            maps_edited = from_upper_triu_batch(model(edited_b).cpu())
            maps_target = from_upper_triu_batch(target_b.cpu())

            urq_orig.extend(insulation_score(maps_orig, URQ_ROW_SLICE, URQ_COL_SLICE))
            urq_edited.extend(insulation_score(maps_edited, URQ_ROW_SLICE, URQ_COL_SLICE))
            urq_target.extend(insulation_score(maps_target, URQ_ROW_SLICE, URQ_COL_SLICE))

    motifs       = {"CTCF": pwm}
    orig_n_ctcf  = []
    ctcf_records = []

    with torch.no_grad():
        for orig_b, edited_b in zip(orig_loader, edited_loader):
            bs = orig_b.shape[0]

            orig_slice   = orig_b[:,   :, EDIT_START - EXTRA_FLANK : EDIT_END + EXTRA_FLANK].cpu().numpy()
            edited_slice = edited_b[:, :, EDIT_START - EXTRA_FLANK : EDIT_END + EXTRA_FLANK].cpu().numpy()

            orig_hits   = fimo(motifs=motifs, sequences=orig_slice,
                               threshold=1e-4, reverse_complement=True)[0]
            edited_hits = fimo(motifs=motifs, sequences=edited_slice,
                               threshold=1e-4, reverse_complement=True)[0]

            for hits in (orig_hits, edited_hits):
                hits["start"] -= EXTRA_FLANK
                hits["end"]   -= EXTRA_FLANK

            for seq_idx in range(bs):
                oh = orig_hits[orig_hits["sequence_name"] == seq_idx]
                orig_n_ctcf.append(len(oh) if not oh.empty else 0)

            ctcf_records.extend(ctcf_hits_per_seq(edited_hits, bs))

    fold_df["insul_score_orig"]    = urq_orig
    fold_df["insul_score_edited"]  = urq_edited
    fold_df["insul_score_target"]  = urq_target
    fold_df["GC_seq"]              = gc_seq
    fold_df["GC_slice_orig"]       = gc_slice_orig
    fold_df["GC_slice_edited"]     = gc_slice_edited
    fold_df["init_CTCFs_num"]      = orig_n_ctcf
    fold_df["CTCFs_num"]           = [r["n"]         for r in ctcf_records]
    fold_df["FIMO_sum"]            = [r["score_sum"] for r in ctcf_records]
    fold_df["FIMO_max"]            = [r["score_max"] for r in ctcf_records]
    fold_df["orientation"]         = [r["strands"]   for r in ctcf_records]
    fold_df["positions"]           = [r["positions"] for r in ctcf_records]

    return fold_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    tag    = strength_tag(args.boundary_strength)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Device     : {device}")
    print(f"Run name   : {args.run_name}")
    print(f"Retry TSV  : {args.retry_tsv}")

    retry_df = pd.read_csv(args.retry_tsv, sep="\t")
    required = {"chrom", "centered_start", "centered_end", "fold"}
    missing  = required - set(retry_df.columns)
    if missing:
        raise ValueError(f"retry_tsv is missing columns: {missing}")
    print(f"Loaded {len(retry_df)} windows across folds {sorted(retry_df['fold'].unique().tolist())}")

    model = load_model(MODEL_CKPT, device)
    pwm   = read_meme_pwm(PWM_PATH)

    results = []
    for fold, fold_df in retry_df.groupby("fold"):
        results.append(analyse_fold(fold, fold_df, args, tag, model, pwm, device))

    out_df   = pd.concat(results, ignore_index=True)
    out_path = os.path.join(
        args.results_base_dir, args.run_name,
        f"retry_results_{tag}.tsv",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved → {out_path}")

    # ── Quick success summary ─────────────────────────────────────────────────
    out_df["insul_score_diff"]     = out_df["insul_score_edited"] - out_df["insul_score_orig"]
    out_df["optimization_success"] = out_df["insul_score_diff"] < 0
    print(f"Success rate: {out_df['optimization_success'].mean():.1%} "
          f"({out_df['optimization_success'].sum()} / {len(out_df)})")


if __name__ == "__main__":
    main()