"""
dot_design_analysis.py

Analyse dot optimisation results: predicted dot scores at multiple window sizes,
GC content, and CTCF motif hits for original, edited, and target sequences.

Usage:
python dot_design_analysis.py \
    --fold 7 \
    --run_name results/dot_d70 \
    --inter_anchor_dist 70
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
sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))

from utils.dataset_utils import DoubleInsertionDataset, SequenceDataset, TriuMatrixDataset
from utils.data_utils import from_upper_triu_batch, gc_content
from utils.fimo_utils import read_meme_pwm, ctcf_hits_per_seq
from utils.model_utils import load_model
from utils.scores_utils import compute_dot_scores

# ── Fixed paths ───────────────────────────────────────────────────────────────
_PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

MODEL_CKPT        = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
SEQ_BASE_DIR      = f"{_PROJ}/analysis/flat_regions"
TARGET_BASE_DIR   = f"{_PROJ}/optimizations/dots/targets"
RESULTS_BASE_DIR  = f"{_PROJ}/optimizations/dots"
FLAT_REGIONS_BASE = f"{_PROJ}/analysis/flat_regions/mouse_flat_regions_chrom_states_tsv"
PWM_PATH          = "/home1/smaruj/ledidi_akita/data/pwm/MA0139.1.meme"

# ── Architecture constants ────────────────────────────────────────────────────
CENTER_BIN_MAP = 256
CROPPING       = 64
BIN_SIZE       = 2048
EXTRA_FLANK    = 60   # extra bp around each anchor bin for FIMO

# ── Dot score half-widths to measure (in bins) ────────────────────────────────
# dot-7: half=3, dot-11: half=5, dot-15: half=7
DOT_HALF_WIDTHS = [3, 5, 7]


# ── Helpers ───────────────────────────────────────────────────────────────────

def anchor_bp_coords(bin_lo: int, bin_hi: int) -> tuple[int, int, int, int]:
    """Convert anchor bin indices to bp coordinates in the full sequence."""
    bp_lo_start = (bin_lo + CROPPING) * BIN_SIZE
    bp_lo_end   = bp_lo_start + BIN_SIZE
    bp_hi_start = (bin_hi + CROPPING) * BIN_SIZE
    bp_hi_end   = bp_hi_start + BIN_SIZE
    return bp_lo_start, bp_lo_end, bp_hi_start, bp_hi_end


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dot optimisation analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fold",              type=int, required=True)
    p.add_argument("--run_name",          type=str, required=True,
                   help="Results subdirectory, e.g. 'lambda/lambda_10.0'")
    p.add_argument("--inter_anchor_dist", type=int, required=True,
                   help="Inter-anchor distance in bins (e.g. 50)")
    p.add_argument("--batch_size",        type=int, default=4)
    p.add_argument("--results_base_dir",  default=RESULTS_BASE_DIR)
    p.add_argument("--flat_regions_base", default=FLAT_REGIONS_BASE)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    fold   = args.fold
    d      = args.inter_anchor_dist
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Derive anchor positions
    half_dist   = d // 2
    bin_lo      = CENTER_BIN_MAP - half_dist
    bin_hi      = CENTER_BIN_MAP + half_dist
    bp_lo_start, bp_lo_end, bp_hi_start, bp_hi_end = anchor_bp_coords(bin_lo, bin_hi)

    # Dot centre in map coordinates
    dot_row = bin_lo   # row = lo anchor
    dot_col = bin_hi   # col = hi anchor

    run_dir  = os.path.join(args.results_base_dir, args.run_name)
    fold_dir = os.path.join(run_dir, f"fold{fold}")

    print(f"Device            : {device}")
    print(f"Run dir           : {run_dir}")
    print(f"Inter-anchor dist : {d}  →  bins {bin_lo} / {bin_hi}")
    print(f"Anchor bp ranges  : lo [{bp_lo_start:,}–{bp_lo_end:,}]  hi [{bp_hi_start:,}–{bp_hi_end:,}]")

    # ── Load opt metadata table ───────────────────────────────────────────────
    df = pd.read_csv(
        os.path.join(run_dir, f"fold{fold}_selected_genomic_windows_centered_chrom_states_opt.tsv"),
        sep="\t",
    )
    print(f"Loaded {len(df)} windows")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(MODEL_CKPT, device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    seq_path    = f"{SEQ_BASE_DIR}/mouse_sequences/fold{fold}/"
    target_path = f"{TARGET_BASE_DIR}/dot_d{d}/fold{fold}/"

    orig_dataset   = SequenceDataset(df, seq_path, "chrom", "centered_start", "centered_end", "X")
    edited_dataset = DoubleInsertionDataset(
        df, seq_path, fold_dir + "/",
        bp_lo_start, bp_lo_end, bp_hi_start, bp_hi_end,
    )
    target_dataset = TriuMatrixDataset(df, target_path, "chrom", "centered_start", "centered_end", "target")

    orig_loader   = DataLoader(orig_dataset,   batch_size=args.batch_size, shuffle=False)
    edited_loader = DataLoader(edited_dataset, batch_size=args.batch_size, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Storage ───────────────────────────────────────────────────────────────
    dot_scores = {
        f"dot{2*hw+1}_{cond}": []
        for hw in DOT_HALF_WIDTHS
        for cond in ("orig", "edited", "target")
    }
    gc_seq, gc_lo_orig, gc_lo_edited, gc_hi_orig, gc_hi_edited = [], [], [], [], []

    # ── Prediction loop ───────────────────────────────────────────────────────
    with torch.no_grad():
        for orig_b, edited_b, target_b in zip(orig_loader, edited_loader, target_loader):
            orig_b   = orig_b.to(device)
            edited_b = edited_b.to(device)
            target_b = target_b.to(device).squeeze(1)

            # GC content — whole sequence, lo anchor, hi anchor
            gc_seq.extend(gc_content(orig_b).tolist())
            gc_lo_orig.extend(gc_content(orig_b,   bp_lo_start, bp_lo_end).tolist())
            gc_hi_orig.extend(gc_content(orig_b,   bp_hi_start, bp_hi_end).tolist())
            gc_lo_edited.extend(gc_content(edited_b, bp_lo_start, bp_lo_end).tolist())
            gc_hi_edited.extend(gc_content(edited_b, bp_hi_start, bp_hi_end).tolist())

            # Predictions → contact maps
            maps_orig   = from_upper_triu_batch(model(orig_b).cpu())
            maps_edited = from_upper_triu_batch(model(edited_b).cpu())
            maps_target = from_upper_triu_batch(target_b.cpu())

            # Dot scores at all window sizes
            for hw in DOT_HALF_WIDTHS:
                size = 2 * hw + 1
                for maps, cond in ((maps_orig, "orig"), (maps_edited, "edited"), (maps_target, "target")):
                    scores = compute_dot_scores(maps, dot_row, dot_col, [hw])
                    dot_scores[f"dot{size}_{cond}"].extend(scores[f"dot{size}"])

    # ── CTCF loop — scan both anchor bins separately ──────────────────────────
    pwm    = read_meme_pwm(PWM_PATH)
    motifs = {"CTCF": pwm}

    orig_n_ctcf_lo,  orig_n_ctcf_hi  = [], []
    ctcf_records_lo, ctcf_records_hi = [], []

    with torch.no_grad():
        for orig_b, edited_b in zip(orig_loader, edited_loader):
            bs = orig_b.shape[0]

            for bp_start, bp_end, orig_list, edited_list in (
                (bp_lo_start, bp_lo_end, orig_n_ctcf_lo, ctcf_records_lo),
                (bp_hi_start, bp_hi_end, orig_n_ctcf_hi, ctcf_records_hi),
            ):
                orig_slice   = orig_b[:,   :, bp_start - EXTRA_FLANK : bp_end + EXTRA_FLANK].cpu().numpy()
                edited_slice = edited_b[:, :, bp_start - EXTRA_FLANK : bp_end + EXTRA_FLANK].cpu().numpy()

                orig_hits   = fimo(motifs=motifs, sequences=orig_slice,
                                   threshold=1e-4, reverse_complement=True)[0]
                edited_hits = fimo(motifs=motifs, sequences=edited_slice,
                                   threshold=1e-4, reverse_complement=True)[0]

                for hits in (orig_hits, edited_hits):
                    hits["start"] -= EXTRA_FLANK
                    hits["end"]   -= EXTRA_FLANK
                    hits["sequence_name"] = hits["sequence_name"].astype(int)

                for seq_idx in range(bs):
                    oh = orig_hits[orig_hits["sequence_name"] == seq_idx]
                    orig_list.append(len(oh) if not oh.empty else 0)

                edited_list.extend(ctcf_hits_per_seq(edited_hits, bs))

    # ── Assemble results ──────────────────────────────────────────────────────
    for col, vals in dot_scores.items():
        df[col] = vals

    df["GC_seq"]       = gc_seq
    df["GC_lo_orig"]   = gc_lo_orig
    df["GC_lo_edited"] = gc_lo_edited
    df["GC_hi_orig"]   = gc_hi_orig
    df["GC_hi_edited"] = gc_hi_edited

    for suffix, orig_list, edited_list in (
        ("lo", orig_n_ctcf_lo, ctcf_records_lo),
        ("hi", orig_n_ctcf_hi, ctcf_records_hi),
    ):
        df[f"init_CTCFs_num_{suffix}"] = orig_list
        df[f"CTCFs_num_{suffix}"]      = [r["n"]         for r in edited_list]
        df[f"FIMO_sum_{suffix}"]       = [r["score_sum"] for r in edited_list]
        df[f"FIMO_max_{suffix}"]       = [r["score_max"] for r in edited_list]
        df[f"orientation_{suffix}"]    = [r["strands"]   for r in edited_list]
        df[f"positions_{suffix}"]      = [r["positions"] for r in edited_list]

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(
        run_dir,
        f"fold{fold}_selected_genomic_windows_centered_chrom_states_results.tsv",
    )
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()