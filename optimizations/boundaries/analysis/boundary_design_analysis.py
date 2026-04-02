"""
boundary_design_analysis.py

Analyse boundary optimisation results: predicted insulation scores (URQ),
GC content, and CTCF motif hits for original, edited, and target sequences.

Usage:
python boundary_design_analysis.py \
    --fold 3 \
    --run_name indep_runs_lambda_0.01/seed9 \
    --boundary_strength -0.5
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from memelite import fimo

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))

from semifreddo.optimization_loop import strength_tag
from utils.dataset_utils import CentralInsertionDataset, SequenceDataset, TriuMatrixDataset
from utils.data_utils import from_upper_triu_batch, gc_content
from utils.fimo_utils import read_meme_pwm, ctcf_hits_per_seq
from utils.model_utils import load_model
from utils.scores_utils import compute_insulation_scores

# ── Fixed paths ───────────────────────────────────────────────────────────────
_PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

MODEL_CKPT        = (
    "/home1/smaruj/pytorch_akita/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
SEQ_BASE_DIR      = f"{_PROJ}/analysis/flat_regions"
TARGET_BASE_DIR   = f"{_PROJ}/optimizations/boundaries/targets"
RESULTS_BASE_DIR  = f"{_PROJ}/optimizations/boundaries"
FLAT_REGIONS_BASE = f"{_PROJ}/analysis/flat_regions/mouse_flat_regions_chrom_states_tsv"
PWM_PATH          = "/home1/smaruj/ledidi_akita/data/pwm/MA0139.1.meme"

# ── Architecture constants ────────────────────────────────────────────────────
CENTER_BIN_MAP = 256
CROPPING       = 64
BIN_SIZE       = 2048
EDIT_START     = (CENTER_BIN_MAP + CROPPING) * BIN_SIZE   # bp start of central bin
EDIT_END       = EDIT_START + BIN_SIZE                    # bp end   of central bin
EXTRA_FLANK    = 60                                       # extra bp around bin for FIMO

# ── URQ slice ─────────────────────────────────────────────────────────────────
URQ_ROW_SLICE = slice(0, 250)
URQ_COL_SLICE = slice(260, 512)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Boundary optimisation analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fold",              type=int,   required=True)
    p.add_argument("--run_name",          type=str,   required=True,
                   help="Results subdirectory, e.g. 'lambda/lambda_0.01' or 'tau/tau_1.0'")
    p.add_argument("--boundary_strength", type=float, required=True,
                   help="Boundary strength value used during optimisation (e.g. -0.5)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--results_base_dir",  default=RESULTS_BASE_DIR)
    p.add_argument("--flat_regions_base", default=FLAT_REGIONS_BASE)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    fold   = args.fold
    tag    = strength_tag(args.boundary_strength)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_dir  = os.path.join(args.results_base_dir, args.run_name)
    fold_dir = os.path.join(run_dir, f"fold{fold}")

    print(f"Device    : {device}")
    print(f"Run dir   : {run_dir}")
    print(f"Fold dir  : {fold_dir}")
    print(f"Tag       : {tag}")

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
    target_path = f"{TARGET_BASE_DIR}/boundary_{tag}/fold{fold}/"

    orig_dataset   = SequenceDataset(df, seq_path, "chrom", "centered_start", "centered_end", "X")
    edited_dataset = CentralInsertionDataset(df, seq_path, fold_dir + "/", EDIT_START, EDIT_END)
    target_dataset = TriuMatrixDataset(df, target_path, "chrom", "centered_start", "centered_end", "target")

    orig_loader   = DataLoader(orig_dataset,   batch_size=args.batch_size, shuffle=False)
    edited_loader = DataLoader(edited_dataset, batch_size=args.batch_size, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Storage ───────────────────────────────────────────────────────────────
    urq_orig, urq_edited, urq_target = [], [], []
    gc_seq, gc_slice_orig, gc_slice_edited = [], [], []

    # ── Prediction loop ───────────────────────────────────────────────────────
    with torch.no_grad():
        for orig_b, edited_b, target_b in zip(orig_loader, edited_loader, target_loader):
            orig_b   = orig_b.to(device)
            edited_b = edited_b.to(device)
            target_b = target_b.to(device).squeeze(1)

            # GC content
            gc_seq.extend(gc_content(orig_b).tolist())
            gc_slice_orig.extend(
                gc_content(orig_b, EDIT_START, EDIT_END).tolist()
            )
            gc_slice_edited.extend(
                gc_content(edited_b, EDIT_START, EDIT_END).tolist()
            )

            # Predictions
            maps_orig   = from_upper_triu_batch(model(orig_b).cpu())
            maps_edited = from_upper_triu_batch(model(edited_b).cpu())
            maps_target = from_upper_triu_batch(target_b.cpu())

            urq_orig.extend(compute_insulation_scores(maps_orig, URQ_ROW_SLICE, URQ_COL_SLICE))
            urq_edited.extend(compute_insulation_scores(maps_edited, URQ_ROW_SLICE, URQ_COL_SLICE))
            urq_target.extend(compute_insulation_scores(maps_target, URQ_ROW_SLICE, URQ_COL_SLICE))

    # ── CTCF loop ─────────────────────────────────────────────────────────────
    pwm     = read_meme_pwm(PWM_PATH)
    motifs  = {"CTCF": pwm}

    orig_n_ctcf  = []
    ctcf_records = []   # edited: n, score_sum, score_max, positions, strands

    with torch.no_grad():
        for orig_b, edited_b in zip(orig_loader, edited_loader):
            bs = orig_b.shape[0]   # true batch size (last batch may be smaller)

            orig_slice   = orig_b[:,   :, EDIT_START - EXTRA_FLANK : EDIT_END + EXTRA_FLANK].cpu().numpy()
            edited_slice = edited_b[:, :, EDIT_START - EXTRA_FLANK : EDIT_END + EXTRA_FLANK].cpu().numpy()

            orig_hits   = fimo(motifs=motifs, sequences=orig_slice,
                               threshold=1e-4, reverse_complement=True)[0]
            edited_hits = fimo(motifs=motifs, sequences=edited_slice,
                               threshold=1e-4, reverse_complement=True)[0]

            # Adjust positions back to central-bin coordinates
            for hits in (orig_hits, edited_hits):
                hits["start"] -= EXTRA_FLANK
                hits["end"]   -= EXTRA_FLANK

            for seq_idx in range(bs):
                oh = orig_hits[orig_hits["sequence_name"] == seq_idx]
                orig_n_ctcf.append(len(oh) if not oh.empty else 0)

            ctcf_records.extend(ctcf_hits_per_seq(edited_hits, bs))

    # ── Assemble results ──────────────────────────────────────────────────────
    df["insul_score_orig"]       = urq_orig
    df["insul_score_edited"]     = urq_edited
    df["insul_score_target"]     = urq_target
    df["GC_seq"]         = gc_seq
    df["GC_slice_orig"]  = gc_slice_orig
    df["GC_slice_edited"]= gc_slice_edited
    df["init_CTCFs_num"] = orig_n_ctcf
    df["CTCFs_num"]      = [r["n"]         for r in ctcf_records]
    df["FIMO_sum"]       = [r["score_sum"] for r in ctcf_records]
    df["FIMO_max"]       = [r["score_max"] for r in ctcf_records]
    df["orientation"]    = [r["strands"]   for r in ctcf_records]
    df["positions"]      = [r["positions"] for r in ctcf_records]

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(
        run_dir,
        f"fold{fold}_selected_genomic_windows_centered_chrom_states_results.tsv",
    )
    
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()