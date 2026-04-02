"""
boundary_design_shuffled_control.py

Compute insulation scores for dinucleotide-shuffled versions of the optimised
central bin, as a null control for the boundary optimisation results.

For each successfully optimised sequence, the optimised 2,048 bp central bin
is replaced by a dinucleotide-preserving shuffle (via seqpro), and the full
Akita v2 model is run to obtain a predicted insulation score. All other parts
of the sequence are kept identical to the optimised sequence.

Usage:
    python boundary_design_shuffled_control.py \
        --fold 7 \
        --run_name results \
        --boundary_strength -0.2
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import seqpro as sp
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath("/home1/smaruj/akita_pytorch/"))
sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))

from semifreddo.optimization_loop import strength_tag
from utils.dataset_utils import ShuffledCentralInsertionDataset
from utils.data_utils import from_upper_triu_batch
from utils.model_utils import load_model
from utils.scores_utils import compute_insulation_scores

# ── Fixed paths ───────────────────────────────────────────────────────────────
_PROJ = "/project2/fudenber_735/smaruj/sequence_design/ledidi_semifreddo_akita"

MODEL_CKPT        = (
    "/home1/smaruj/akita_pytorch/models/finetuned/mouse/Hsieh2019_mESC"
    "/checkpoints/Akita_v2_mouse_Hsieh2019_mESC_model0_finetuned.pth"
)
SEQ_BASE_DIR      = f"{_PROJ}/analysis/flat_regions"
RESULTS_BASE_DIR  = f"{_PROJ}/optimizations/boundaries_no_ctcf"

# ── Architecture constants ────────────────────────────────────────────────────
CENTER_BIN_MAP = 256
CROPPING       = 64
BIN_SIZE       = 2048
EDIT_START     = (CENTER_BIN_MAP + CROPPING) * BIN_SIZE
EDIT_END       = EDIT_START + BIN_SIZE

# ── URQ slice ─────────────────────────────────────────────────────────────────
URQ_ROW_SLICE = slice(0, 250)
URQ_COL_SLICE = slice(260, 512)

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shuffled dinucleotide control for boundary optimisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fold",              type=int,   required=True)
    p.add_argument("--run_name",          type=str,   required=True)
    p.add_argument("--boundary_strength", type=float, required=True)
    p.add_argument("--batch_size",        type=int,   default=4)
    p.add_argument("--results_base_dir",  default=RESULTS_BASE_DIR)

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    fold   = args.fold
    tag    = strength_tag(args.boundary_strength)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_dir  = os.path.join(args.results_base_dir, args.run_name)
    fold_dir = os.path.join(run_dir, f"fold{fold}")

    print(f"Device  : {device}")
    print(f"Run dir : {run_dir}")
    print(f"Fold    : {fold}")

    # ── Load results table ────────────────────────────────────────────────────
    results_tsv = os.path.join(
        run_dir,
        f"fold{fold}_selected_genomic_windows_centered_chrom_states_results.tsv",
    )
    df = pd.read_csv(results_tsv, sep="\t")
    print(f"Loaded {len(df)} windows")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(MODEL_CKPT, device)

    # ── Dataset & loader ──────────────────────────────────────────────────────
    seq_path = f"{SEQ_BASE_DIR}/mouse_sequences/fold{fold}/"

    shuffled_dataset = ShuffledCentralInsertionDataset(
        coord_df   = df,
        seq_path   = seq_path,
        slice_path = fold_dir + "/",
        edit_start = EDIT_START,
        edit_end   = EDIT_END,
    )
    shuffled_loader = DataLoader(shuffled_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Prediction loop ───────────────────────────────────────────────────────
    urq_shuffled = []

    with torch.no_grad():
        for batch in shuffled_loader:
            batch      = batch.to(device)
            maps       = from_upper_triu_batch(model(batch).cpu())
            urq_shuffled.extend(compute_insulation_scores(maps, URQ_ROW_SLICE, URQ_COL_SLICE))

    # ── Save ──────────────────────────────────────────────────────────────────
    df["insul_score_shuffled"] = urq_shuffled

    out_path = os.path.join(
        run_dir,
        f"fold{fold}_selected_genomic_windows_centered_chrom_states_shuffled_control.tsv",
    )
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()